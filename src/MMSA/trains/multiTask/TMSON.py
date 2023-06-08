import logging
import os
import pickle as plk
from cv2 import log
from importlib_metadata import metadata
from pyrsistent import v
from regex import B
import torch.nn as nn
import numpy as np
import torch
from torch import optim, rand
from tqdm import tqdm
import numpy as np
import sys
import random

from zmq import RECONNECT_IVL

sys.path.append('../../')
from utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class TMSON():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.weight = {'M':1., 'T':0.8, 'A':0.1, 'V':0.1}

        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }


    def do_train(self, model, dataloader, summary, return_epoch_results=False):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

        # init labels
        logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        num_trials = self.args.num_trials
        # loop util earlystop

        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            model.train()
            train_loss = 0.0
            total_ord = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    labels = self.label_map[self.name_map['M']][indexes]
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                    
                    # compute loss
                    kl_loss = outputs['kl_loss'].cpu() / 3
                    rec_loss = outputs['rec_loss'].cpu() / 3
                    ord_loss = Ordinal_Regression_Loss(outputs['mu'], outputs['var'], labels).cpu()
                    loss = 0.1 * rec_loss + 0.01 * kl_loss + 0.5 * ord_loss
                    for m in self.args.tasks:
                        w = self.weight[m]
                        loss += self.weighted_loss(outputs[m], self.label_map[self.name_map[m]][indexes], \
                                                    indexes=indexes, weight=w, mode=self.name_map[m]).cpu()
                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    total_ord += ord_loss.item()
                    torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], 5.0)
                    
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
           
            train_loss = train_loss / len(dataloader['train'])
            total_ord = total_ord / len(dataloader['train'])
            # summary.add_scalar('Loss', train_loss, epochs*self.args.batch_size)
            # summary.add_scalar('ord_loss', total_ord, epochs*self.args.batch_size)

            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))

            val_results, features = self.do_test(model,  dataloader['test'], summary, epochs, mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid) if min_or_max == 'min' else cur_valid >= (best_valid)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                print("Found new best model!")
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                torch.save(optimizer.state_dict(), self.args.optim_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                num_trials -= 1
           
            if num_trials <= 0:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, summary, epochs=0, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        # criterion = nn.L1Loss()
        data_id = []

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indx = batch_data['id']
                    data_id.extend(indx)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        # for item in features.keys():
                        #     features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels_m.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        # test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(preds.squeeze())
                    
                    loss = self.weighted_loss(outputs['M'], labels_m)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())

        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.model_name + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = round(eval_loss, 4)

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results, []
    
    def weighted_loss(self, y_pred, y_true, indexes=None, weight=1., mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = weight
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss
    
    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels
      

def WassersteinDistance(mu, var, index1, index2):
    u1, var1, u2, var2 = mu[index1, :], var[index1, :], mu[index2, :], var[index2, :]
    
    dis1 = torch.sum(torch.pow(u1 - u2, 2), dim=1)
    dis2 = torch.sum(torch.pow(torch.pow(var1, 0.5) - torch.pow(var2, 0.5), 2), dim=1)
    
    return torch.pow(dis1 + dis2, 0.5)

def triple_loss(dis_1, dis_2, labels, index):
    margin = 1.0
    batch_size = labels.shape[0]
    anchor_index, reference_index, hard_index= index[0], index[1], index[2]
    
    sign = torch.sign(torch.abs(labels[anchor_index] - labels[reference_index]) - torch.abs(labels[anchor_index] - labels[hard_index]))
    sign = sign.view(batch_size)

    loss_temp = (dis_1 - dis_2) * sign.float() + margin
    
    loss = torch.max(torch.zeros(1).cuda(), loss_temp) * torch.abs(sign).float()
    mask = (loss_temp > 0).to(dtype=sign.dtype)
    
    if sum(torch.abs(sign) * mask) > 0:
        ord_loss = torch.sum(loss) / sum(torch.abs(sign) * mask)
    else:
        ord_loss = torch.tensor(0.0).cuda()

    return ord_loss

def Ordinal_Regression_Loss(mu, var, labels):
    batch_size = mu.shape[0]
    var = torch.exp(var)

    # anchor sample index
    anchor_index = [i for i in range(batch_size)]

    # reference sample index
    random_id = random.randint(2, batch_size - 1)
    reference_index = [(i + random_id) % batch_size for i in anchor_index]

    # compute the distance between anchor sample and other samples
    distance = torch.abs(labels.view(-1, 1).repeat(1, batch_size) - labels.view(1, -1).repeat(batch_size, 1))

    # distance between anchor and reference sample
    dist_ar = torch.abs(labels[anchor_index] - labels[reference_index]).view(-1, 1)

    ## choose hard sample
    distance = torch.abs(distance - dist_ar.repeat(1, batch_size))
    distance = distance + 100 * torch.eye(batch_size).cuda().to(dtype=distance.dtype)
    distance[distance == 0] = 10 
    hard_index = torch.argmin(distance, dim=1)
    
    dis_ar = WassersteinDistance(mu, var, anchor_index, reference_index)
    dis_ah = WassersteinDistance(mu, var, anchor_index, hard_index)

    index = [anchor_index, reference_index, hard_index]
    ord_loss = triple_loss(dis_ah, dis_ar, labels, index)

    return ord_loss
