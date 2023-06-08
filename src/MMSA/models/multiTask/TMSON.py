

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from ..subNets import BertTextEncoder

__all__ = ['TMSON']

def reparametrize(mu, logvar):
    std = logvar.div(2).exp() 
    eps = Variable(std.data.new(std.size()).normal_())
    return std*eps + mu

class Uncertain_Block(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super(Uncertain_Block, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(in_features=in_dim, out_features=128),
                                    nn.LayerNorm(128),
                                    nn.Tanh(),
                                    )

        self.fc_mu = nn.Sequential(nn.Linear(in_features=128, out_features=64), nn.Sigmoid())
        self.fc_var = nn.Sequential(nn.Linear(in_features=128, out_features=64), nn.Sigmoid())

        self.decoder = nn.Sequential(nn.Linear(in_features=64, out_features=128),
                                    nn.LayerNorm(128),
                                    nn.Tanh(),
                                    nn.Linear(in_features=128, out_features=128),
                                    nn.LayerNorm(128),
                                    nn.Tanh(),
                                    )

    def forward(self, x):
        self.org = x
        out1 = self.encoder(self.org)

        self.mu = self.fc_mu(out1)
        self.var = self.fc_var(out1)
        self.new_sample = reparametrize(self.mu, self.var)

        self.rec = self.decoder(self.new_sample)

        return self.rec, self.KL_loss(), self.get_recon_loss(), self.mu, self.var
    
    def KL_loss(self, ):
        kl_loss = -0.5 * torch.mean(torch.sum(1 + self.var - self.mu **2 - self.var.exp(), dim=-1), dim=0)
        return kl_loss

    def get_recon_loss(self, ):
        diffs = torch.add(self.org, -self.rec)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class TMSON(nn.Module):
    def __init__(self, args):
        super(TMSON, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout, bidirectional=False)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout, bidirectional=False)

        self.project_v = nn.Sequential(nn.Linear(in_features=args.video_out, out_features=128),
                                       nn.Tanh(),
                                       )
                                       
        self.project_t = nn.Sequential(nn.Linear(in_features=args.text_out, out_features=128),
                                       nn.Tanh()
                                       )
    
        self.project_a = nn.Sequential(nn.Linear(in_features=args.audio_out, out_features=128),
                                       nn.Tanh()
                                       )

        self.text_uncertain_block = Uncertain_Block()
        self.visual_uncertain_block = Uncertain_Block()
        self.acoustic_uncertain_block = Uncertain_Block()

        # the fusion layers
        self.fusion_classify_layer = nn.Sequential(nn.Dropout(p=args.post_fusion_dropout),
                                                            nn.Linear(args.text_out + args.video_out + args.audio_out + 64, args.post_fusion_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(args.post_fusion_dim, args.post_fusion_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(args.post_fusion_dim, 1))

        # the classify layer for text
        self.text_classify_layer = nn.Sequential(nn.Dropout(p=args.post_text_dropout),
                                                            nn.Linear(args.text_out, args.post_text_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(args.post_text_dim, 1))

        # the classify layer for audio
        self.audio_classify_layer = nn.Sequential(nn.Dropout(p=args.post_audio_dropout),
                                                            nn.Linear(args.audio_out, args.post_audio_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(args.post_audio_dim, 1))

        # the classify layer for video
        self.video_classify_layer = nn.Sequential(nn.Dropout(p=args.post_video_dropout),
                                                            nn.Linear(args.video_out, args.post_video_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(args.post_video_dim, 1))


    def uncertain_fusion(self, mu, var, eps = 1e-6):
        mu0, var0 = mu[0], var[0]
        mu1, var1 = mu[1], var[1]
        mu2, var2 = mu[2], var[2]

        new_mu = (mu0*var1 + mu1*var0) / torch.clip((var0 + var1), min=eps, max=1.)
        new_var = (var0*var1) / torch.clip((var0 + var1), min=eps, max=1.)

        new_mu = (new_mu*var2 + mu2*new_var) / torch.clip((new_var+ var2), min=eps, max=1.)
        new_var = (new_var*var2) / torch.clip((new_var + var2), min=eps, max=1.)

        return new_mu, new_var, reparametrize(new_mu, new_var)

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        
        f_text = self.project_t(text)
        f_visual = self.project_v(video)
        f_acoustic = self.project_a(audio)

        new_text, t_kl_loss, rec_t_loss, t_mu, t_var = self.text_uncertain_block(f_text)
        new_visual, v_kl_loss, rec_v_loss, v_mu, v_var = self.visual_uncertain_block(f_visual)
        new_acoustic, a_kl_loss, rec_a_loss, a_mu, a_var = self.acoustic_uncertain_block(f_acoustic)
        kl_loss = t_kl_loss + v_kl_loss + a_kl_loss
        rec_loss = rec_t_loss + rec_v_loss + rec_a_loss

        mu = [t_mu, v_mu, a_mu]
        var = [t_var, v_var, a_var]
        
        # Fusing uncertainty
        new_mu, new_var, new_sample = self.uncertain_fusion(mu, var)
        
        # fusion
        fusion_h = torch.cat([text, audio, video, new_sample], dim=-1)
        output_fusion = self.fusion_classify_layer(fusion_h)
        # # text
        output_text = self.text_classify_layer(text)
        # audio
        output_audio = self.audio_classify_layer(audio)
        # vision
        output_video = self.video_classify_layer(video)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'kl_loss': kl_loss,
            'rec_loss': rec_loss,
            'mu': new_mu, 
            'var': new_var,
            't_mu':t_mu,
            'v_mu':v_mu,
            'a_mu':a_mu,
            't_var': t_var,
            'v_var': v_var,
            'a_var': a_var
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        self.bidirectional = bidirectional
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        if self.bidirectional:
            y_1 = y_1.permute(1,0,2).contiguous().view(x.size(0), -1)
        return y_1
