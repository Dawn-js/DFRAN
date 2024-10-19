import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder
from block import *
from glow import Glow
from rcan import Group

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse
    
class Flow(nn.Module):
    def __init__(self): 
        super().__init__()
        self.d_l = self.d_v1 = 512
        self.d_l1 = self.d_v11 = 1024
        self.MSE = MSE()

        self.flow_l = Glow(in_channel=self.d_l, n_flow=32, n_block=1, affine=True, conv_lu=False)
        self.flow_t = Glow(in_channel=self.d_v1, n_flow=32, n_block=1, affine=True, conv_lu=False)

        self.rec_l = nn.Sequential(
            nn.Conv1d(self.d_l, self.d_l1, 1),
            Group(num_channels=self.d_l1, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_l1, self.d_l, 1)
        )

        self.rec_t = nn.Sequential(
            nn.Conv1d(self.d_v1, self.d_v11, 1),
            Group(num_channels=self.d_v11, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_v11, self.d_v1, 1)
        )
      

    def forward(self, image, text):  
            proj_x_l = image.transpose(1, 2)
            proj_x_t = text.transpose(1, 2)
            conv_feat_l, conv_feat_t = proj_x_l, proj_x_t

            #  normalizing flow for language
            _, _, z_outs_l = self.flow_l(proj_x_l.unsqueeze(-1))
            z_l = z_outs_l

            #  normalizing flow for vision
            _, _, z_outs_t = self.flow_t(proj_x_t.unsqueeze(-1))
            z_t = z_outs_t

            proj_x_t = self.flow_t.reverse(z_l, reconstruct=True).squeeze(-1).detach()
            proj_x_t = self.rec_t(proj_x_t)
            loss_rec_t = self.MSE(proj_x_t, conv_feat_t.detach())

            proj_x_l = self.flow_l.reverse(z_t, reconstruct=True).squeeze(-1).detach()
            proj_x_l = self.rec_l(proj_x_l)
            loss_rec_i = self.MSE(proj_x_l, conv_feat_l.detach())  

            return proj_x_l.transpose(1, 2), proj_x_t.transpose(1, 2), loss_rec_t, loss_rec_i

class AVmodel(nn.Module):
    def __init__(self, model_args):
        """
        Text-Visual Model Initialization
        """
        super(AVmodel, self).__init__()
        
        # Model Hyperparameters
        self.num_heads = model_args.num_heads  # 自注意力头数，例如 4
        self.layers = model_args.layers          # Transformer 层数，例如 2
        self.attn_mask = model_args.attn_mask    # 注意力掩码
        output_dim = model_args.output_dim       # 输出维度
        self.t_dim, self.v_dim = 512, 512        # 文本和视觉特征维度
        self.d_v = 512                            # 交叉模态的特征维度
        
        # Dropout parameters
        self.attn_dropout = model_args.attn_dropout
        self.relu_dropout = model_args.relu_dropout
        self.res_dropout = model_args.res_dropout
        self.out_dropout = model_args.out_dropout
        self.embed_dropout = model_args.embed_dropout

        # Hidden layers
        self.hidden_1 = 512     #256
        self.hidden_2 = 256     #128

        # Self Attentions 
        self.t_mem = self.transformer_arch(self_type='text_self')
        self.v_mem = self.transformer_arch(self_type='visual_self')
        
        self.trans_t_mem = self.transformer_arch(self_type='text_self', scalar=False)
        self.trans_v_mem = self.transformer_arch(self_type='visual_self', scalar=False)
        
        # Cross-modal 
        self.trans_v_with_t = self.transformer_arch(self_type='visual/text', pos_emb=True)
        self.trans_t_with_v = self.transformer_arch(self_type='text/visual', pos_emb=True)
       
        # Auxiliary networks linear layers
        self.proj_aux1 = nn.Linear(self.d_v, self.hidden_2)
        self.proj_aux2 = nn.Linear(self.hidden_2, self.hidden_1)
        self.proj_aux3 = nn.Linear(self.hidden_1, self.d_v)
        self.out_layer_aux = nn.Linear(self.d_v, output_dim)

        # Linear layers
        combined_dim = 2 * self.d_v
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def transformer_arch(self, self_type='text/visual', scalar=False, pos_emb=False):
        if self_type == 'visual/text':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'text/visual':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'text_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout    
        elif self_type == 'visual_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Not a valid network")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  scalar=scalar,
                                  pos_emb=pos_emb)
    
    def forward(self, text, visual, training = False):
        """
        Text and visual inputs should have dimension [batch_size, seq_len, n_features]
        """     
        text = text.transpose(1, 2)  # Change to [batch_size, n_features, seq_len]
        visual = visual.transpose(1, 2)  # Change to [batch_size, n_features, seq_len]

        # 1-D Convolution visual/text features
        # proj_t_v = text if self.t_dim == self.d_v else self.conv_1d_t(text)   可以考虑添加
        proj_x_t = text.permute(2, 0, 1)  # Change to [seq_len, batch_size, n_features]
        proj_x_v = visual.permute(2, 0, 1)     # Change to [seq_len, batch_size, n_features]

        # Text/Visual
        h_tv = self.trans_t_with_v(proj_x_t, proj_x_v, proj_x_v)
        h_ts = self.trans_t_mem(h_tv)
        representation_text = h_ts[-1]

        # Visual/Text
        h_vt = self.trans_v_with_t(proj_x_v, proj_x_t, proj_x_t)
        h_vs = self.trans_v_mem(h_vt)
        representation_visual = h_vs[-1]
    
        # Concatenating text-visual representations
        av_h_rep = torch.cat([representation_text, representation_visual], dim=1)
        
        # Auxiliary text network
        h_t1 = self.t_mem(proj_x_t)
        h_t2 = self.t_mem(h_t1)
        h_t3 = self.t_mem(h_t2)
        h_rep_t_aux = h_t3[-1]   
            
        # Auxiliary visual network
        h_v1 = self.v_mem(proj_x_v)
        h_v2 = self.v_mem(h_v1)
        h_v3 = self.v_mem(h_v2)
        h_rep_v_aux = h_v3[-1]
            
        # Text auxiliary network output
        linear_hs_proj_t = self.proj_aux3(
            F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_t_aux)), p=self.out_dropout, training=training))), p=self.out_dropout, training=training)
        )
        linear_hs_proj_t += h_rep_t_aux
        output_t_aux = self.out_layer_aux(linear_hs_proj_t)
        
        # Visual auxiliary network output
        linear_hs_proj_v = self.proj_aux3(
            F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_v_aux)), p=self.out_dropout, training=training))), p=self.out_dropout, training=training)
        )
        linear_hs_proj_v += h_rep_v_aux
        output_v_aux = self.out_layer_aux(linear_hs_proj_v)
        
        # Main network output
        linear_hs_proj_av = self.proj2(F.dropout(F.relu(self.proj1(av_h_rep)), p=self.out_dropout, training=training))
        linear_hs_proj_av += av_h_rep
        output = self.out_layer(linear_hs_proj_av)
        
        return output, output_t_aux, output_v_aux
