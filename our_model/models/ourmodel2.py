'''
Description:
Author: J Chen
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#pass
def init_weights(m):
    if type(m)==torch.nn.modules.linear.Linear:
        try:
            torch.nn.init.uniform_(m.weight,a=-0.1,b=0.1)#weight
            torch.nn.init.uniform_(m.bias,a=-0.1,b=0.1)#bias
        except Exception:
            pass

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x): 
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Myloss2(torch.nn.Module):
    def __init__(self,opt):
        super(Myloss2, self).__init__()
        self.opt = opt

    def forward(self,weight):
        return weight

class Myloss(torch.nn.Module):
    def __init__(self,opt):
        super(Myloss, self).__init__()
        self.opt = opt

    def forward(self,inputs,polarity,standar_score):
        standar_score.requires_grad = False 

        inputs_h_extend = torch.unsqueeze(inputs, dim=1).repeat(1, self.opt.polarities_dim,1) 

        loss_all = inputs_h_extend - standar_score.to(self.opt.device)  
        loss_all_sqr = torch.bmm(loss_all, torch.permute(loss_all, dims=(0, 2, 1)).contiguous())

        loss_all_sqr_eye = torch.diagonal(loss_all_sqr, dim1=-1, dim2=-2)
        mask_min = torch.zeros(inputs_h_extend.size(0), self.opt.polarities_dim, requires_grad=False).to(self.opt.device)  

        for i, j in enumerate(polarity.cpu().numpy()):
            mask_min[i][int(j)] = 1

        loss_min = torch.sum(torch.sum(loss_all_sqr_eye * mask_min,dim=-1),dim=-1)/(2*self.opt.attention_dim*self.opt.batch_size)

        cos0_1 = cos_(standar_score[0],standar_score[1]) #cos(0,1)
        cos0_2 = cos_(standar_score[0],standar_score[2]) #cos(0,2)
        cos1_2 = cos_(standar_score[1],standar_score[2]) #cos(1,2)

        loss = loss_min + 15*(cos0_1*cos0_1 + cos0_2*cos0_2 + cos1_2*cos1_2) 

        return loss

def cos_(input1,input2):
    return torch.sum(input1*input2,dim=-1)/torch.sqrt(torch.sum(input1*input1,dim=-1))*torch.sqrt(torch.sum(input2*input2,dim=-1))

def kl_divergence(input1, input2):

    return F.kl_div((input2+0.01).log(), input1+0.01, reduction='sum') 

def d_(input1,input2):
    return torch.sum(input1-input2,dim=-1)

class OurModelBertClassifier(nn.Module):
    def __init__(self, bert, opt):  #
        super().__init__()
        self.opt = opt

        self.ave_pool = AvePool(bert, opt)
        self.batch_pool = BatchPoolLoss(opt)

        self.fnn2 = nn.Linear(opt.attention_dim * 2, opt.polarities_dim)  

    def forward(self, inputs):

        outputs1, outputs2, porality, noise = self.ave_pool(inputs) 


        standar_score = self.batch_pool(outputs2[1],porality) 

        # aspect_output
        logits_gate_aspect = self.fnn2(outputs1[1])
        logits_bgcn_aspect = 1
        # (batch,3)、list、score
        return [logits_bgcn_aspect, logits_gate_aspect, outputs2[1], porality, standar_score,noise], None
########################################################################################################################

########################################################################################################################


class BatchPoolLoss(torch.nn.Module):
    def __init__(self,opt):
        super(BatchPoolLoss, self).__init__()
        self.opt = opt

    def forward(self,inputs,porality):


        sta_score_init = torch.zeros(3, inputs[1].size(-1)).to(self.opt.device)  
        for i, j in enumerate(porality.cpu().numpy()):
            sta_score_init[int(j)] = sta_score_init[int(j)] + inputs[i]

        '''(polariti_dim,attention_dim*2)'''
        #(tensor([0., 1., 2.]), tensor([3, 2, 2]))
        num_0_1_2 = torch.unique(porality, return_counts=True) 
        num_0_1_2_init = torch.ones(3).to(self.opt.device) 
        for i, j in zip(num_0_1_2[0].cpu().numpy(), num_0_1_2[1].cpu().numpy()):
            num_0_1_2_init[int(i)] = j

        num_0_1_2_init = torch.unsqueeze(num_0_1_2_init, dim=-1) 
        sta_score = sta_score_init / num_0_1_2_init 
        return sta_score

####################################################################################################################
# pooling
class AvePool(torch.nn.Module):
    def __init__(self, bert, opt):
        super(AvePool, self).__init__()
        self.opt = opt
        self.bertstruture = BertStructure(bert, opt)

    def forward(self, inputs):

        bgcn_result, attention_result, gate_result, porality, noise = self.bertstruture(inputs)  


        ave_att_result_aspect = torch.sum(attention_result[1], dim=1) / attention_result[2]  
        ave_att_result_context = torch.sum(attention_result[0], dim=1) / torch.unsqueeze(
            torch.Tensor([self.opt.max_length]), dim=0).repeat(attention_result[0].size(0),1).to(self.opt.device) 


        ave_att_result_context = torch.nn.functional.normalize(ave_att_result_context, p=2, dim=1)/2

        ave_bgcn_result_aspect=1
        ave_bgcn_result_context =1

        #ave aspect:gcn、gate|ave context：gan、gate
        return [ave_bgcn_result_aspect, ave_att_result_aspect],[ave_bgcn_result_context, ave_att_result_context], porality, noise
####################################################################################################################

####################################################################################################################
class BertStructure(torch.nn.Module):
    def __init__(self, bert, opt):
        super(BertStructure, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layer_norm = LayerNorm(opt.bert_dim)

        self.attentionmodule1 = AttentionModule(opt, opt.c_c_Attention_num)  
        self.attentionmodule2 = AttentionModule(opt, opt.c_a_Attention_num)  

    def forward(self, inputs):
        adj_f, adj_b, adj_f_aspect, adj_b_aspect, text_bert_indices, bert_segments_ids, attention_mask, \
        text_len, post_1, asp_start, asp_end, src_mask, aspect_mask, polarity = inputs  

        lin_shi = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output, pooled_output = lin_shi[0], lin_shi[1]  

        sequence_output = self.layer_norm(sequence_output)  
        aspect_output = sequence_output * aspect_mask.unsqueeze(-1).repeat(1, 1,sequence_output.size(-1))  

        aspect_mask1 = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bgcn_dim) 
        aspect_mask2 = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bgcn_dim * 2)  
        aspect_mask3 = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.attention_dim) 
        aspect_mask4 = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.attention_dim * 2) 
        context_output_att1, context_output_att2, heads_context_scores, noise1 = self.attentionmodule1(opt=self.opt,
                                                                                               sequence1=sequence_output,
                                                                                               sequence2=sequence_output,
                                                                                               aspect_mask=None)  

        context_output_att3, aspect_output_att, heads_aspect_scores, noise2 = self.attentionmodule2(opt=self.opt,
                                                                                            sequence1=sequence_output,
                                                                                            sequence2=aspect_output,
                                                                                            aspect_mask=aspect_mask)  

        context_output_att_cat = torch.cat([context_output_att1, aspect_output_att], dim=-1)  
        aspect_output_att_cat = context_output_att_cat * aspect_mask4


        aspect_len = torch.unsqueeze((asp_end - asp_start + 1), dim=-1)


        context_output_bgcn_cat =1
        aspect_output_bgcn_cat =1
        all_output =1
        all_aspect_output =1

        return [context_output_bgcn_cat, aspect_output_bgcn_cat, aspect_len], \
               [context_output_att_cat, aspect_output_att_cat, aspect_len], \
               [all_output, all_aspect_output, aspect_len],\
               polarity,\
               [noise1,noise2]
####################################################################################################################

####################################################################################################################
class AttentionModule(torch.nn.Module):
    def __init__(self, opt, layer_num):
        super(AttentionModule, self).__init__()
        self.opt = opt
        self.attention = Attention(opt, layer_num)

    def forward(self, opt, sequence1, sequence2, aspect_mask):
        sequence_list1 = [sequence1]
        sequence_list2 = [sequence2]  
        score_list = []
        score_k_list =[]
        for i in range(opt.c_c_Attention_num):
            c_c_attention1, c_c_attention2, c_c_score, scores_noise_weight_sum = self.attention(sequence1=sequence_list1[-1],
                                                                       sequence2=sequence_list2[-1],
                                                                       head=opt.c_c_heads,
                                                                       len_=len(sequence_list1),
                                                                       mask=aspect_mask)
            sequence_list1.append(c_c_attention1)
            sequence_list2.append(c_c_attention1)
            score_list.append(c_c_score)
            score_k_list.append(scores_noise_weight_sum)
        return sequence_list1[-1], sequence_list2[-1], score_list[-1],sum(score_k_list)/opt.c_c_Attention_num

class Attention(torch.nn.Module):
    def __init__(self, opt, layer_num):
        super(Attention, self).__init__()
        self.dropout = torch.nn.Dropout(p=opt.gcn_dropout)

        self.w_b_q = torch.nn.Linear(opt.bert_dim, opt.attention_dim,bias=False)
        self.w_b_k = torch.nn.Linear(opt.bert_dim, opt.attention_dim,bias=False)
        self.w_b_v = torch.nn.Linear(opt.bert_dim, opt.attention_dim,bias=False)

        self.w_b_q1 = [torch.nn.Linear(opt.attention_dim, opt.attention_dim, device=opt.device) for _ in range(layer_num - 1)]
        self.w_b_k1 = [torch.nn.Linear(opt.attention_dim, opt.attention_dim, device=opt.device) for _ in range(layer_num - 1)]
        self.w_b_v1 = [torch.nn.Linear(opt.attention_dim, opt.attention_dim, device=opt.device) for _ in range(layer_num - 1)]

    def forward(self, sequence1, sequence2, head, len_, mask):

        sequence1 = self.dropout(sequence1)
        sequence2 = self.dropout(sequence2)

        if len_ > 1:
            '''Q K=V'''
            querys = self.w_b_q1[len_-2](sequence1)  # Q = X*W = (batch,max_length,attention_dim)
            keys = self.w_b_k1[len_-2](sequence2)  # Q = X*W = (batch,max_length,attention_dim)
            values = self.w_b_v1[len_-2](sequence2)  # Q = X*W = (batch,max_length,attention_dim)

        else:
            '''Q K=V'''
            querys = self.w_b_q(sequence1)  # [N, T_q, num_units]
            keys = self.w_b_k(sequence2)  # [N, T_k, num_units]
            values = self.w_b_v(sequence2)


        split_size = self.opt.attention_dim // head  
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = F.tanh(scores)
        scores = scores / (self.opt.bert_dim ** 0.5)
        scores = F.softmax(scores, dim=3)  # softmax （5,16，85,85）

        scores_k_list = []
        mask_k_list = []
        scores_max = scores
        scores_min = scores

        for i in range(self.opt.trac_att_k):

            max_values, max_index = torch.max(scores_max, dim=-1)
            mask_k_max = torch.zeros_like(scores)
            mask_k_max.scatter_(-1, max_index.unsqueeze(-1), 1)
            scores_k_max = scores * mask_k_max
            scores_k_list.append(scores_k_max) 
            mask_k_list.append(mask_k_max) 
            scores_max = scores_max - scores_k_max


        # scores_k_list_ = []
        mask_k_list_ = []
        for i in range(self.opt.trac_att_k):
            min_values, min_index = torch.min(scores_min, dim=-1)
            mask_k_min = torch.zeros_like(scores)
            mask_k_min.scatter_(-1, min_index.unsqueeze(-1), 1)
            mask_k_list_.append(mask_k_min) 
            scores_max = scores_max + mask_k_min


        mask_k_all = sum(mask_k_list)
        mask_k_all_ = sum(mask_k_list_)
        # print('max',mask_k_all_)
        # print('max',torch.sum(mask_k_all))
        # print('min',mask_k_all_)
        # print('min',torch.sum(mask_k_all_))


        if self.opt.trac_att_k > self.opt.max_length-self.opt.trac_att_k:
            mask_k_r_all = 1-mask_k_all 
        else:
            mask_k_r_all = mask_k_all_ 

        if mask is None:
            scores_noise_weight = mask_k_r_all * scores
            scores_noise_weight_sum = torch.sum(scores_noise_weight) / torch.sum(mask_k_r_all)
            scores = F.softmax(sum(scores_k_list),dim=3) 

        else:
            mask_ = mask.unsqueeze(-1).repeat(1, 1, self.opt.max_length)
            m = torch.unsqueeze(mask_,dim=0).repeat(head,1,1,1) * mask_k_r_all
            scores_noise_weight = scores * m
            scores_noise_weight_sum = torch.sum(scores_noise_weight) / torch.sum(m)

            scores = F.softmax(sum(scores_k_list), dim=3) * m  
            # print('aspect',torch.sum(m[0][0])/sum(m[0][0][0]))
        ## out = score * V
        context_out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        context_out = torch.cat(torch.split(context_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        # if mask is not None:
        #     aspect_out = context_out * mask
        #     return context_out, aspect_out, scores, scores_noise_weight_sum

        return context_out, context_out, scores, scores_noise_weight_sum
########################################################################################################################
