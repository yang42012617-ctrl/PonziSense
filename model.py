import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        
        self.cluster_centers = nn.Parameter(torch.randn(10, config.hidden_size))  
        self.temperature = nn.Parameter(torch.tensor(0.5))
        
        self.edge_perturbator = nn.Linear(config.hidden_size, 1)
        self.interpreter_head = nn.Linear(config.hidden_size, config.hidden_size)
    
        
    def extract_features(self, inputs_ids,position_idx,attn_mask):
        bs,l=inputs_ids.size()
        inputs_ids=(inputs_ids.unsqueeze(1)).view(bs*1,l)
        position_idx=(position_idx.unsqueeze(1)).view(bs*1,l)
        attn_mask=(attn_mask.unsqueeze(1)).view(bs*1,l,l)
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]
        return outputs
    
    def forward(self, inputs_ids,position_idx,attn_mask,labels=None): 
        outputs = self.extract_features(inputs_ids,position_idx,attn_mask)
        logits=self.classifier(outputs)
        prob=F.softmax(logits, dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
    
    def forward_interpreter(self, inputs_ids, position_idx, attn_mask, original_logits):
        
        features = self.extract_features(inputs_ids, position_idx, attn_mask)
        
        
        edge_scores = torch.sigmoid(self.edge_perturbator(features))
        
        
        perturbed_attn_mask = attn_mask * (torch.rand_like(edge_scores) < edge_scores).float()
        
        
        perturbed_features = self.extract_features(inputs_ids, position_idx, perturbed_attn_mask)
        perturbed_logits = self.classifier(perturbed_features)
        
        
        loss_perturb = F.mse_loss(original_logits, perturbed_logits)
        
        
        reversed_features = self.interpreter_head(perturbed_features)
        pred_edge_scores = torch.sigmoid(torch.matmul(features, reversed_features.transpose(1,2)))
        loss_reverse = F.binary_cross_entropy(pred_edge_scores, edge_scores)
        
        
        return loss_perturb + 0.1 * loss_reverse
    
    def forward_contrastive(self, inputs_ids_s,position_idx_s,attn_mask_s,
                            inputs_ids_w,position_idx_w,attn_mask_w,
                            inputs_ids_m,position_idx_m,attn_mask_m):
        
        features_s = self.extract_features(inputs_ids_s,position_idx_s,attn_mask_s)
        features_w = self.extract_features(inputs_ids_w,position_idx_w,attn_mask_w)
        features_m = self.extract_features(inputs_ids_m,position_idx_m,attn_mask_m)
        
        
        delta_s = torch.matmul(features_s[:,0,:], self.cluster_centers.t())
        delta_w = torch.matmul(features_w[:,0,:], self.cluster_centers.t())
        delta_m = torch.matmul(features_m[:,0,:], self.cluster_centers.t())
        
        
        sim_s_w = F.cosine_similarity(delta_s, delta_w)
        sim_s_m = F.cosine_similarity(delta_s, delta_m)
        sim_w_m = F.cosine_similarity(delta_w, delta_m)
        
        
        similarities = torch.stack([sim_s_w, sim_s_m, sim_w_m], dim=1) / self.temperature
        
        
        labels = torch.zeros(similarities.size(0), dtype=torch.long, device=inputs_ids_s.device)
        
        
        loss = F.cross_entropy(similarities, labels)
        return loss
      
        

       
