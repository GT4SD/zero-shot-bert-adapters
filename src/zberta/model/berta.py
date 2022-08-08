import torch.nn as nn

class BERTA(nn.Module):
    def __init__(self,
                 bert_model,
                 hidden_dim,
                 output_dim,
                 dict=False
                ):
        
        super().__init__()
        
        self.bert = bert_model
        self.config = bert_model.config
        self.dict = dict
        embedding_dim = bert_model.config.to_dict()['hidden_size']
        
        #self.fc = nn.Linear(embedding_dim, hidden_dim)

        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        #sequence = [sequence len, batch_size]
        #attention_mask = [seq_len, batch_size]
        #token_type = [seq_len, batch_size]
                
        embedded = self.bert(input_ids, attention_mask, token_type_ids)[1]
        #print('emb ', embedded.size())

        #self.bert() gives tuple which contains hidden outut corresponding to each token.
        #self.bert()[0] = [seq_len, batch_size, emd_dim]
                
        #embedded = [batch size, emb dim]
        
        #ff = self.fc(embedded)
        #ff = [batch size, hid dim]

        #ff1 = self.fc2(ff)
                
        
        
        output = self.out(embedded)
        #print('output: ', output.size())
        #output = [batch size, out dim]
        
        return {'logits' : output} if self.dict else output
