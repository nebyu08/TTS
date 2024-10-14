import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=512,
                 num_conv_layers=3,
                 conv_filters=512,
                 kernel_size=5,
                 lstm_units=512
                 ):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        conv_layers=[]
        for _ in range(num_conv_layers);
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=embedding_dim,out_channels=conv_filters,kernel_size=kernel_size,padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(conv_filters),
                    nn.ReLU()
                )
            )


        #make it sequential
        self.conv_stack=nn.Sequential(*conv_layers)

        self.lstm=nn.LSTM(input_size=conv_filters,hidden_size=lstm_units,batch_first=True,bidirectional=True)
    
    def forward(self,x):
        x=self.embedding(x)   #dim is [Batch size,sequence length,embedding dime]
        
        #transpose for 1d conv
        x=x.transpose(1,2)   #shape is [Batch size,embedding dim,sequence length]

        x=self.conv_stack(x)
        
        #return back to original shape
        x=x.transpose(1,2)

        outputs,_=self.lstm(x)
        return outputs
    

class Attention(nn.Module):
    def __init__(self,attention_dim):
        super().__init__()
        self.query=nn.Linear(attention_dim,attention_dim) #decoder hidden state from LSTM
        self.key=nn.Linear(attention_dim,attention_dim) #Encoder output
        self.energy_layer=nn.Linear(attention_dim,1)

    def forward(self,query,key):
        query=self.query(query)  # [batch size,attention dim]
        query=query.unsqueeze(1)
        key=self.key(key)  # batch size,sequence length ,attention dim

        #calculate similarity
        energy=self.energy_layer(torch.tanh(query+key))  # batch size,seq_len,1
        attention_weight=torch.softmax(energy.squeeze(-1),dim=1)  #batch size, seq len

        #getting the context
        context=torch.bmm(attention_weight.unsqueeze(1) ,key).squeeze(1) #batch size, attention dim

        return context,attention_weight


    
#Gegenerate MEL SPECTOGRAM

class Decoder(nn.Module):
    def __init__(self,
                 attention_dim=512,
                 lstm_units=1024,
                 prenet_dim=256,
                 mel_dim=80
                 ):
        super().__init()

        #prenet
        self.prenet=nn.Sequential(
            nn.Linear(mel_dim,prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(prenet_dim,prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        #attention 
        self.attention=Attention(attention_dim)

        #auto regressive generation of mel spectogram
        self.lstm=nn.LSTM(input_size=prenet_dim+attention_dim,hidden_size=lstm_units,num_layers=2,batch_first=True,dropout=0.1)

        #linear projection to mel spectogram 
        self.mel_projection=nn.Linear(lstm_units+attention_dim,mel_dim)

        #postnet to refine the spectogram
        self.postnet=nn.Sequential(
            nn.Conv1d(mel_dim,512,kernel_size=5,padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Conv1d(512,512,kernel_size=5,padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Conv1d(512,512,kernel_size=5,padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Conv1d(512,512,kernel_size=5,padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            
            nn.Conv1d(512,mel_dim,kernel_size=5,padding=2)
        )

        self.stop_token_prediction=nn.Linear(lstm_units+attention_dim,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,encoder_outputs,prev_mel_frame,hidden,cell):
        prenet_output=self.prenet(prev_mel_frame)

        context,_=self.attention(hidden[0][-1],encoder_outputs)

        #lstm input
        lstm_concat_input=torch.cat([prenet_output,context],dim=-1).unsqueeze(1)

        lstm_output,(hidden,cell)=self.lstm(lstm_concat_input,(hidden,cell))

        #concatinate lstm and attention context 
        lstm_output=lstm_output.squeeze(1)
        context_lstm_concat=torch.cat([lstm_output,context],dim=-1)

        #mel spectogram prediction
        mel_frame=self.mel_projection(context_lstm_concat)

        #stop word prediction
        stop_token=self.sigmoid(self.stop_token_prediction(context_lstm_concat))

        #refine the output post refinement
        mel_frame=mel_frame.unsqueeze(1).transpose(1,2)
        refine_mel_frame=self.postnet(mel_frame).transpose(1,2).squeeze(1)


        return mel_frame.squeeze(1),refine_mel_frame,stop_token,hidden,cell
        