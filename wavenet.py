import torch
import torch.nn as nn 


class GatedActivationUnits(torch.nn):
    def __init__(self,in_shape):
        super().__init__()
        self.filter_conv=nn.Conv1d(in_channels=in_shape,out_channels=in_shape)
        self.gated_conv=nn.Conv1d(in_channels=in_shape,out_channels=in_shape)

    def forward(self,inputs):
        filter_output=self.filter_conv(inputs)
        gated_output=self.gated_conv(inputs)
        return torch.tanh(filter_output) * torch.sigmoid(gated_output)

class CasualDilatedConv(nn.Module):
    def __init__(self,in_channel,out_channle,dilation,kernel_sz):
        super().__init__()
        self.conv=nn.Conv1d(in_channels=in_channel,out_channels=out_channle,kernel_size=kernel_sz,dilation=dilation,padding=(kernel_sz-1)*dilation) #reason for padding is input and output shape must be the same

    def forward(self,x):
        return self.conv(x)

class WaveNet(nn.Module):
    def __init__(self,
                 n_layers_per_block,
                 n_block,
                 kernel_size,
                 skip_con,
                 residual_con,
                 dilation_sz,
                 in_channel=64,
                 out_channel=64,
                 ):
        
        super().__init__()

        #hyperparamters
        self.n_layers=n_layers_per_block
        self.n_block=n_block

        #initialize gated activation units
        self.gated_units=GatedActivationUnits(in_channel)

        #layer of dilated conv and gated activation 
        self.layers=nn.ModuleList()
        self.skip_con=nn.ModuleList()
        self.residual_con=nn.ModuleList

        
        for block_num in range(n_block):
            for layer_block in range(n_layers_per_block):
                self.layers.append(CasualDilatedConv(1 if block_num==0 and layer_block==0 else in_channel,out_channel,kernel_sz=2,dilation=dilation_sz))
                self.layers.append(GatedActivationUnits(in_shape=out_channel))

                self.skip_con.append(nn.Conv1d(out_channel,out_channel,kernel_size=1))
                self.residual_con.append(nn.Conv1d(out_channel,in_channel,kernel_size=1))

        #final conv
        self.final_conv1=nn.Conv1d(in_channels=in_channel,out_channels=in_channel,kernel_size=1)
        self.final_conv2=nn.Conv1d(in_channels=in_channel,out_channels=256,kernel_size=1 )
    
    def mu_encoding(audio,mu=255):
        """this function received raw audion as input and returns a compressed values 

        Args:
            audio (audio): average human sounds
        """

        #audio mu
        audio=torch.clamp(audio,min=-1.0,max=1.0)

        #scale to range -1 to 1
        result=torch.sign(audio)*(torch.log1p(mu*torch.abs(audio)))/(torch.log1p(torch.tensor(mu,dtype=torch.float32)))

        #scale from 1 to mu[255]
        quantized=(((result+1)/2)*mu).long

        return quantized
    

    def forward(self,inputs):  #input shape is batch size,1,sequence length(16000)
        #quantized data
        inputs=self.mu_encoding(inputs)  #input here batch size,1,16000

        skip_output=[]

       for i in range(len())