import torch
import torch.nn as nn 


class GatedActivationUnits(torch.nn):
    def __init__(self,in_shape,conditional_wavent=False):
        super().__init__()
        self.filter_conv=nn.Conv1d(in_channels=in_shape,out_channels=in_shape)
        self.gated_conv=nn.Conv1d(in_channels=in_shape,out_channels=in_shape)

    def forward(self,inputs):
        filter_output=self.filter_conv(inputs)
        gated_output=self.gated_conv(inputs)
        return torch.tanh(filter_output) * torch.sigmoid(gated_output)

class WaveNet(nn.Module):
    def __init__(self,
                 n_layers,
                 in_channel,
                 out_channel,
                 kernel_size,
                 skip_con,
                 residual_con,
                 dilation_sz
                 ):
        
        super().__init__()

        #hyperparamters
        self.n_layers=n_layers

        #initialize gated activation units
        self.gated_units=GatedActivationUnits(in_channel)

        #implement the casual conv 
        self.casual_conv=nn.Conv1d(in_channels=in_channel,
                                   out_channels=in_channel,
                                   kernel_size=kernel_size
                                   )

        #layer of dilated conv
        for _ in range(self.n_layers):
            self.layers=nn.ModuleList([nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,dilation=dilation_sz)] )

    
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

        #pass data through causal conv
        layer_input=self.casual_conv(inputs)  #batch size,1,16000

        #pass thorught a

        #pass data through gated conv
        gated_output=self.gated_units(layer_input)