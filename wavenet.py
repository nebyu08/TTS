import torch
import torch.nn as nn 

class WaveNet(nn.Module):
    def __init__(self,
                 n_layers,
                 n_cnns,
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


        #implement the casual conv 
        self.casual_conv=nn.Conv1d(in_channels=in_channel,
                                   out_channels=in_channel,
                                   kernel_size=kernel_size
                                   )
        
        #dilated conv
        self.dilated_conv=nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,

        )
    

    def forward(self,inputs):
        layer_input=self.casual_conv(inputs)
        main_module=nn.ModuleList(self.dilated_conv for _ in range(self.n_layers) )
        layers_output=[]

        #gated activation
        for i in range(self.n_layers):
            temp_tanh=torch.tanh(main_module[i](inputs))
            temp_sigma=nn.tor