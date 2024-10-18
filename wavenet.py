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

        self.blocks=nn.ModuleList()

        for block_num in range(self.n_block):
            block_layer=nn.ModuleList()
            for layer_block in range(self.n_layers):
                dilation=dilation_sz**layer_block
                block_layer.append(
                    CasualDilatedConv(
                        in_channel if layer_block==1 and block_num==1 else out_channel,
                        out_channel,
                        dilation=dilation,
                        kernel_sz=kernel_size
                    )
                )

                block_layer.append(GatedActivationUnits(in_shape=out_channel))

            self.blocks.append(block_layer)
        
        
        self.skip_con=nn.ModuleList([
            nn.Conv1d(out_channel,out_channel,kernel_size=1) for _ in range(n_block*n_layers_per_block)
        ])

        self.residual_con=nn.ModuleList([
            nn.Conv1d(out_channel,in_channel,kernel_size=1) for _ in range(n_block*n_layers_per_block)
        ])

        #final layer
        self.final_conv1=nn.Conv1d(in_channel,in_channel,kernel_size=kernel_size)
        self.final_conv2=nn.Conv1d(in_channel,256,kernel_size=1 )
    
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
        x=self.mu_encoding(inputs)  #input here batch size,1,16000

        skip_outputs=[]

        for block_num,block_layer in enumerate(self.blocks):
            for layer_num in range(self.n_layers):
                #conv and gated output
                conv=block_layer[layer_num*2](x)
                gated=block_layer[layer_num*2+1](x)

                #skip and residual connection
                skip_out=self.skip_con[self.block_num*self.n_layers+layer_num](gated)
                skip_outputs.append(skip_out)

                residual_out=self.residual_con(block_num*self.n_layers+layer_num)(gated)
                x=x+residual_out

        #sum the skip connections
        skip_sum=sum(skip_outputs)    #batch size,n_skip_channels,sequence length

        #final output
        output=torch.relu(self.final_conv1(skip_out))
        output=self.final_conv2(output)  #batch size,256,sequence length

        return  torch.softmax(output,dim=1)