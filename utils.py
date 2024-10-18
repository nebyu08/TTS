import torchaudio
import torch
import os
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self,file_list,segment_length=16000,sample_rate=16000,mu=255):
         super().__init__()
         self.file_list=file_list
         self.segement_length=segment_length
         self.sample_rate=sample_rate
         self.mu=mu

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

    def load_audio(self,filepath):
        waveform,original_sample=torchaudio.load(filepath)
        if (original_sample!=self.segement_length):
            resampler=torchaudio.transforms.Resample(orig_freq=original_sample,new_freq=self.sample_rate)
            waveform=resampler(waveform)
        return waveform.squeeze(0)  #remove channel dim (mono)  it returns 1D dim

    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_path=self.file_list[idx]
        audio=self.load_audio(audio_path)

        if audio.size(0)<self.segement_length:
            raise ValueError(f"Audio file {audio_path} is short than the expected segement.")
        
        #randonly select section from the audio
        start_idx=torch.randint(0,audio.size(0)-self.segement_length+1,(1,)).item()
        audio_segement=audio[start_idx:start_idx+self.segement_length]

        #apply mu encoding 
        mu_encoded=self.mu_encoding(audio_segement,self.mu)

        return mu_encoded.float()/self.mu #normalized data


def list_audio_files(dir):
    audio_files=[]
    for dir_path,dir_name,file_names in os.walk(dir):
        for file_name in [f for f in file_names if f.endswith((".mp3",".wav",".aif"))]:
            audio_files.append(os.path.join(dir_path,file_name))


    if(len(audio_files)==0):
        print("no audio found in the location")

    return audio_files