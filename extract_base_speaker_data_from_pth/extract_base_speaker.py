import os
import torch

# save list to txt
def SaveList(name, outlist):
    with open(name, 'w') as f:
        for i in outlist:
            f.write(i+'\n')

speakers = ['en-au', 'en-india', 'es', 'jp', 'en-br', 'en-newest', 'kr', 'en-default', 'en-us', 'fr', 'zh']

for s in speakers:
    data = torch.load(s+'.pth')
    print(s, data.shape)
    SaveList(s+'.txt', [str(i) for i in data.squeeze(0).squeeze(1).numpy()])
