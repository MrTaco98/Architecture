import torch
from tresnet import TResnetM
from datasets import load_dataset
import os
from time import time
from tqdm import tqdm
os.environ["HF_TOKEN"] = "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"

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('CUDA is not available. Aborting')
        raise SystemExit
    else:
        print('CUDA is available. Proceeding')
        device = torch.device('cuda')

    print('Loading Model')
    start=time()
    model=TResnetM({'num_classes':10, 'remove_aa_jit':False})
    end=time()
    print("Finished Loading, took: ", end-start)

    print('Loading Dataset')
    start=time()
    ds = load_dataset("imagenet-1k", split='train', streaming=True)
    ds = ds.shuffle()
    end=time()
    print('Finished Loading, took: ',end-start)

    #print(next(iter(ds)))
    print('Starting Processing')
    start=time()
    for sample in ds.take(10000):
        dataloader=torch.utils.data.DataLoader(list(sample['image']), batch_size=64, num_workers=4)
        result=model(sample)
        print(result)
    end=time()
    print('Finished Processing, took: ', end-start)
    #print('Result: ',result)
