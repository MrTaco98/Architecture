import torch

if not torch.cuda.is_available():
    print('CUDA is not available. Aborting')
    raise SystemExit
else:
    print('CUDA is available. Proceeding')
    device = torch.device('cuda')

resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50')