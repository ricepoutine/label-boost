import torch
import torch.nn as nn
import gin
import torch.nn.functional as F

@gin.configurable
class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels, h_dim = (10, 10), data_parallel=True, drop_rate=0.0, device="cuda:0"):
        super().__init__()

        self.network = nn.Sequential()
        self.network.add_module(f'fc_{0}', nn.Linear(in_channels,h_dim[0]))
        self.network.add_module(f'relu_{0}', nn.LeakyReLU(inplace=True))
        for i, dim in enumerate(h_dim):
            if i < len(h_dim) - 1:
                self.network.add_module(f'hidden_{i+1}', nn.Linear(dim, h_dim[i+1]))
                self.network.add_module(f'relu_{i+1}', nn.LeakyReLU(inplace=True))
        self.network.add_module(f'fc_{i+1}', nn.Linear(h_dim[-1], out_channels))
        
        if data_parallel:
            self.network = nn.DataParallel(self.network)
        
    def forward(self, z):
        batch_size = z.size(0)
        features = z.view(batch_size, -1)
        logits = self.network(features)
        #print(logits)
        return F.log_softmax(logits, dim=1) #log softmax or just softmax?

if __name__ == '__main__':
    import sys
    sys.path.append(".")
    from lib.criterion import ClsCriterion
    net = MultiLayerPerceptron(32, 3, (128,64,32), True)
    batch_size = 1
    # random z sampled latent x batch size
    #print(net)
    #a = torch.Tensor([[0.25, 0.25, 0.50]]).to(device="cuda:0")
    a = torch.randn(batch_size, 32).to(device="cuda:0")
    label = [1, 0, 0]
    label = torch.Tensor(label).to(device="cuda:0")
    y = net(a)
    #y=a
    print(y, label)
    CEloss = ClsCriterion()
    print(CEloss(y, label))