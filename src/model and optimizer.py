## ######################################################### ##
##     This contains the function for model and optimizer    ##
##     which is custom made and its results are compared     ##
##                with adam's and rmsprop's                  ##
## ######################################################### ##


##-------------------------------------------##
##             Model building                ##
##-------------------------------------------##

class RegressionNet(nn.Module):
    def __init__(self, num_features):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

##-------------------------------------------##
##           Optimizer defining              ##
##-------------------------------------------##

class ModifiedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.003, beta1=0.45, beta2=0.92, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super(ModifiedAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['v'] = torch.zeros_like(p.data)
                    state['k'] = torch.zeros_like(p.data)

                v, k = state['v'], state['k']
                beta1, beta2, eps, lr = group['beta1'], group['beta2'], group['eps'], group['lr']

                v.mul_(beta1).add_(grad, alpha=1 - beta1)
                k.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                lookahead_grad = grad - beta1 * v
                p.data.addcdiv_(lookahead_grad, torch.sqrt(k) + eps, value=-lr)

        return loss
