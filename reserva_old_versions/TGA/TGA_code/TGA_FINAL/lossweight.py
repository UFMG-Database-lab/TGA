class LossWeighter(torch.nn.Module):
    def __init__(self, model, criterions):
        super(LossWeighter, self).__init__()

        self.criterions = criterions
        self.model = torch.nn.ModuleList([model])        

        self.sigma = torch.nn.Parameter(torch.ones(2)) #pesos

    def forward(self, x, targets=None):
        out1, out2 = self.model[0](x)
        
        if type(targets) == type(None):
            return (out1, out2)

        y1 = self.criterion[0](out1, targets[0])
        y2 = self.criterion[1](out2, targets[1])

        y1 = y1/(2*(self.sigma[0]**2)) + torch.log(self.sigma[0]) #quando a loss é continua, o denominador do primeiro termo fica multiplicado por 2 
        y2 = y2/(self.sigma[1]**2) + torch.log(self.sigma[1]) #quando a loss e discreta, e so dividir pelo quadrado e somar o log

        return (y1+y2).mean()
