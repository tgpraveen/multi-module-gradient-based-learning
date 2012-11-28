require 'nn' 

local NegExp, parent = torch.class('nn.NegExp', 'nn.Module')

function NegExp:updateOutput(input)
   -- return input.nn.Exp_updateOutput(self, input)
   return input.nn.Exp_updateOutput(self, torch.mul(input,-1))
end

function NegExp:updateGradInput(input, gradOutput)
   -- dE/dX = (gradOutput) * (-e^-x)
   -- temp = input.nn.Exp_updateGradInput(self, input, gradOutput)
   return torch.mul(input.nn.Exp_updateGradInput(self, torch.mul(input,-1), gradOutput), -1)
   -- return temp
end

