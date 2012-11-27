require 'nn' 

local NegExp, parent = torch.class('nn.NegExp', 'nn.Module')

function NegExp:updateOutput(input)
   -- return input.nn.Exp_updateOutput(self, input)
   return input.nn.Exp_updateOutput(self, input:mul(-1))
end

function NegExp:updateGradInput(input, gradOutput)
   -- dE/dX = (gradOutput) * (-e^-x)
   temp = input.nn.Exp_updateGradInput(self, input:mul(-1), gradOutput)
   return temp:mul(-1)
   -- return temp
end

