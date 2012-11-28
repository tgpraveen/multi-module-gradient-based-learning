require 'nn' 

--use math.exp(w) ??

local MulPos, parent = torch.class('nn.MulPos', 'nn.Module')

function MulPos:__init(inputSize)
   parent.__init(self)
  
   self.weight = torch.Tensor(1)
   self.gradWeight = torch.Tensor(1)
   
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(inputSize) 

   self:reset()
end

function MulPos:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   self.weight[1] = torch.uniform(-stdv, stdv);
end

function MulPos:updateOutput(input)
   self.output:copy(input);
   -- self.output:mul(self.weight[1]);
   self.output:mul(math.exp(self.weight[1]));
   return self.output 
end

function MulPos:updateGradInput(input, gradOutput) 
   self.gradInput:zero()
   -- self.gradInput:add(self.weight[1], gradOutput)
   -- gradOut_mul_e_pow_X = gradOutput:mul(math.exp(self.gradWeight[1]))
   -- self.gradInput:add(self.weight[1], gradOut_mul_e_pow_X)

self.gradInput:add((math.exp(self.weight[1])), gradOutput)
   return self.gradInput
end

function MulPos:accGradParameters(input, gradOutput, scale) 
   -- print("In MulPos:accGradParameters().")
   -- scale is learning rate.
   scale = scale or 1
   -- For normal Mul, Y=w*X, E=(1/2)*(wX-Y)^2, gradOutput=(wX-Y) ie Predicted - Actual, dE/dW=X.gradOutput    X-> Input
   -- For MulPos, Y=e^w*X, E=(1/2)*(e^w*X-Y)^2, gradOutput=(e^w*X-Y) ie Predicted - Actual, dE/dW=e^w*X.gradOutput X-> Input
   -- self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);
   -- local e_pow_input_mul_input = torch.mul(input, math.exp(self.weight[1]))
   -- torch:mul(e_pow_input_mul_input, input, math.exp(self.weight[1]))
   -- gradOut_dot_input = input:dot(gradOutput)
   

-- self.gradWeight[1] = self.gradWeight[1] + scale*(torch.mul(input, math.exp(self.weight[1]))):dot(gradOutput);

   -- self.gradWeight[1] = self.gradWeight[1] + scale*input:dot(gradOutput);

  local tempOut = self.output:clone()
  -- local tempOut = (input:mul(math.exp(self.weight[1]))):clone()
  self.gradWeight[1] = self.gradWeight[1] + scale*tempOut:dot(gradOutput);

end
