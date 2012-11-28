require 'nn' 

local RBF, parent = torch.class('nn.RBF', 'nn.Module')

function RBF:__init(inputSize,outputSize)
   parent.__init(self)

   --self.weight = torch.Tensor(inputSize,outputSize)
   --self.gradWeight = torch.Tensor(inputSize,outputSize)

   self.weight = torch.Tensor(inputSize)
   self.gradWeight = torch.Tensor(inputSize)

   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)
   self.temp = torch.Tensor(inputSize)

   self:reset()
end

function RBF:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

  --[[ 
  for i=1,self.weight:size(1) do
      self.weight:select(2, i):apply(function()
                                        return torch.uniform(-stdv, stdv)
                                     end)
   end
--]]
for i=1,self.weight:size(1) do
self.weight[i] = torch.uniform(-stdv, stdv);
end
end

function RBF:updateOutput(input)
   self.output:zero()
   for o = 1,self.weight:size(1) do
      self.output[o] = self.output[o] + math.pow(input[o]-self.weight[o],2)
   end
   return self.output
end

function RBF:updateGradInput(input, gradOutput)
   self:updateOutput(input)
   if self.gradInput then
      self.gradInput:zero()
    --  for o = 1,self.weight:size(1) do
--[[        if self.output[o] ~= 0 then
            self.temp:copy(input):add(-1,self.weight:select(2,o))
            self.temp:mul(gradOutput[o]/self.output[o])
            self.gradInput:add(self.temp)
        end
--]]
print("self.output is:")
print(self.output)
  self.gradInput = torch.mul(gradOutput, (self.output*2))

    --  end
      return self.gradInput
   end
end

function RBF:accGradParameters(input, gradOutput, scale)
   self:updateOutput(input)
   scale = scale or 1
   --[[ for o = 1,self.weight:size(2) do
      if self.output[o] ~= 0 then
         self.temp:copy(self.weight:select(2,o)):add(-1,input)
         self.temp:mul(gradOutput[o]/self.output[o])
         self.gradWeight:select(2,o):add(self.temp)
      end
   end --]]
for o=1,self.gradWeight:size(1) do 
 self.gradWeight[o] = self.gradWeight[o] + scale*(gradOutput[o] * math.mul(math.sqrt(self.output),-2))
end
end
