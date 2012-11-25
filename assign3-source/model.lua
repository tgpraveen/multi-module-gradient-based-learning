require "nn"

function logisticRegression(dataset)
mlp = nn.Sequential();
inputs = dataset:features(); outputs = 1; HUs = dataset:classes(); -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.SoftMax())
criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(mlp, criterion)
return trainer
end
