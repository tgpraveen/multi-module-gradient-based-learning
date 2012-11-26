require "nn"

function logisticRegression(traindataset, testdataset)
mlp = nn.Sequential();
inputs = traindataset:features(); outputs = 1; HUs = traindataset:classes(); -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
-- criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 100
trainer:train(traindataset)
return trainer, mlp
end
