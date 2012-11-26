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
--[[print("testdataset[1]")
print(testdataset[1])
print("testdataset[1][1]")
print(testdataset[1][1])
print("testdataset[1][2]")
print(testdataset[2][2])
print("Prediction:")
print(mlp:forward(testdataset[2][1]))
--]]
for q=1, testdataset:size() do

end

return trainer, mlp
end
