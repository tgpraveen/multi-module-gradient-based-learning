require "nn"

function logisticRegression(traindataset, testdataset)
mlp = nn.Sequential();
inputs = traindataset:features(); outputs = 1; HUs = traindataset:classes(); -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
-- criterion = nn.MSECriterion()

-- TRAINING PHASE
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 600
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


-- TESTING PHASE

-- Counter for wrong classifications
local testerror = 0
local predictedClass = 0
local minError = 999999
local predictionsTensor

for q=1, testdataset:size() do

	predictionsTensor = mlp:forward(testdataset[q][1])
    predictionsTensor:abs()

-- Now we find the actual predcited one by seeing the one with minimum error.
	predictedClass = 0
	minError = 999999
	for w=1, testdataset:classes() do
     if (predictionsTensor[w]<minError) then 
        minError = predictionsTensor[w]
	    predictedClass = w
	 end
	end

    if (predictedClass == testdataset[q][2]) then
        print("Correct Prediction")
		testerror = testerror*(q-1)/q
    else
        print("Wrong Prediction")
        print("PredictionsTensor")
		print(predictionsTensor)
        print("Prediction is: "..predictedClass.." Actual class is: "..testdataset[q][2])
		testerror = (testerror*q-testerror + 1)/q
	end
end

print("Logistic Regression: Testing error is "..testerror)

return trainer, mlp
end
