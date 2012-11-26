require "nn"

-- Q2.1.:
function logisticRegression(traindatasetorig, testdatasetorig)

local traindataset = {}
for i = 1,traindatasetorig:size() do
-- Cloning train data instead of referencing, so that the datset can be modified multiple times
traindataset[i] = {traindatasetorig[i][1]:clone(), traindatasetorig[i][2]:clone()}
end
function traindataset:size() return traindatasetorig:size() end
function traindataset:features() return traindatasetorig:features() end
function traindataset:classes() return traindatasetorig:classes() end

local testdataset = {}
for i = 1,testdatasetorig:size() do
-- Cloning test data instead of referencing, so that the datset can be modified multiple times
testdataset[i] = {testdatasetorig[i][1]:clone(), testdatasetorig[i][2]:clone()}
end
function testdataset:size() return testdatasetorig:size() end
function testdataset:features() return testdatasetorig:features() end
function testdataset:classes() return testdatasetorig:classes() end

mlp = nn.Sequential();
inputs = traindataset:features(); outputs = 1; HUs = traindataset:classes(); -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.LogSoftMax())

-- Modify Train and Test Dataset to have an integer as predicted class instead of a Tensor.
for n=1, traindataset:size() do
traindataset[n][2]=traindataset[n][2][1]
end

for m=1, testdataset:size() do
testdataset[m][2]=testdataset[m][2][1]
end


criterion = nn.ClassNLLCriterion()
-- criterion = nn.MSECriterion()

-- TRAINING PHASE
trainer = nn.StochasticGradient(mlp, criterion)
--trainer.learningRate = 0.1
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

print("Q.2.1: Logistic Regression: Testing error is "..testerror)

return trainer, mlp
end

-- Q2.2.:
function twoLayerNN(traindatasetorig, testdatasetorig)

local traindataset = {}
for i = 1,traindatasetorig:size() do
-- Cloning train data instead of referencing, so that the datset can be modified multiple times
traindataset[i] = {traindatasetorig[i][1]:clone(), traindatasetorig[i][2]:clone()}
end
function traindataset:size() return traindatasetorig:size() end
function traindataset:features() return traindatasetorig:features() end
function traindataset:classes() return traindatasetorig:classes() end

local testdataset = {}
for i = 1,testdatasetorig:size() do
-- Cloning test data instead of referencing, so that the datset can be modified multiple times
testdataset[i] = {testdatasetorig[i][1]:clone(), testdatasetorig[i][2]:clone()}
end
function testdataset:size() return testdatasetorig:size() end
function testdataset:features() return testdatasetorig:features() end
function testdataset:classes() return testdatasetorig:classes() end

mlp = nn.Sequential();
inputs = traindataset:features(); outputs = 1; HUs = traindataset:classes(); -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.LogSoftMax())

-- Modify Train and Test Dataset to have an integer as predicted class instead of a Tensor.
for n=1, traindataset:size() do
traindataset[n][2]=traindataset[n][2][1]
end

for m=1, testdataset:size() do
testdataset[m][2]=testdataset[m][2][1]
end


criterion = nn.ClassNLLCriterion()
-- criterion = nn.MSECriterion()

-- TRAINING PHASE
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.1
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

print("Q.2.2: For 2 Layer Neural Network: Testing error is "..testerror)

return trainer, mlp
end
