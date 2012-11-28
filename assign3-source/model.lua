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

return trainer, mlp, testerror
end

-- Q2.2.:
function twoLayerNN(traindatasetorig2, testdatasetorig2)

local traindataset2 = {}
for i = 1,traindatasetorig2:size() do
-- Cloning train data instead of referencing, so that the datset can be modified multiple times
traindataset2[i] = {traindatasetorig2[i][1]:clone(), traindatasetorig2[i][2]:clone()}
end
function traindataset2:size() return traindatasetorig2:size() end
function traindataset2:features() return traindatasetorig2:features() end
function traindataset2:classes() return traindatasetorig2:classes() end

local testdataset2 = {}
for i = 1,testdatasetorig2:size() do
-- Cloning test data instead of referencing, so that the datset can be modified multiple times
testdataset2[i] = {testdatasetorig2[i][1]:clone(), testdatasetorig2[i][2]:clone()}
end
function testdataset2:size() return testdatasetorig2:size() end
function testdataset2:features() return testdatasetorig2:features() end
function testdataset2:classes() return testdatasetorig2:classes() end

mlp = nn.Sequential();
inputs = traindataset2:features(); outputs = 1; HUs = traindataset2:classes(); -- parameters
mlp:add(nn.Linear(inputs, HUs*2))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs*2, HUs))
mlp:add(nn.LogSoftMax())

-- Modify Train and Test Dataset to have an integer as predicted class instead of a Tensor.
for n=1, traindataset2:size() do
traindataset2[n][2]=traindataset2[n][2][1]
end

for m=1, testdataset2:size() do
testdataset2[m][2]=testdataset2[m][2][1]
end


criterion = nn.ClassNLLCriterion()
-- criterion = nn.MSECriterion()

-- TRAINING PHASE
trainer = nn.StochasticGradient(mlp, criterion)
--trainer.learningRate = 0.1
trainer.maxIteration = 600
trainer:train(traindataset2)

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

for q=1, testdataset2:size() do

	predictionsTensor = mlp:forward(testdataset2[q][1])
    predictionsTensor:abs()

-- Now we find the actual predcited one by seeing the one with minimum error.
	predictedClass = 0
	minError = 999999
	for w=1, testdataset2:classes() do
     if (predictionsTensor[w]<minError) then 
        minError = predictionsTensor[w]
	    predictedClass = w
	 end
	end

    if (predictedClass == testdataset2[q][2]) then
        print("Correct Prediction")
		testerror = testerror*(q-1)/q
    else
        print("Wrong Prediction")
        print("PredictionsTensor")
		print(predictionsTensor)
        print("Prediction is: "..predictedClass.." Actual class is: "..testdataset2[q][2])
		testerror = (testerror*q-testerror + 1)/q
	end
end

print("Q.2.2: For 2 Layer Neural Network: Testing error is "..testerror)

return trainer, mlp, testerror
end









-- Q3.4.: RBF Model Constructor
function rbfModelConstructor(traindatasetorig, testdatasetorig)

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

-- rbf → mulpos → negexp → linear → logsoftmax

mlp = nn.Sequential();
inputs = traindataset:features(); outputs = 1; HUs = traindataset:classes(); -- parameters
mlp:add(nn.RBF(inputs, HUs))
mlp:add(nn.NegExp())
mlp:add(nn.Linear(HUs, HUs))
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

print("Q.3.4: RBF Model Constructor: Testing error is "..testerror)

return trainer, mlp, testerror
end





-- Q4.2.: Beats All Models
function beatsAllModels(traindatasetorig, testdatasetorig)

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
mlp:add(nn.RBF(inputs, HUs))
mlp:add(nn.NegExp())
mlp:add(nn.Linear(HUs, HUs))
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

print("Q.4.2: Beats All Models: Testing error is "..testerror)

return trainer, mlp, testerror
end







