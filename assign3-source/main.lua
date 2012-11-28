--[[
Main file
By Praveen Thirukonda @ New York University

This file is implemented for the assigment 3 of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file contains sample of experiments.
--]]

-- Load required libraries and files
dofile("isolet.lua")
dofile("whitening.lua")
dofile("model.lua")

dofile("RBF.lua")
dofile("MulPos.lua")
dofile("NegExp.lua")

--dofile("mnist.lua")

require("nn")
--require("libnn")
--include("model.lua")

require("libnn")
--include("RBF.lua")
--include("MulPos.lua")
--include("NegExp.lua")
--include("RBF.lua")

-- dofile("Q_4_1_and_2.lua")

function main()

	-- local RBF, parent = torch.class('nn.RBF', 'nn.Module')

   -- 1. Load isolet dataset
    print("Initializing datasets...")
    local data_train_isolet, data_test_isolet = isolet:getDatasets(600,100)
	
    local whitened_data_train_isolet, whitened_data_test_isolet = whitenDatasets(data_train_isolet, data_test_isolet, 100)

	--local data_train_one_vs_all, data_test_one_vs_all = mnist:getDatasets(6000,1000)

--[[
    function printDataSet()
    print("Training set:")
    for i = 1,1 do
      -- print("["..i.."][1]: "..data_train_isolet[i][1])
      print(data_train_isolet[i][1])
      print(data_train_isolet[i][2])
   end
   print("Testing set:")
    for i = 1,data_test_isolet:size() do
      print("["..i.."][1]: "..data_test_isolet[i][1])
      print("["..i.."][2]: "..data_test_isolet[i][2])
   end
    end
 printDataSet()
]]

---[[
    function printWhitenDataSet()
    print("Whitened Training set:")
    for i = 1,1 do
      -- print("["..i.."][1]: "..data_train_isolet[i][1])
      print(whitened_data_train_isolet[i][1])
      print(whitened_data_train_isolet[i][2])
   end
   print("Whitened Testing set:")
    for i = 1,data_test_isolet:size() do
      print("["..i.."][1]: "..whitened_data_test_isolet[i][1])
      print("["..i.."][2]: "..whitened_data_test_isolet[i][2])
   end
    end
 printDataSet()
---]]
--print(data_train_isolet)
-- Logistic Regression code:
--local logisticRegressionTrainer, logisticRegressionMLP, logisticRegressionTestError = logisticRegression(data_train_isolet, data_test_isolet)
-- local twoLayerNNTrainer, twoLayerNNMLP, twoLayerTestError = twoLayerNN(data_train_isolet, data_test_isolet)

--       local rbfModelTrainer, rbfModelNNMLP, rbfModelTestError = rbfModelConstructor(data_train_isolet, data_test_isolet)

--local logisticRegressionTrainer = logisticRegression(data_train_one_vs_all)

-- local logisticRegression_loss_train, logisticRegression_error_train = trainer:train(data_train_isolet) -- train using some examples
-- local logisticRegression_loss_test, logisticRegression_error_test = trainer:test(data_test_isolet) -- test using some datapoints
--print("Pro")
--print(data_train_isolet)
--print("complete dataset size "..data_train_one_vs_all:size())
-- print("1's size "..data_train_isolet[1]:size())
--trainer:train(data_train_one_vs_all)



--print("Logistic Regression: Training loss is "..logisticRegression_loss_train.." Training error is "..logisticRegression_error_train.."Testing loss is "..logisticRegression_loss_test.." Testing error is "..logisticRegression_error_test)

end

main()
