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

function main()
   -- 1. Load isolet dataset
   print("Initializing datasets...")
    local data_train_isolet, data_test_isolet = isolet:getDatasets(10,6)
   
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



end

main()
