--[[
Torch/Lua file which will be used for retrieving data from isolet dataset and forming train and test data objects from the dataset.
Original dataset from http://archive.ics.uci.edu/ml/datasets/ISOLET
]]

-- the spambase dataset
isolet = {};

-- The dataset has 7797 rows (observation)
function isolet:size() return 7797 end

-- The dataset has 6238 rows in train dataset (observation)
function isolet:trainsize() return 6238 end

-- The dataset has rows 1559 in test dataset (observation)
function isolet:testsize() return 1559 end

-- Each row (observaton) has 617 features
function isolet:features() return 617 end

-- We have 26 classes, where the digit i is class (i+1).
function isolet:classes() return 26 end

-- Read csv files from the isolet1+2+3+4.data
function isolet:readFile()
   -- CSV reading using simple regular expression :)
   isolet.orig = {}
   isolet.orig.train = {}
   isolet.orig.test = {}
   local file = 'isolet1+2+3+4.data'
   local fp = assert(io.open (file))
   local csvtable = {}
   for line in fp:lines() do
      local row = {}
      for value in line:gmatch("[^,]+") do
	 -- note: doesn\'t work with strings that contain , values
	 row[#row+1] = value
      end
      csvtable[#csvtable+1] = row
   end
   -- Generating random order
   local rorder = torch.randperm(isolet:trainsize())
   -- iterate over rows
   for i = 1, isolet:trainsize() do	
      -- iterate over columns (1 .. num_features)
      local input = torch.Tensor(isolet:features())
      local output = torch.Tensor(1)
      for j = 1, isolet:features() do
	 -- set entry in feature matrix
	 input[j] = csvtable[i][j]
      end
      -- get class label from last column (num_features+1)
      output[1] = csvtable[i][isolet:features()+1]
      -- it should be class -1 if output is 0
      -- if output[1] == 0 then output[1] = -1 end
      -- Shuffled dataset
      isolet.orig.train[rorder[i]] = {input, output}
   end


   -- Read test dataset file.
   local file2 = 'isolet5.data'
   local fp2 = assert(io.open (file2))
   local csvtable2 = {}
   for line2 in fp2:lines() do
      local row2 = {}
      for value2 in line2:gmatch("[^,]\s+") do
	 -- note: doesn\'t work with strings that contain , values
	 row2[#row2+1] = value2
      end
      csvtable2[#csvtable2+1] = row2
   end
   -- Generating random order
   local rorder2 = torch.randperm(isolet:testsize())
   -- iterate over rows
   for i = 1, isolet:testsize() do	
      -- iterate over columns (1 .. num_features)
      local input = torch.Tensor(isolet:features())
      local output = torch.Tensor(1)
      for j = 1, isolet:features() do
	 -- set entry in feature matrix
	 input[j] = csvtable[i][j]
      end
      -- get class label from last column (num_features+1)
      output[1] = csvtable[i][isolet:features()+1]
      -- it should be class -1 if output is 0
      -- if output[1] == 0 then output[1] = -1 end
      -- Shuffled dataset
      isolet.orig.test[rorder2[i]] = {input, output}
   end
end

-- Split the dataset into two sets train and test
-- spambase:readFile() must have been executed
function isolet:split(train_size, test_size)
   local train = {}
   local test = {}
   function train:size() return train_size end
   function test:size() return test_size end
   function train:features() return isolet:features() end
   function test:features() return isolet:features() end
   function train:classes() return isolet:classes() end
   function test:classes() return isolet:classes() end
   
   -- iterate over rows
   for i = 1,train:size() do
      -- Cloning data instead of referencing, so that the datset can be split multiple times
      train[i] = {isolet.orig.train[i][1]:clone(), isolet.orig.train[i][2]:clone()}
   end
   -- iterate over rows
   for i = 1,test:size() do
      -- Cloning data instead of referencing
      test[i] = {isolet.orig.test[i][1]:clone(), isolet.orig.test[i][2]:clone()}
   end

   return train, test
end

-- Normalize the dataset using training set's mean and std
-- train and test must be returned from isolet:split
function isolet:normalize(train, test)
   -- Allocate mean and variance vectors
   local mean = torch.zeros(train:features())
   local var = torch.zeros(train:features())
   -- Iterative mean computation
   for i = 1,train:size() do
      mean = mean*(i-1)/i + train[i][1]/i
   end
   -- Iterative variance computation
   for i = 1,train:size() do
      var = var*(i-1)/i + torch.pow(train[i][1] - mean,2)/i
   end
   -- Get the standard deviation
   local std = torch.sqrt(var)
   -- If any std is 0, make it 1
   std:apply(function (x) if x == 0 then return 1 end end)
   -- Normalize the training dataset
   for i = 1,train:size() do
      train[i][1] = torch.cdiv(train[i][1]-mean, std)
   end
   -- Normalize the testing dataset
   for i = 1,test:size() do
      test[i][1] = torch.cdiv(test[i][1]-mean, std)
   end

   return train, test
end

-- Add a dimension to the inputs which are constantly 1
-- This is useful to make simple linear modules without thinking about the bias
function isolet:appendOne(train, test)
   -- Sanity check. If dimensions do not match, do nothing.
   if train:features() ~= isolet:features() or test:features() ~= isolet:features() then
      return train, test
   end
   -- Redefine the features() functions
   function train:features() return isolet:features() + 1 end
   function test:features() return isolet:features() + 1 end
   -- Add dimensions
   for i = 1,train:size() do
      train[i][1] = torch.cat(train[i][1], torch.ones(1))
   end
   for i = 1, test:size() do
      test[i][1] = torch.cat(test[i][1], torch.ones(1))
   end
   -- Return them back
   return train, test
end

-- Get the train and test datasets
function isolet:getDatasets(train_size, test_size)
   -- If file not read, read the files
   if isolet.orig == nil then isolet:readFile() end
   -- Split the dataset
   local train, test = isolet:split(train_size, test_size)
   -- Normalize the dataset
   train, test = isolet:normalize(train, test)
   -- Append one to each input
   train, test = isolet:appendOne(train, test)
   -- return train and test datasets
   return train, test
end
