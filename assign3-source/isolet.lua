--[[
Torch/Lua file which will be used for retrieving data from isolet dataset and forming train and test data objects from the dataset.

Original dataset from http://archive.ics.uci.edu/ml/datasets/ISOLET
]]

-- the spambase dataset
isolet = {};

-- The dataset has 7797 rows (given in the dataset's description)
function isolet:size() return 7797 end

-- Each row (observaton) has 617 features
function isolet:features() return 617 end


