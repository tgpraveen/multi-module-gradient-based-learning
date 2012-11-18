--[[
Torch/Lua file which will be used for retrieving data from isolet dataset and forming train and test data objects from the dataset.
]]

-- the spambase dataset
isolet = {};

-- The dataset has 4601 rows (observations) 
function isolet:size() return 4601 end

-- Each row (observaton) has 617 features
function isolet:features() return 617 end




