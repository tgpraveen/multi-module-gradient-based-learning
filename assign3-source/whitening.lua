function whitenDatasets(train, test, k)
	-- whitentrain = {}
	-- whitentest = {}
	local sigmasummationpart = torch.zeros(train:features(), train:features())
	local sigmatrain = {}
    

    function makeItTransposable(particulartrainsamplefeatures)
    temptransposable=torch.zeros(train:features(),1)
    for i=1,train:features() do
    temptransposable[i]=particulartrainsamplefeatures[i]
	end
    print("particulartrainsamplefeatures is:")
    print(particulartrainsamplefeatures)
    print("temptransposable is:")
    print(temptransposable)
    print("i is: "..i)
	return temptransposable
    end    


	for i =1, train:size() do
        --print(train[i][1])
        --print("transpose")
        -- tempfortranspose = makeItTransposable(train[i][1])
        -- print(train[i][1]:t())
        -- print(tempfortranspose)
	    -- print(torch.mm(makeItTransposable(train[i][1]),makeItTransposable(train[i][1]):t()))
        -- print("PRINTED transpose above.")
		-- sigmasummationpart = torch.add(sigmasummationpart,torch.mm(train[i][1],train[i][1]:transpose(1,2)))
        -- print(i)
        sigmasummationpart = torch.add(sigmasummationpart,torch.mm(makeItTransposable(train[i][1]),makeItTransposable(train[i][1]):t()))
	end

	sigmatrain = torch.div(sigmasummationpart,train:size())

	local u,s,v = torch.svd(sigmatrain)

	local ureduce = torch.Tensor(train:features(),k)

	for j = 1,train:features() do
		for l = 1,k do
			ureduce[j][l] = u[j][l]
		end
	end

	for i=1,train:size() do
		train[i][1]=torch.mm(makeItTransposable(ureduce:t()), makeItTransposable(train[i][1]))
	end
    
    function train:features() return k
    end
	-- Now doing the same thing for testing data set.

    sigmasummationpart2 = torch.zeros(train:features(), train:features())
	sigmatest = {}

	for i =1, test:size() do
        sigmasummationpart2 = torch.add(sigmasummationpart2,torch.mm(makeItTransposable(test[i][1]),makeItTransposable(test[i][1]):t()))
	end

    sigmatest = torch.div(sigmasummationpart2,test:size())
	u,s,v = torch.svd(sigmatest)

	ureduce2 = {}

	for j = 1,test:features() do
		for l = 1,k do
			ureduce2[j][l] = u[j][l]
		end
	end

	for i=1,test:size() do
		test[i][1]=torch.mm(makeItTransposable(ureduce2):t(), makeItTransposable(test[i][1]))
	end
    
    function test:features() return k
    end

	return train, test
end
