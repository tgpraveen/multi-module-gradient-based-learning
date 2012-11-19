function whitenDatasets(train, test, k)
	-- whitentrain = {}
	-- whitentest = {}
	sigmasummationpart = {}
	sigmatrain = {}

	for i =1, train:size()
		sigmasummationpart = torch.mul(train[i][1],train[i][1]:t())
	end

	sigmatrain = (1/train:size()) * sigmasummationpart

	u,s,v = torch.svd(sigmatrain)

	ureduce = {}

	for j = 1,train:features()
		for l = 1,k
			ureduce[j][l] = u[j][l]
		end
	end

	for i=1,train:size()
		train[i][1]=torch.mul(ureduce:t(), train[i][1])
	end
    
    function train:features() return k

	-- Now doing the same thing for testing data set.

    sigmasummationpart2 = {}
	sigmatest = {}

	for i =1, test:size()
		sigmasummationpart2 = torch.mul(test[i][1],test[i][1]:t())
	end

	sigmatest = (1/test:size()) * sigmasummationpart2

	u,s,v = torch.svd(sigmatest)

	ureduce2 = {}

	for j = 1,test:features()
		for l = 1,k
			ureduce2[j][l] = u[j][l]
		end
	end

	for i=1,test:size()
		test[i][1]=torch.mul(ureduce2:t(), test[i][1])
	end
    
    function test:features() return k


	return train, test
end
