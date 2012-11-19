function whitenDatasets(train, test, k)
whitentrain = {}
whitentest = {}
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



return whitenedtrain, whitenedtest
end
