require 'torch'
require "NegExp"

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local nntest = {}
local nntestx = {}

function nntest.NegExp()
   -- print("Hiiiii")
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.NegExp()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

mytester:add(nntest)

if not nn then
   require 'nn'
   jac = nn.Jacobian
   mytester:run()
else
   jac = nn.Jacobian
   function nn.test(tests)
      -- randomize stuff
      math.randomseed(os.time())
      mytester:run(tests)
   end
end

nn.test(nntest.NegExp())

