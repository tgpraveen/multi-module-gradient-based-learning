require 'torch'
--require "NegExp"
require "MulPos"

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local nntest = {}
local nntestx = {}

--[[
function nntest.NegExp()
   print("Came in NegExp")
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
--]]

---[[
function nntest.MulPos()
   -- print("Came in MulPos")
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.MulPos(ini*inj*ink)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')
   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end
---]]


function nntest.Module_getParameters_1()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   local p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'getParameters(): weights wrong')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'getParameters(): bias wrong')
end

function nntest.Module_getParameters_2()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   local p = n:getParameters()

   n:add( nn.Linear(10,10) )
   p = n:getParameters()

   mytester:asserteq((p[{ {111,210} }] - n.modules[2].weight):norm(), 0, 'error when appending new module')
   mytester:asserteq((p[{ {211,220} }] - n.modules[2].bias):norm(), 0, 'error when appending new module')
end

function nntest.Module_getParameters_3()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone() )
   local p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq((p[{ {111,210} }] - n.modules[2].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {211,220} }] - n.modules[2].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq((p[{ {111,210} }] - n.modules[1].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {211,220} }] - n.modules[1].bias):norm(), 0, 'error when using cloning')

   n:reset()

   mytester:assertgt((p[{ {111,210} }] - n.modules[1].weight):norm(), 0, 'error when using cloning')
   mytester:assertgt((p[{ {211,220} }] - n.modules[1].bias):norm(), 0, 'error when using cloning')
end

function nntest.Module_getParameters_4()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone() )
   local p = n:getParameters()

   n:add(nn.Linear(10,10))
   p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq((p[{ {111,210} }] - n.modules[2].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {211,220} }] - n.modules[2].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq((p[{ {221,320} }] - n.modules[3].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {321,330} }] - n.modules[3].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq(p:nElement(), 3*(10*10+10), 'error: incorrect number of elements in flat vector')
end

function nntest.Module_getParameters_5()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone('weight','bias') )
   local p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq((p[{ {1,100} }] - n.modules[2].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[2].bias):norm(), 0, 'error when using cloning+sharing')

   n:reset()

   mytester:asserteq((p[{ {1,100} }] - n.modules[2].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[2].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq(p:nElement(), (10*10+10), 'error: incorrect number of elements in flat vector')
end

function nntest.Module_getParameters_6()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone('weight','bias') )
   local p = n:getParameters()

   n:add(nn.Linear(10,10))
   p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq((p[{ {1,100} }] - n.modules[2].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[2].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq((p[{ {111,210} }] - n.modules[3].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {211,220} }] - n.modules[3].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq(p:nElement(), 2*(10*10+10), 'error: incorrect number of elements in flat vector')
end

function nntest.Module_getParameters_7()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone('weight','bias') )
   local p = n:getParameters()

   n:add(nn.Linear(10,10))
   p = n:getParameters()

   local n1 = nn.Sequential()
   n1:add( nn.Linear(10,10) )

   local n2 = nn.Sequential()
   n2:add( nn.Linear(10,10) )

   local n = nn.Sequential()
   n:add( n1 )
   n:add( n2 )

   local p = n:getParameters()

   local nf = nn.Sequential()
   nf:add( n1 )
   nf:add( nn.Linear(10,1) )

   local p = nf:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n1.modules[1].weight):norm(), 0, 'error when using cloning+partial realloc')
   mytester:asserteq((p[{ {101,110} }] - n1.modules[1].bias):norm(), 0, 'error when using cloning+partial realloc')

   mytester:asserteq((p[{ {111,120} }] - nf.modules[2].weight):norm(), 0, 'error when using cloning+partial realloc')
   mytester:asserteq((p[{ {121,121} }] - nf.modules[2].bias):norm(), 0, 'error when using cloning+partial realloc')

   mytester:asserteq(p:nElement(), 121, 'error: incorrect number of elements in flat vector')
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

-- nn.test(nntest.NegExp())
nn.test(nntest.MulPos())
