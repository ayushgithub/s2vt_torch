require 'torch'
require 'nn'
require 'DataLoader'
require 'rnn'
local utils = require 'utils'
------------------------------
-- Input arguments
------------------------------

cmd = torch.CmdLine()

cmd:text()
cmd:text('train s2vt model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','dataset/dataset.h5','path to h5 file')
cmd:option('-input_json','dataset/dataset.json','path to json')
cmd:option('-start_from','','path to model checkpoint')

-- Model settings
cmd:option('-rnn_size',512,'size of the lstm hidden vector')
cmd:option('-input_encoding_size',512,'word vector size')



-- Evaluation and Checkpointing
cmd:option('-val_image_use',3200,'how many images to use for validation')
cmd:option('-save_checkpoint_every',2500,'how often to save the checkpoint')
cmd:option('-losses_log_every',25,'how often do we snapshot the loss')

-- misc
cmd:option('-backend','cudnn','nn or cudnn')
cmd:option('-id','','an id identifying this run/job')
cmd:option('-seed',123,'random number generator')
cmd:option('-gpuid',0,'which gpu to use 0 or 1')

cmd:text()

--------------------------------
-- Basic torch Initialization
--------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

--------------------------------
-- Create the dataloader instance
----------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
-- local loader = DataLoader{h5_file = 'dataset/dataset.h5', json_file='dataset/dataset.json'}
data = loader:getBatch{split = 'train'}



