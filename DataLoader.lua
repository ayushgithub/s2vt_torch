require 'hdf5'
local utils = require 'utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init( opt )
	print('DataLoader loading json file: ', opt.json_file)
  	self.info = utils.read_json(opt.json_file)
  	self.itow = self.info.itow
  	self.vocab_size = utils.count_keys(self.itow)
	print('vocab size is ' .. self.vocab_size)

	print('DataLoader loading h5 file: ', opt.h5_file)
	self.h5_file = hdf5.open(opt.h5_file, 'r')

	local seq_size = self.h5_file:read('/labels'):dataspaceSize()
 	self.seq_length = seq_size[2]

 	self.image_start_ix = self.h5_file:read('/image_start_idx'):all()
 	self.image_end_ix = self.h5_file:read('/image_end_idx'):all()	

 	self.split_ix = {}
 	self.iterators = {}
 	for i,img in pairs(self.info.final_list) do
 	  local split = img.split
 	  if not self.split_ix[split] then
 	    -- initialize new split
 	    self.split_ix[split] = {}
 	    self.iterators[split] = 1
 	  end
 	  table.insert(self.split_ix[split], i)
 	end
 	for k,v in pairs(self.split_ix) do
 	  print(string.format('assigned %d images to split %s', #v, k))
 	end
end


function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.itow
end

function DataLoader:getSeqLength()
  return self.seq_length
end

function DataLoader:getBatch( opt )
	local split = utils.getopt(opt, 'split')
	local batch_size = utils.getopt(opt, 'batch_size', 1)

	local split_ix = self.split_ix[split]
	assert(split_ix, 'split ' .. split .. ' not found.')

	local max_index = #split_ix

	local label_batch = torch.LongTensor(batch_size, self.seq_length)
	local image_batch = torch.Tensor(batch_size,4096)
	for i=1,batch_size do
		local ri = self.iterators[split]
		local ri_next = ri + 1
		if ri_next > max_index then ri_next = 1; wrapped = true end
		self.iterators[split] = ri_next
		ix = split_ix[ri]
		local label = self.h5_file:read('/labels'):partial({ix,ix},{1,self.seq_length})
		label_batch[i] = label

		local ix1 = self.image_start_ix[ix]
		local ix2 = self.image_end_ix[ix]
		image_batch:resize(batch_size*(ix2[1]-ix1[1]+1),4096)
		image_batch = self.h5_file:read('/image_vgg'):partial({ix1[1],ix2[1]},{1,4096})
		-- print(#image_batch)
	end
	local data = {}
	data.images = image_batch
	data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
	data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
	-- data.infos = infos
	return data	
end
	