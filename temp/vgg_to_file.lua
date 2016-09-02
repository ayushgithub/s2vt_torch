require 'torch'
require 'nn'
require 'loadcaffe'
require 'cudnn'


vgg16_proto = '/home1/ayushmn/thesis/mymodels/neuraltalk2/model/VGG_ILSVRC_16_layers_deploy.prototxt'
vgg16_model = '/home1/ayushmn/thesis/mymodels/neuraltalk2/model/VGG_ILSVRC_16_layers.caffemodel'

cnn = loadcaffe.load(vgg16_proto, vgg16_model, 'cudnn')
local cnn_part = nn.Sequential()
for i = 1, #cnn - 3 do
  local layer = cnn:get(i)

  if i == 1 then
    -- convert kernels in first conv layer into RGB format instead of BGR,
    -- which is the order in which it was trained in Caffe
    local w = layer.weight:clone()
    -- swap weights to R and B channels
    print('converting first layer conv filters from BGR to RGB...')
    layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
    layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
  end
  cnn_part:add(layer)
end

cnn = cnn_part
cnn_part = nil

cnn:evaluate()
