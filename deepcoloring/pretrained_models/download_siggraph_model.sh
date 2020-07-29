mkdir -p ./checkpoints/siggraph_retrained


curl http://colorization.eecs.berkeley.edu/siggraph/models/pytorch.pth -O ./checkpoints/siggraph_retrained/latest_net_G.pth

mkdir -p ./checkpoints/siggraph_caffemodel

curl http://colorization.eecs.berkeley.edu/siggraph/models/caffemodel.pth -O ./checkpoints/siggraph_caffemodel/latest_net_G.pth