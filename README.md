# cnnpruner
This is a toy project I use try compress neural network in pytorch. I know the code is messy, but if it helps someone I am happy.

The code is based on this article:
Pruning Convolutional Neural Networks for Resource Efficient Inference
Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz
https://arxiv.org/abs/1611.06440

The biggest difference is that we support more complex neural network structure than the article. We attempt to prun merge and resnet block.

