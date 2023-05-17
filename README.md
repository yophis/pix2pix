# pix2pix in PyTorch
A pytorch implementation of pix2pix paired facade images translation with CycleGANs.\
Implement based on the paper "Image-to-Image Translation with Conditional Adversarial Networks".

# Training result on facade layout data
after 20,000 iterations\
![training result after 20,000 iterations](https://github.com/yophis/pix2pix/assets/62210017/bdce8aa0-de28-45e6-b65b-217c6945bb39)

For training results progression, see [training sample notebook](https://github.com/yophis/pix2pix/blob/main/facade_training_sample.ipynb)\
Note that if available, the notebook utilizes PyTorch's cuDNN benchmark for faster training. In case of highly variable input sizes, the autotuner should be disabled.

# prerequisites
Python, Numpy, Matplotlib, Pytorch, and OpenCV.
