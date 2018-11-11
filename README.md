# MIL-for-Breast-Cancer-Histology-Images
Multiple Instance Learning using Bayesian Learning & Self-supervised Learning on features extracted from CNNs.

'Bayesian_MIL' folder contains Python implementation of the algorithm for MIL proposed in "Bayesian multiple instance learning: automatic feature selection and inductive transfer" (link - http://www.engr.uconn.edu/~jinbo/doc/mil_icml2008.pdf). The Report file contains details of the results of MIL on 2 datasets using this algorithm applied on features extracted from an auto-encoder.

'Attention_MIL' folder contains modifications to the implementation of the deep attention based MIL Algorithm proposed in https://arxiv.org/abs/1802.04712
It overfits a lot!

'Self_Supervised_Learning' contains 2 implementations of self-supervised learning using a U-Net type architecture for histology images. 
Self-supervised learning can be used for 2 tasks - firstly, to learn good lower dimensional embeddings which can be used for other tasks such as MIL and secondly, to obtain a good initialization for other tasks such as segmentation. The proxy task of colorization is used as it seems the most suitable for histology images. The 2 provided implementations correspond to 2 different loss functions. The first one is with the cross-entropy loss wherein the output image is discretized into different classes based on the intensity of the pixels, i.e. the entire intensity range is divided into a fixed number of bins and the class of a pixel is determined from the bin number into which the intensity corresponding to that pixel falls. This didn't give very good results ): The second one is with the MS-SSIM loss which gave good results (: 

In order to learn good features to use for MIL, the connections between the up-sampling and down-sampling parts must be removed so that the lowest feature maps learn good representations of the original data. However, in order to obtain a good initialization for tasks such as segmentation, these connections should be obviously kept and self-supervised learning using the MS-SSIM loss does indeed provide a good initialization for segmentation (: More details to be added soon!







