

### Multi-view 3D model retrieval based on enhanced detail features with contrastive center loss

This is the code for the paper https://doi.org/10.1007/s11042-022-12281-9 published in  Multimedia Tools and Applications(MTAP) Feb 15,2022.<br>
**Qiang CHEN<sup>1</sup>, Yinong CHEN\*<sup>2</sup><br>**
1.College of Computer and Information Science, Southwest University, Chongqing, 400175, China<br>
2.School of Computing, Informatics and Decision Systems Engineering, Arizona State University, Tempe, AZ, USA<br>
Received
13 May 2021

Revised
17 August 2021

Accepted
14 January 2022

Published
15 February 2022

Created by Qiang Chen, Yinong Chen Southwest University and Arizona State University.


### Introduction
In recent years, 3D model retrieval has become a hot topic. With the development of deep learning technology, many state-of-the-art deep learning based multi-view 3D model retrieval algorithms have emerged. One of the major challenges in view-based 3D model retrieval is how to achieve rotation invariant. MVCNN (Multi-View Convolutional Neural Networks) achieving higher performance while maintaining rotation invariant. However, the element-wise maximum operation across the views leads to the loss of detailed information. To address this problem, in this paper, we use a deep cross-modal learning method to treat the features of different views as different modal features. First, we select two of the views as the input of the deep multimodal learning method. Then we combine the proposed method with an improved contrastive center loss, so that we can align the features in the same subspace and obtain a higher discriminative fused feature. Experimental results show that the training of the proposed CNN (Convolutional Neural Networks) model is based on the existing MVCNN pre-trained model, which takes only 18 epochs to converge, and it obtains 90.07% in terms of mAP (mean average precision) using only the MVCNN as the backbone, which is comparable to the feature fusion algorithm PVRNet (Point-View Relation Neural Network) and much higher than the mAP of MVCNN (80.2%). The experimental results demonstrated that the proposed method avoids explicitly learning the weights for fusion of different view features, while incorporating more details into the 3D model???s final descriptor can improve the retrieval results.<br>

This code is heavily borrowed from PVRNet https://github.com/iMoonLab/PVRNet

### Citation
If you find our work useful in your research, please cite our paper:
https://doi.org/10.1007/s11042-022-12281-9


### Configuration
Code is tested under the environment of Pytorch 1.0.0, Python 3.6.7 and CUDA 10.0 on Ubuntu 16.04.

ModelNet40 dataset and pretrained MVCNN model can be download from https://github.com/iMoonLab/PVRNet <br>
[multi-view(12-view) data](https://drive.google.com/file/d/12JbIPLvcSUsMjxb_CZYXI8xQK2UKosio/view?usp=sharing) for ModelNet40 dataset.<br>
Pretrained Model: [multi-view part(MVCNN)](https://drive.google.com/file/d/1dZG7XojtPS9Cl5aaH4iWXA_N2PximB6i/view?usp=sharing)<br>

### Usage
+ Download data 12_ModelNet40.zip to ./data/
    ```
    mkdir -p data/pc
    #mkdir -p data/12_ModelNet40
    mv 12_ModelNet40.zip ./data/
    unzip 12_ModelNet40.zip

    ```
+ Download pretrained MVCNN model file 'MVCNN-ALEXNET-ckpt.pth' to result/ckpt
    ```mkdir -p data result/ckpt```
    
+ Train Our Network.

    ``` python qchen_train_alexnet_mvcnn_version_my_model_use_mvcnn_extract_different_layers_dscmr_2_4_layer.py```

+ If validate the performance of our pretrained model:
    Download our pretrained model  from <a href='http://computer.swu.edu.cn/r/cms/computer/computer/images/mvcnn_different_layer_test_2_4_dscmr_18.ckpt.zip'>mvcnn_different_layer_test_2_4_dscmr_18.ckpt.pth</a>.Unzip it, and store it in the same directory as codes.
    ```
    python qchen_test_alexnet_mvcnn_version_my_model_use_mvcnn_extract_different_layers_dscmr_2_4_layer.py
    ```

+ Results:<br>
    map_all at epoch : 0.9007452413018803<br>
    map_F2 at epoch : 0.9001967923785649<br>
    map_F4 at epoch : 0.8870950024416322<br>
    mean class accuracy at epoch 999: all 91.24797406807131  @F2 91.00486223662885 @F4 90.55915721231767<br>
<img src='http://computer.swu.edu.cn/r/cms/computer/computer/images/pr_result.png'>

### License
Our code is released under MIT License (see LICENSE file for details).


    
