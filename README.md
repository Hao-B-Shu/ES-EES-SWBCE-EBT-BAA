# ES-EES-architecture-SWBCE-EBT-loss-and-BAA

Integrated codes for ES-EES architectures, SWBCE loss, EBT loss, and BAA. They are the implemented codes for the four papers and integrated for simplicity.

(a) Python Packages: 
      requirements.freeze (Output from 'freeze_requirements.py'):  
      imutils~=0.5.4  
      matplotlib~=3.7.1  
      numpy~=1.24.3  
      opencv_python~=4.8.0.76  
      pipreqs~=0.4.13  
      scipy~=1.7.3  
      torch~=1.13.1  
      torchvision~=0.8.2  

(b) How to use:  
      Use     'Train_Final.py'    to train models.  
      Use   'Predict_Final.py'  to generate and save predictions after obtaining checkpoints.  
      'ES_EES_SWBCE_EBT_BAA.py'      includes ES/EES models, SWBCE/EBT/BAA losses.   
          'Preprocessing_final.py'           is for data preprocessing, such as data augmentation.  
      'HED_Edge_Select.py'/'RCF_Edge_Select.py'/'BDCN_Edge_Select.py'/'Dexi_Edge_Select.py' include previous models and their modifications as extractors for ES-EES.  
      'Matlab' includes the evaluation codes from Matlab.

(c) Checkpoints of each paper:  
ES/EES: https://www.alipan.com/s/tuKMaRwKUpZ  
SWBCE: https://www.alipan.com/s/zqAMQbo49ZG  
EBT: https://www.alipan.com/s/EchYdKeNzwN  
BAA: https://www.alipan.com/s/YQPFSYtBY6i  
Datasets and partitions are in the Tag.

(d) Related Papers:

ES-EES Architecture:  
@Article{S2025ES,  
      title={Boosting Edge Detection with Pixel-wise Feature Selection: The Extractor-Selector Paradigm},   
      author={Hao Shu},  
      year={2025},  
      eprint={2501.02534},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV},  
      url={https://arxiv.org/abs/2501.02534},   
}

SWBCE Loss:  
@Article{S2025SWBCE,  
      title={Rethinking Edge Detection through Perceptual Asymmetry: The SWBCE Loss},   
      author={Hao Shu},  
      year={2025},  
      eprint={2501.13365},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV},  
      url={https://arxiv.org/abs/2501.13365},   
}

EBT Loss:  
@Article{S2025EBT,  
      title={Edge-Boundary-Texture Loss: A Tri-Class Generalization of Weighted Binary Cross-Entropy for Enhanced Edge Detection},   
      author={Hao Shu},  
      year={2025},  
      eprint={2507.06569},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV},  
      url={https://arxiv.org/abs/2507.06569},   
}

BAA Loss:  
@Article{S2025BAA,  
      title={Binarization-Aware Adjuster: Bridging Continuous Optimization and Binary Inference in Edge Detection},   
      author={Hao Shu},  
      year={2025},  
      eprint={2506.12460},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV},  
      url={https://arxiv.org/abs/2506.12460},   
}
