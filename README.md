# SPG-3DVS
The test code for the proposed SPG-3DVS method used for 3D vessel segmention  

Required packages:  
Pytorch: >= 1.1.0  
Python3: >= 3.6  

File contents:  
**test_single.py**: test code for SPG-3DVS.  
**Unet_bidrecionalLSTM_EdgeBranch_test_revised_3con2_Noise.py**： code for the proposed model.  
**dense_121_edge_test_revised_3con.py**: code for the proposed model.  
**utils.py**: other code for running the program.  

How to use the code:  
1. Download the pretrained model for the VesslNN dataset from [here](https://drive.google.com/file/d/1VzICZUf92pclEf0BCDDuCjJD68d_GnYe/view?usp=sharing).  
2. Set the values of "src_folder" and "src_folder_GT" in **test_single.py** to the location where the test images and ground truth images are stored.  
3. Run code: python test_single.py --checkpoint "the/location/where/you/stored/your/pre-trained/model".  


A more elegant version will come soon.
