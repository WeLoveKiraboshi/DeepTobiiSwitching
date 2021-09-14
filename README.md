# Deep Camera Selection with 
**Authors:** [Yuki Saito], [Ryo Hachiuma], [Hideo Saito]  


This is a repository of our paper: "Camera Selection for Occlusion-less Surgery Recording via Training with an Egocentric Camera".

This script is composed of ORB-SLAM2 part, DepthEstimation part, and DenseReconstruction part.


<a href="http://hvrl.ics.keio.ac.jp/saito_y/images/IW-FCV/system_overview.png" target="_blank"><img src="http://hvrl.ics.keio.ac.jp/saito_y/images/IW-FCV/system_overview.png" 
alt="ORB-SLAM2" width="698" height="178" border="50" /></a>



### Related Publications:

Now creating...

[comment]: <> (Yuki Saito, Ryo Hachiuma, and Hideo Saito. **In-Plane Rotation-Aware Monocular Depth Estimation using SLAM**. *International Workshop on Frontiers of Computer Vision&#40;IW-FCV 2020&#41;,* pp. 305-317, 2020. **[PDF]&#40;https://link.springer.com/chapter/10.1007%2F978-981-15-4818-5_23&#41;**.)

# 1. License

If you use our scripts in an academic work, please cite:

    @article{In-PlaneRotationAwareMonoDepth2020,
      title={In-Plane Rotation-Aware Monocular Depth Estimation Using SLAM},
      author={Yuki Saito, Ryo Hachiuna, and Hideo Saito},
      journal={International Conference on Frontiers of Computer Vision},
      pages={305--317},
      publisher={Springer Singapore},
      year={2020}
     }



# 2. Prerequisites
We have tested the library in **Ubuntu **18.04** and **16.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

For other related libaries, please search in official page of ORB-SLAM2 **[Git](https://github.com/raulmur/ORB_SLAM2)**


# 3. How to Run

## Monocular Depth Estimation Module

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```

3. Execute the following command. Change `XXX.yaml` to TUMX.yaml, or OurDataset.yaml for each sequence respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.
```
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/RGB-D/XXX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```


## Dense Reconstruction Module

1. Execute the following command. Change `XXXXX.yaml` to TUM1.yaml, or OurDataset.yaml for each sequence respectively. ALl yaml files are summarized in Examples/rgbd_monodepth folder. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.
```
./Examples/rgbd_monodepth/rgbd_monodepth Vocabulary/ORBvoc.txt Examples/Monocular/XXX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```



# 4. Demo Results

## Monocular Depth Estimation Module

<a href="http://hvrl.ics.keio.ac.jp/saito_y/site/FCV2020.png/" target="_blank"><img src="http://hvrl.ics.keio.ac.jp/saito_y/site/FCV2020.png"
alt="ORB-SLAM2" width="916" height="197" border="30" /></a>

## Dense Reconstruction Module
<a href="http://hvrl.ics.keio.ac.jp/saito_y/images/IW-FCV/DenseReconstruction.png" target="_blank"><img src="http://hvrl.ics.keio.ac.jp/saito_y/images/IW-FCV/DenseReconstruction.png"
alt="ORB-SLAM2" width="648" height="512" border="30" /></a>