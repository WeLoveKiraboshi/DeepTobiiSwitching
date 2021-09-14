# Camera Selection for Occlusion-less Surgery Recording via Training with an Egocentric Camera
**Authors:** [Yuki Saito], [Ryo Hachiuma], [Hideo Saito], [Hiroki Kajita], [Yoshifumi Takatsume], [Tetsu Hayashida] 


This is a repository of our paper: "Camera Selection for Occlusion-less Surgery Recording via Training with an Egocentric Camera".

<a href="http://hvrl.ics.keio.ac.jp/saito_y/images/IEEEAccess/overview.png" target="_blank"><img src="http://hvrl.ics.keio.ac.jp/saito_y/images/IEEEAccess/overview.png" 
alt="ORB-SLAM2" width="584" height="413" border="50" /></a>



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



# 2. Dataset structure
```
dataset_folder/
  labels/
    surgery_01.csv
    surgery_02.csv
    surgery_04.csv
    surgery_03.csv
  meta/
    meta_file.yml
  frames/
    surgery_01/
      cam1/
        000000.jpg
        ...
      cam2/
      cam3/
      cam4/
      cam5/
      tobii/
    surgery_02/
    surgery_03/
    surgery_04/
```


#3. Training the model
To train the model with configuration file (contrastive_resnet18.yml) in Sequence-Out setting, run the script file below.
```
python switching/train.py --cfg model_01 --cfg contrastive_resnet18 --meta sequence.
```
To train the model with configuration file (contrastive_resnet18.yml) in Surgery-Out setting, run the script file below.
```
python switching/train.py --cfg model_01 --cfg contrastive_resnet18 --meta surgeryoutX.
```


#4. Results
After training the model (with cfg), results are saved with the following folder structure.
```
results/
  contrastive_resnet18/
    models/
    results/
    tb/
```

The testing is done like this.
```
python swtiching/test.py --mode test --cfg contrastive_resnet18 --meta surgeryoutX
```
If you would like to test for all sequences at the same time, type like this (you do not have to indicate meta file)
```
python swtiching/autotest.py --mode test --cfg contrastive_resnet18.
```

# 5. Demo Results

now creating...

[comment]: <> (<a href="http://hvrl.ics.keio.ac.jp/saito_y/site/FCV2020.png/" target="_blank"><img src="http://hvrl.ics.keio.ac.jp/saito_y/site/FCV2020.png")

[comment]: <> (alt="ORB-SLAM2" width="916" height="197" border="30" /></a>)
