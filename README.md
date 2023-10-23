# catkin_clocs
It is a ROS bag.  An Implementation of Late-Fusion 3D Ojectb Detection Method CLOCs.  
It recieves the detection results from 2D and 3D detectors(for example `YOLOP` and `Pointpillars`）, and 
implements the Object-Level Fusion.

## Dependencies
To run this ROS bag we need `ros-noetic` and` torch` environment.
Install dependencies.

    pip install torch==1.10.0 
    pip install mmcv==2.0.0rc4 
    pip install onnxruntime 
    pip install numpy==1.23.4 
    pip install numba 

##  Run  catkin_clocs

    vim ~/.bashrc & export  PYTHONPATH=$PYTHONPATH:/path/to/your/dependencies
    source ~/.bashrc
    source /opt/ros/noetic/setup.bash
    source /home/data/catkin_clocs/devel/setup.bash
    roslaunch clocs clocs.launch

## Citation
Thanks for the great work by the author.

        @article{pang2020clocs,
          title={CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection},
          author={Pang, Su and Morris, Daniel and Radha, Hayder},
          booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
          year={2020}
          organization={IEEE}
        }

