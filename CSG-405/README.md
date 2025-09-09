## Dataset Processing Pipeline ##

#### Video Downloading

- ALL Video Filenames:

  ```
  ./all.txt:
  {Filename}_{Split}_{Start_Time}_{End_Time}.mp4
  ```

  ```
  ./all_with_crop.json:
  vid, part, start_time, end_time, crop_box
  ```

  1. download raw videos and audios from youtube based on the given vid using download_videos.py 
  2. clip and crop videos and audios using clip_and_crop_videos.py based on the given start_time, end_time and cropbox
  
#### Skeleton Annotation
- We have uploaded the pose files to huggingface:
   ```
   https://huggingface.co/datasets/galway/CSG-405/tree/main
   ```
- Please download, cat and unzip them to get the pose files. 
  ```
  cat CSG-405-pose.zip.part* > CSG-405-pose.zip
  ```
  
