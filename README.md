# NtechLab Test Task

## Task 1
      Task 1 is located in file task1.py
      The idea of solution is that we don't want to have negative prefix in our subarray,  
      unless whole array consists of negative numbers,   
      so we just set our sum to 0 when we see that our prefix is negative   
      and remember boundaries of subarray with max sum
## Task 2
## 1.How to run training:
      Run script:   
        
      python train_net --root_dir *your_data_root_dir* --male_dir_name *name of subfolder in the root folder with male pics* --female_dir_name *with female pics*

## Current best model: 
      best_checkp_gender_effnet-b3_focal.pth, as you can see from the name it is efficientnet-b3 trained with focal loss.
      Accuracy on 20000 validation images is about 0.984 or 98.4%.
      
## Validation on your own data:
      You can use script process.py, the best model already loading there
      
      python process.py --img_folder *your image folder*
      
      Results will be in the same folder as process.py with name "process_results.json"

## About model and training:
      I used efficientnet-b3 as my main model, also I experimented with resnet-18,    
      resnet-34 and efficientnet-b0, there're also quite good but have less accuracy.
      Results are in the graph "Current best model".
      As my optimizer I used Adam with 1e-4 lr and 1e-6 weight decay.
      For loss function I've chosen focal loss because it has nice ability to use hard negative examples,     
      I think it's helped in the training process.   
      As for data preprocessing I've just opened images with PIL,   
      converted them to numpy, added some augmentations like rotation, flips etc. and normalized it and converted to Tensor.
      
## Requirements
      There is a file requirements.txt which you can use to install necessary dependencies,  
      but it's not guaranteed to have all dependencies you need.
      
      Use: pip install -r requirements.txt
