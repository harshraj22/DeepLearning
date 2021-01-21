### Contest link [HackerEarth](https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-holidays/machine-learning/holiday-season-11-2c924626/)

### Summary:
&emsp; Classify images into one of 6 categories. 

Image Classification using computer vision

### Data Details:
  - train: Contains 6469 images for 6 classes 
  - test: Contains 3489 images
  
### Evaluation Metric:
&emsp;  score = 100 * f1-score(actual_values, predicted_values, average='weighted')
  
### Tried Models and results:
| Model | Epoch | Optimizer | Score On LeaderBoard | Source |
|-------|-------|-----------|----------------------|--------|
| [BagNet](https://github.com/wielandbrendel/bag-of-local-features-models) | 40 | Adam | 72.77 | [4f75d9c](https://github.com/harshraj22/DeepLearning/blob/4f75d9cdce2ca63fe6eea590379336c037d83ef1/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
| DenseNet | 40 | Adam | 87.33 | [2becce6](https://github.com/harshraj22/DeepLearning/blob/2becce64c9311ff83f9bc249d7caed5f504c78d8/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
| InceptionResnetV2 (last 2 layers trainable) | 40 | Adam | 89.58 | [7ca3684](https://github.com/harshraj22/DeepLearning/blob/7ca3684d5c5f914ed0dacfedf4c66d14af607b5e/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
| Resnet50 | 40 | Adam | 85.107 | [d57357b](https://github.com/harshraj22/DeepLearning/blob/d57357bb508039b80089c250b0002a2ffce1d106/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
| Resnet50 (last 2 layers trainable) | 40 | Adam | 88.929 | [ef4357c](https://github.com/harshraj22/DeepLearning/blob/ef4357cc7cd8a8d56ad021573a9a47e56e3d8889/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
| [senet154](https://github.com/osmr/imgclsmob) | 40 | Adam | 79.153 | [623a89a](https://github.com/harshraj22/DeepLearning/blob/623a89a6e79c7e3ebe14a5f4b0b4b688455f92a9/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
| Resnet50 + InceptionResnetV2 | 40 | Adam | 64.833 | [de070e7](https://github.com/harshraj22/DeepLearning/blob/de070e7b54f9c6bfc2b89aed9f57fb60580fd400/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb)
| resnext101_32x8d | 14 | Adam | <strong>90.00</strong> | |

