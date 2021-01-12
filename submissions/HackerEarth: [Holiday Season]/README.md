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
| Resnet50 | 40 | Adam | 85.107 | [d57357b](https://github.com/harshraj22/DeepLearning/blob/d57357bb508039b80089c250b0002a2ffce1d106/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
| InceptionResnetV2 | 40 | Adam | 84.747 | [6d73b87](https://github.com/harshraj22/DeepLearning/blob/6d73b874ec1f7573e5c36150a4ecca2c226890d9/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
| [BagNet](https://github.com/wielandbrendel/bag-of-local-features-models) | 40 | Adam | 72.77 | [4f75d9c](https://github.com/harshraj22/DeepLearning/blob/main/submissions/HackerEarth:%20%5BHoliday%20Season%5D/Holiday_Season.ipynb) |
