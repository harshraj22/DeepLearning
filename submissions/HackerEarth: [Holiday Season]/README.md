### Contest link [HackerEarth](https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-holidays/machine-learning/holiday-season-11-2c924626/)

### Summary:
&emsp; Classify images into one of 6 categories.

Image Classification using computer vision

### Data Details:
  - train: Contains 6469 images for 6 classes 
  - test: Contains 3489 images
  
### Evaluation Metric:
&emsp; $$score = 100 * f1-score$$(actual_values, predicted_values, average='weighted')
  
### Tried Models and results:
| Model | Epoch | Optimizer | Score On LeaderBoard |
|-------|-------|-----------|----------------------|
| Resnet50 | 40 | Adam | 85.107 |
| InceptionResnetV2 | 40 | Adam | 84.747 |
