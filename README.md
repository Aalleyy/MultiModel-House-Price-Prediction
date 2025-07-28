#  Multimodal House Price Prediction

This project builds a **multimodal machine learning model** to predict house prices using both:

-  **Tabular data** (bedrooms, bathrooms, area, zipcode)
-  **Image data** (frontal, kitchen, bedroom, bathroom images)

---

## Objective

To improve house price prediction by combining structured attributes and visual cues using CNNs and feature fusion techniques.

---

## Techniques Used

- Convolutional Neural Networks (CNNs)
- Tabular data preprocessing (MinMaxScaler, One-hot encoding)
- Multimodal feature fusion (Mean pooling, Concatenation)
- Regression modeling using:
  -  Deep Neural Networks
  -  Log-transformed price targets
- Evaluation using:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

---

##  Dataset Structure

```bash
Multimodal Housing Price Prediction/
├── Dataset/
│ ├── HousesInfo.txt # Tabular data (id, bed, bath, area, zip, price)
│ ├── 1_frontal.jpg # Each house has 4 images: frontal, kitchen, bedroom, bathroom
│ ├── 1_kitchen.jpg
│ ├── 1_bedroom.jpg
│ └── 1_bathroom.jpg
├── Multimodal-ML_House_Price_Prediction.ipynb # Full multimodal training pipeline
└── README.md
```


---

## Model Architecture

### CNN Feature Extractor
Custom CNN used to extract features from each room image (or optionally, ResNet18).

### Fusion Strategy
- **Mean Pooling** across room features
- Concatenated with tabular features
- Passed to a DNN with Dropout for regression

---

## How It Works

1. **Preprocessing**:
   - Normalize numeric values
   - One-hot encode zipcodes
   - Filter zipcodes with <25 samples

2. **Image Tiling (Optional)**:
   - Combine 4 room images into a 64×64 tile

3. **CNN Feature Extraction**:
   - Extract features for each image using a shared CNN

4. **Model Training**:
   - Inputs: `[CNN(image features), tabular features]`
   - Output: log(price)

5. **Evaluation**:
   - MAE and RMSE on un-logged predictions

---

## Results (Sample)

| Metric | Value |
|--------|-------|
| MAE    | ~$142,000 |
| RMSE   | ~$195,000 |

---

## Future Work

- Replace CNN with pretrained ResNet18 or EfficientNet
- Add room-type attention or fusion weights
- Try tabular + image feature fusion with XGBoost
- Hyperparameter tuning and K-Fold CV

---

## Requirements

- Python ≥ 3.8
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Pandas / NumPy
- Matplotlib
