# 🌾 Agricultural Yield Prediction Using Machine Learning

A comprehensive machine learning project that predicts crop yield using three different algorithms: **Random Forest**, **Decision Tree**, and **Recurrent Neural Network (LSTM)**. The project includes both regression (precise yield prediction) and classification (yield category prediction) approaches.

## 📊 Project Overview

This project analyzes agricultural data to predict crop yield per hectare using various environmental and farming factors. It provides farmers and agricultural researchers with data-driven insights to optimize crop production.

### Key Features
- **Multiple ML Models**: Random Forest, Decision Tree, and RNN (LSTM)
- **Dual Approach**: Both regression and classification tasks
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, and performance comparisons
- **Data Visualization**: Feature correlations, yield distributions, and model comparisons
- **Automated Data Processing**: Handles missing values and data cleaning
- **Feature Importance Analysis**: Identifies the most impactful farming factors

## 🗂️ Dataset Information

**Dataset**: `agricultural_yield_train.csv`
- **Size**: 16,000 samples
- **Features**: 6 agricultural variables
- **Target**: Yield (kg per hectare)

### Features Description
| Feature | Description | Type |
|---------|-------------|------|
| `Soil_Quality` | Soil quality rating | Float |
| `Seed_Variety` | Type of seed variety used | Integer |
| `Fertilizer_Amount_kg_per_hectare` | Amount of fertilizer applied | Float |
| `Sunny_Days` | Number of sunny days in growing season | Float |
| `Rainfall_mm` | Total rainfall in millimeters | Float |
| `Irrigation_Schedule` | Irrigation schedule type | Integer |
| `Yield_kg_per_hectare` | **Target Variable** - Crop yield | Float |

### Classification Categories
The project automatically creates balanced yield categories based on data distribution:
- **Low Yield**: Bottom 33% of yields
- **Medium Yield**: Middle 33% of yields  
- **High Yield**: Top 33% of yields

## 🤖 Machine Learning Models

### 1. Random Forest
- **Type**: Ensemble method
- **Use Cases**: Both regression and classification
- **Advantages**: Feature importance, handles non-linear relationships
- **Parameters**: 100 estimators, max_depth=10

### 2. Decision Tree
- **Type**: Tree-based algorithm
- **Use Cases**: Both regression and classification
- **Advantages**: Interpretable, handles categorical features
- **Parameters**: max_depth=8, random_state=42

### 3. RNN (LSTM)
- **Type**: Deep learning neural network
- **Use Cases**: Both regression and classification
- **Architecture**: 
  - LSTM layers (64, 32 units)
  - Dropout layers (0.2-0.3)
  - Dense layers for output
- **Training**: 100 epochs, batch_size=16

## 📈 Evaluation Metrics

### Regression Metrics
- **MSE (Mean Squared Error)**: Measures average squared differences
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals
- **MAE (Mean Absolute Error)**: Average absolute differences
- **R² Score**: Coefficient of determination (model fit quality)

### Classification Metrics
- **Accuracy**: Overall correct predictions percentage
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification performance

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Google Colab (recommended) or Jupyter Notebook

### Required Libraries
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### For Google Colab
All required libraries are pre-installed. Simply upload your dataset and run the notebook.

## 🚀 Usage Instructions

### 1. Data Preparation
```python
# Place your dataset at this path in Colab:
/content/agricultural_yield_train.csv
```

### 2. Run the Complete Pipeline
```python
# Execute all cells in the notebook sequentially
# The code will automatically:
# - Load and clean the data
# - Perform exploratory data analysis
# - Train all three models
# - Generate comprehensive results
```

### 3. Key Outputs
- **Data Visualizations**: Distribution plots, correlation matrices
- **Model Performance**: Accuracy scores, R² values, confusion matrices
- **Feature Importance**: Most impactful farming factors
- **Predictions**: Actual vs predicted comparisons

## 📋 File Structure

```
agricultural-yield-prediction/
│
├── README.md                          # Project documentation
├── agricultural_yield_prediction.py   # Main code file
├── agricultural_yield_train.csv       # Dataset
│
├── outputs/
│   ├── model_comparison_plots.png     # Performance visualizations
│   ├── confusion_matrices.png         # Classification results
│   ├── feature_importance.png         # Important factors
│   └── training_history.png           # RNN training progress
│
└── models/
    ├── random_forest_model.pkl        # Saved Random Forest
    ├── decision_tree_model.pkl        # Saved Decision Tree
    └── rnn_model.h5                   # Saved RNN model
```

## 📊 Expected Results

### Typical Performance Range
- **Random Forest**: R² ~ 0.85-0.95, Accuracy ~ 85-95%
- **Decision Tree**: R² ~ 0.75-0.90, Accuracy ~ 80-90%
- **RNN (LSTM)**: R² ~ 0.80-0.92, Accuracy ~ 82-92%

*Results may vary based on data quality and distribution*

### Key Insights
- **Most Important Features**: Usually fertilizer amount, soil quality, and rainfall
- **Best Performing Model**: Typically Random Forest for this type of agricultural data
- **Class Balance**: Automatic percentile-based categorization ensures balanced classes

## 🔍 Model Interpretation

### Feature Importance
The project automatically ranks features by importance:
1. **Fertilizer Amount**: Often the most predictive factor
2. **Soil Quality**: Critical for crop growth
3. **Rainfall**: Essential for plant development
4. **Sunny Days**: Affects photosynthesis
5. **Seed Variety**: Different varieties have different yields
6. **Irrigation Schedule**: Supplemental water management

### Performance Analysis
- **Regression**: Measures how accurately models predict exact yield values
- **Classification**: Evaluates how well models categorize yields into Low/Medium/High
- **Comparison**: Shows which algorithm works best for your specific dataset

### Areas for Improvement
- Additional ML algorithms (XGBoost, SVM, etc.)
- Hyperparameter optimization
- Cross-validation implementation
- Time series analysis for multi-year data
- Ensemble methods combining all models

## 📚 References & Resources

### Agricultural Data Science
- [FAO Agricultural Statistics](http://www.fao.org/faostat/)
- [Precision Agriculture Research](https://www.precisionag.com/)

### Machine Learning
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [Agricultural ML Papers](https://scholar.google.com/scholar?q=machine+learning+crop+yield+prediction)

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:

- **Issues**: Please open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Refer to the code comments for detailed explanations

## 🎯 Future Enhancements

### Planned Features
- [ ] **Model Ensemble**: Combine all models for better predictions
- [ ] **Hyperparameter Tuning**: Automated optimization using GridSearchCV
- [ ] **Cross-Validation**: K-fold validation for robust evaluation
- [ ] **Web Interface**: Flask/Streamlit app for easy predictions
- [ ] **Real-time Predictions**: API endpoint for live yield forecasting
- [ ] **Additional Features**: Weather data integration, satellite imagery
- [ ] **Time Series**: Multi-year yield trend analysis
- [ ] **Economic Analysis**: Cost-benefit optimization

### Research Applications
- **Precision Agriculture**: Field-specific yield optimization
- **Policy Making**: Agricultural planning and resource allocation
- **Climate Impact**: Understanding weather effects on crop yields
- **Supply Chain**: Harvest planning and market predictions

---

## 🌟 Acknowledgments

Special thanks to:
- Agricultural research community for domain knowledge
- Open-source ML libraries (scikit-learn, TensorFlow)
- Data science community for best practices
- Farmers and agricultural experts for real-world insights

---

*Built with ❤️ for sustainable agriculture and data-driven farming*
