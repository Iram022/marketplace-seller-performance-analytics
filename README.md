# marketplace-seller-performance-analytics
This project analyzes seller performance in an e-commerce marketplace using the Olist Brazilian E-commerce Dataset.

The goal is to extract insights about marketplace operations and build a machine learning model to predict late delivery risk.

Key Objectives

• Analyze revenue trends across the marketplace
• Identify top performing sellers
• Evaluate delivery delays and logistics performance
• Segment customers using RFM analysis
• Predict late delivery using machine learning

Machine Learning Task
Predicting Late Delivery Risk

The model predicts whether an order will be delivered after the estimated delivery date.

Target variable:

late_delivery
1 → delivered late
0 → delivered on time

Models used:

• Logistic Regression
• Random Forest
• Gradient Boosting

Final model:

Soft Voting Ensemble
Project Pipeline
1 Data Collection
2 Data Cleaning
3 Feature Engineering
4 Business Analytics
5 Machine Learning
6 Model Evaluation
Key Features

Engineered features include:

delivery_delay
shipping_days
product_volume
seller_state
payment_installments
Model Evaluation Metrics
ROC-AUC
Precision
Recall
F1 Score
Confusion Matrix
Project Structure
data/
notebooks/
src/
models/
README.md
Future Improvements
Power BI dashboard
Streamlit analytics application
Real-time seller risk prediction
