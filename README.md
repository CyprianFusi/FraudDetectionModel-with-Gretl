# FraudDetectionModel

In this notebook I use a publickly available dataset 'FraudDetection.xlsx'to train fraud detection models.

* See Notebook **fraud_detection_imbalance_classes** in this repo for **Weighted Logistic Regression**.

# Two Fraud Detection models trained:
1. **Logistic Regression**  
2. **AdaBoost Classifier** 

The accuracy of both the **Logistic Regression** and **AdaBoost Classifier** models using the test data are greater than **85%**!! This suggests that the model is **really good** but can further be improved by tuning some hyper-parameters.

**The CAP curve analysis shows that that if 50% of the data is analysed without the model only 50% of the fraudulent cases would be discovered. But using the model, we would be able to identify more than 90% of the fraudulent cases in the same amount of data!!** 

# Using the model: 
* the amount of **backlog** (incoming and unfinished work) would be **reduced significantly**, 
* the **amount of staff** needed to do the job would be **reduced drastically**, 
* the **processing time** would be **shortened significantly** and 
* **more cases of fraudulent transactions** would be tracked down in a given amount of data processed - **`more than 40% increase in effeciency!`**
