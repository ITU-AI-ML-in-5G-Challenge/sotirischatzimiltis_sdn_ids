### Intrusion and Vulnerability Detection in Software Defined Networks
Problem stated by [ITU AI For Good Global Summit](https://challenge.aiforgood.itu.int/match/matchitem/81) and presented by [ITU & ULAK](https://www.youtube.com/watch?v=zgne_H0Ki7M)

Dataset available at [Zenodo Dataset](https://zenodo.org/records/13939009?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijg3M2RiZDc1LTQzYTQtNGVmMy1iZTdlLWM4MWE0ZTVmNzY0MSIsImRhdGEiOnt9LCJyYW5kb20iOiJhNWM2NjJmNTQ2NjI3YTU1NTUzNTM5ZTE2MzA3ZTE1NyJ9.fP1GLAQo7o7cCr6CNpffZP0Oso2zZ7OAKMOgfZqcOvKWW3Y66RuS1top0fS9U7jslFw1wwBb8CHuaVR0kX0lSw)
#### Abstract:
Software Defined Networks (SDNs) have revolutionised the way modern networks are managed and
orchestrated. This sophisticated infrastructure can provide numerous benefits but at the same time introduce several
security challenges. A centralised controller holds the responsibility of managing the network traﬀic, thus making it
an attractive target to attackers. Intrusion detection systems (IDS) play a crucial role in identifying and addressing
security threats within the SDN. By utilising machine learning algorithms an anomaly based detection system was
established to identify deviation in network behaviour. Five machine learning algorithms were employed to train the
SDN-IDS, and ultimately, the most appropriate one was chosen. The SDN-IDS demonstrated an exceptional overall
performance, particularly when using the XGBoost Classifier trained with a feature-reduced dataset, reaching 99.9%
accuracy as well as 99.9 F1-score. Moreover, an analysis of the impact of class imbalance practices was conducted to
demonstrate that specific techniques can enhance the classification performance of minority attacks. The Decision Tree
(DT) model, trained on a feature-reduced and resampled dataset using cost-sensitive learning, achieved an impressive
overall performance with 99.87% accuracy and an F1-score of 99.87. Additionally, it demonstrated outstanding
proficiency in identifying 12 out of the 15 possible traﬀic classes

####  Software Defined Network Intrusion Detection System (SDN-IDS) Architecture:
![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/ITU_SDN_IDS.png)
Our SDN-IDS utilises a 4-step approach. Firstly the data are pre-processed (cleaned,encoded,normalised), the dimension of the dataset was reduced using a RF feature selection, then data were resampled when necessary. Finally, different ML models are trained and evaluated in order to obtain the best one. 

#### Dataset:
The dataset was provided by ULAK. After the data cleaning phase the training and test sets are as described in the table below. 
![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/original_dataset.PNG)
#### About Machine Learning Models Used:
Five ML models in total were chosen to be used. Decision Trees (DT), Random Forest (RF) and K-Nearest Neighbours (K-NN) were selected
since they have an extensive use in the topic of IDS, are easy to implement and support multi-class classification. Also a Bagging and a Boosting classifier were utilised. 

#### About Resampling Techniques:
Network traffic datasets used for IDS are usually imbalanced. Imbalanced data usually lead to a biased model towards the majority class. 
From our perspective to tackle this problem, resampling techniques such as SMOTE and Tomek’s link  were utilised in order to alleviate data imbalances between classes.
![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/SMOTE.png)
![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/TOMEK.png)

#### Results

|Model  |Precision  |Recall  |F1-Score  |Accuracy|
| ----- |:---------:|:-------|:--------:|:------:|
|DT| 0.9983| 0.9983| 0.9983| 0.9983|
|RF| 0.9981| 0.9980| 0.9980| 0.9980|
|K-NN| 0.9971| 0.9971| 0.9971| 0.9971|
|Bagging| 0.9984| 0.9984| 0.9984| 0.9984|
|XGBoost| 0.9986| 0.9986| 0.9986 |0.9986|
> SDN-IDS Weighted Average performance evaluation for 5-Fold Cross-validation using the final dataset.

|Model  |Precision  |Recall  |F1-Score  |Accuracy|
| ----- |:---------:|:-------|:--------:|:------:|
|DT| 0.9988| 0.9987| 0.9987| 0.9987|
|RF| 0.9988| 0.9987| 0.9987| 0.9987|
|K-NN| 0.9957| 0.9954| 0.9955| 0.9954|
|Bagging| 0.9986| 0.9986| 0.9986| 0.9986|
|XGBoost| 0.9989| 0.9989| 0.9989| 0.9989|
> SDN-IDS Weighted Average performance evaluation of the Test Set when models were trained with the final dataset.


![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/xgb_test_eval_35_features_rfe_final_cm.png)
> Performance Evaluation Breakdown for every data traffic for XGBoost Model trained of the feature-reduced dataset.

![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/dt_test_eval_35_features_rfe_resampled_cm_best.png)
> Performance Evaluation Breakdown for every data traffic for DT Model trained of the feature-reduced and resampled dataset with cost-sensitive learning.





#### Discussion



#### Authors 
Sotiris Chatzimiltis, Mohammad Shojafar, Mahdi Boloursaz Mashhadi, and Rahim Tafazolli <br />
5GIC \& 6GIC, Institute for Communication Systems (ICS), University of Surrey, Guildford, UK <br />
sc02449@surrey.ac.uk, m.shojafar@surrey.ac.uk, m.boloursazmashhadi@surrey.ac.uk, r.tafazolli@surrey.ac.uk

