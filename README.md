### Intrusion and Vulnerability Detection in Software Defined Networks
Problem stated by [ITU AI For Good Global Summit](https://challenge.aiforgood.itu.int/match/matchitem/81) and presented by [ITU & ULAK](https://www.youtube.com/watch?v=zgne_H0Ki7M)
#### Abstract:
Software Defined Networks (SDNs) have revolutionised the way modern networks are managed and orchestrated. This sophisticated infrastructure can provide numerous benefits but at the same time introduce several security challenges. A centralised controller holds the responsibility of managing the network traffic, thus making it an attractive target to attackers. Intrusion detection systems (IDS) play a crucial role in identifying and addressing security threats within the SDN. By utilising machine learning algorithms an anomaly based detection system was established to identify deviation in network behaviour. Five machine learning algorithms were employed to train the SDN-IDS, and ultimately, the most appropriate one was chosen. The SDN-IDS demonstrated an exceptional overall performance, particularly when using the XGBoost Classifier trained with a reduced feature train dataset, reaching 99.9\% accuracy as well as 99.9 F1-score. Furthermore, it exhibited near-perfect performance in identifying the most types of attacks within the traffic data.

####  Software Defined Network Intrusion Detection System (SDN-IDS) Architecture:
Our SDN-IDS utilises a 4-step approach. Firstly the data are pre-processed (cleaned,encoded,normalised), the dimension of the dataset was reduced using a RF feature selection, then data were resampled when necessary. Finally, different ML models are trained and evaluated in order to obtain the best one. 

#### About Machine Learning Models Used:
Five ML models in total were chosen to be used. Decision Trees (DT), Random Forest (RF) and K-Nearest Neighbours (K-NN) were selected
since they have an extensive use in the topic of IDS, are easy to implement and support multi-class classification. Also a Bagging and a Boosting classifier were utilised. 

#### About Resampling Techniques:
Network traffic datasets used for IDS are usually imbalanced. Imbalanced data usually lead to a biased model towards the majority class. 
From our perspective to tackle this problem, resampling techniques such as SMOTE and Tomek’s link  were utilised in order to alleviate data imbalances between classes.

#### Results
Experiment were made using 4 variations of the initial dataset. The first dataset (called baseline) is the cleaned dataset containing all the features. The second dataset (reduced-baseline) contains 34 features extracted using random forest feature importance. The third dataset (resampled_baseline) is a the baseline dataset but contains a lot more instances of minority class data traffic in an attempt to reduce data imbalance. The final dataset is the reduced resampled dataset that contains all the instances from the resampled baseline dataset but contains the 34 features selected in the second dataset. 
 It can be observed that different combination result in better macro performance where different combination results in better weighted performance. XGBoost
model trained with the reduced baseline dataset achieved an accuracy of 99.9% and an F1 score of 99.9 as well. On the other hand the DT model trained with the reduced feature resampled dataset had an accuracy of 87.77% and an F1 score of 87.20.

![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/test_set_performance.PNG)
![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/macro_test_performance.PNG)


The confusion matrices reveal that the XGBoost model faces challenges when classifying the three web attacks and the Infiltration attack. On the other hand, the DT model
exhibits a slight improvement in classifying these attacks but at the cost of a slight decrease in normal traffic classification performance.


|Model |Training Time (seconds)     | Test Time (seconds)    | 
| ------------- |:-----------:|:--------:|
| XGBoost |1755|1.44|
| DecisionTree |485|0.29|
> Training and Testing times for the final 2 models selected.
> 
> Models Trained on Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz , 32GB RAM

![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/xgb_test_reduced_baseline_cf.png)
|Data Traffic | Precision     | Recall    | F1-Score  | Accuracy|FPR| Support|
| ------------- |:-----------:|:---------:|:----------:|:---------:|:---:|:----:|
| Benign     | 1.000     | 1.000          | 1.000       | 0.9993| 0.001      | 410865|
| Bot     | 0.860    | 0.810          | 0.840       | 0.8107|0.00008       | 354|
| DDoS     | 1.000     | 1.000          | 1.000       | 0.9999|0.000008       | 23160|
| DoS Golden Eye     | 1.000     | 1.000          | 1.000       | 0.9957|0.000007       | 1861|
| DoS Hulk   | 1.000     | 1.000          | 1.000       | 0.9998|0.0001       | 41626|
| DoS Slowhttptest   | 0.990     | 0.990          | 0.990       | 0.9899|0.00002       | 994|
| DoS Slowloris     | 1.000     | 1.000          | 1.000       | 0.9981|0.000009       | 1048|
| FTP Patator     | 1.000     | 1.000          | 1.000       | 0.9993|0.000001       | 1436|
| Heartbleed    | 1.000     | 1.000          | 1.000       | 1.000|0       | 2|
| Infiltration    | 1.000     | 0.500          | 0.670      | 0.5000|0       | 6|
| Portscan     | 0.990     | 1.000          | 1.000       | 0.9997|0.0003       | 28728|
| SSH Patator     | 1.000     | 1.000          | 1.000       | 0.9991|0.000005       | 1067|
| Web Attack Brute Force     | 0.730    | 0.820          | 0.770       | 0.8199|0.001       |272 |
| Web Attack Sql Injection    | 0.000     | 0.000          | 0.000       | 0.000|0.000       | 4|
| Web Attack XSS     | 0.470    | 0.320          | 0.380       | 0.3162|0.00008       |117|
> Performance Evaluation Breakdown for every data traffic for XGBoost Model trained of the reduced baseline dataset.

![plot](https://github.com/ITU-AI-ML-in-5G-Challenge/sotirischatzimiltis_sdn_ids/blob/main/Figures/dt_test_reduced_cf.png)

|Data Traffic | Precision     | Recall    | F1-Score  | Accuracy|FPR| Support|
| ------------- |:-----------:|:---------:|:----------:|:---------:|:---:|:----:|
| Benign     | 1.000     | 1.000          | 1.000       | 0.9989| 0.009      | 410865|
| Bot     | 0.680    | 0.930          | 0.780       | 0.9266|0.0003       | 354|
| DDoS     | 1.000     | 1.000          | 1.000       | 0.9998|0.00001       | 23160|
| DoS Golden Eye     | 0.9990     | 1.000     | 0.990       | 0.9973|0.00005      | 1861|
| DoS Hulk   | 1.000     |0.980          | 0.990       | 0.9786|0.0001       | 41626|
| DoS Slowhttptest   | 0.970     | 0.990          | 0.980       | 0.9920|0.00006       | 994|
| DoS Slowloris     | 0.990     | 0.980          | 0.980       | 0.9809|0.00002       | 1048|
| FTP Patator     | 1.000     | 1.000          | 1.000       | 0.9993|0.000001       | 1436|
| Heartbleed    | 1.000     | 1.000          | 1.000       | 1.000|0       | 2|
| Infiltration    | 0.710     | 0.830          | 0.770      | 0.8333|0.000004       | 6|
| Portscan     | 0.990     | 1.000          | 1.000       | 0.9977|0.0004       | 28728|
| SSH Patator     | 1.000     | 1.000          | 1.000       | 0.9991|0.000004       | 1067|
| Web Attack Brute Force     | 0.760    | 0.680          | 0.720       | 0.6801|0.00001       |272 |
| Web Attack Sql Injection    | 1.000     | 0.250          | 0.400       | 0.250|0.000       | 4|
| Web Attack XSS     | 0.400    | 0.510          | 0.450       | 0.5128|0.0001     |117|
> Performance Evaluation Breakdown for every data traffic for DT Model trained of the reduced resampled dataset.


#### Discussion
By observing the two confusion matrices we can easily conclude that XGBoost classifier has a better performance predicting Benign instances, whereas DT give better predictions on minority classes.
A reason can be due to resampling of the minority classes the classifier has more data to be train with leading in a better per-class performance. 


#### Authors 
Sotiris Chatzimiltis, Mohammad Shojafar, Mahdi Boloursaz Mashhadi, and Rahim Tafazolli <br />
5GIC \& 6GIC, Institute for Communication Systems (ICS), University of Surrey, Guildford, UK <br />
sc02449@surrey.ac.uk, m.shojafar@surrey.ac.uk, m.boloursazmashhadi@surrey.ac.uk, r.tafazolli@surrey.ac.uk

