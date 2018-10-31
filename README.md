# MLProject1 - Higgs Boson
Machine Learning Project 1

### Team Members

Asli Yorusun: asli.yorusun@epfl.ch

Erdem Bocugoz: erdem.bocugoz@epfl.ch

Serif Soner Serbest: serif.serbest@epfl.ch

### Aim :
In this project we predict CERNs simulated particle collision events as either a Higgs Boson signal or background noise as of binary classification, which is a Kaggle 


### Result:
We ranked 34th in Kaggle LeaderBoard among 211 teams,with our score: 0.83224.


### Run
To get the exact results run the "run.py" file.

### Functions

###### Loading data
labels,features,data,id_ = load_csv_data(data_path, sub_sample=False)

###### Divide training data into 3 subsets
3 subsest according to Jet category = categorize_data(prediction, data)

###### Process data and build model
Cleaned and standartized features = process_data(features) 
build_model_data(prediction, x) 

###### Apply Polynomial Expansion 
Polynomially Expanded Features =  build_poly(features, degree) : 


###### Apply Regularized Logistic Regression

weight, loss = reg_logistic_regression(labels,feature_model,lambda,max iteration, gamma)

###### Apply Ridge Regression
weight, loss = ridge_regression(labels, feature_model, lambda)

###### Label predictions
prediction = predict_labels(weight,feature_model)

###### Merge all three categories
labels = decategorize_prediction(row_size, label1, label2, label3, indices1, indices2, indices3)

###### Create submission file
create_csv_submission(test_id_,labels,"submission.csv")


