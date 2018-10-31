
import numpy as np

from helpers import *
from implementations import *
from data import *



#Loading training data
##################################################################
prediction,data,id_ = load_csv_data("train.csv", sub_sample=False)

#Divide training data into subsets
pred_0, pred_1, pred_2, data_0, data_1, data_2, indices_0, indices_1, indices_2 = categorize_data(prediction, data)

#Process data and build model
x, mean_x, std_x = process_data(data)
y, tx = build_model_data(prediction, x)

x0, mean_x0, std_x0 = process_data(data_0)
y0, tx0 = build_model_data(pred_0,x0)

x1, mean_x1, std_x1 = process_data(data_1)
y1, tx1 = build_model_data(pred_1,x1)

x2, mean_x2, std_x2 = process_data(data_2)
y2, tx2 = build_model_data(pred_2,x2)

#Getting weights
##################################################################

#Apply Regularized Logistic Regression for Subset 0
degree0 = 2
poly_tx0 = (build_poly(x0, degree0))
max_iter = 10000
gamma = 0.2
initial_w = np.zeros((poly_tx0.shape[1], 1))
w0, loss0 = reg_logistic_regression(y0, poly_tx0,0.00001,max_iter, gamma)


#Apply Ridge Regression for Subset 1

degree1 = 12
poly_tx1_rid = (build_poly(x1, degree1))
w1, loss1 = ridge_regression(y1, poly_tx1_rid, 0.00016)

#Apply Ridge Regression for Subset 2

degree2 = 12
poly_tx2_rid =build_poly(x2, degree2)
w2, loss2 = ridge_regression(y2, poly_tx2_rid, 7.91e-7)


#Loading testing data
##################################################################

test_label,test_data,test_id_ = load_csv_data("test.csv", sub_sample=False)

#Divide test data into subsets

pred_0_test, pred_1_test, pred_2_test, data_0_test, data_1_test, data_2_test, indices_0_test, indices_1_test, indices_2_test = categorize_data(test_label, test_data)

#Process training data into subsets

data_0_test,_,_ = process_data(data_0_test)
data_1_test,_,_  = process_data(data_1_test)
data_2_test,_,_  = process_data(data_2_test)

#Build model

y0_test, tx0_test = build_model_data(pred_0_test,data_0_test)
y1_test, tx1_test = build_model_data(pred_1_test,data_1_test)
y2_test, tx2_test = build_model_data(pred_2_test,data_2_test)

#Apply polynomial expension

tx0_test = build_poly(data_0_test,degree0)
tx1_test = build_poly(data_1_test,degree1)
tx2_test = build_poly(data_2_test,degree2)

# Label predictions
y_pred0 = predict_labels(w0,tx0_test)
y_pred1 = predict_labels(w1,tx1_test)
y_pred2 = predict_labels(w2,tx2_test)

y_pred1 = np.expand_dims(y_pred1, axis=1)
y_pred2 = np.expand_dims(y_pred2, axis=1)

# Prepare for submission
rows = test_label.shape[0]
labels = decategorize_prediction(rows, y_pred0, y_pred1, y_pred2, indices_0_test, indices_1_test, indices_2_test)


# Create submission file

create_csv_submission(test_id_,labels,"submission.csv")

