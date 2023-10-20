import pandas as pd
from preps import *
from log_regession import *
from characteristics import *
df = pd.read_csv('diabetes.csv')
#define_distrib(df)
nan_check(df)
cat_features(df)
X = df.drop(['Pregnancies', 'Outcome'], axis = 1)
y = df['Outcome']
X_train, X_test, y_train, y_test = prep_data(X, y)
#print(X_train.head(5))
iters = [10, 100, 500, 1000, 5000, 10000]
rates = [0.1, 0.01, 0.001]
lr = Log_Reg(max_iter=10000, learning_rate=0.001, method='grad_dec')
lr.fit(X_train.values, y_train.values)
y_pred = lr.predict(X_test.values)

acc, precision, recall, f1 = metrics(y_test.values, y_pred)
print(f'Accuracy: {acc} \nPrecision: {precision} \nRecall: {recall} \nF1_score: {f1}')

acc_cur = -1
prec_cur = -1
recall_cur = -1
f1_cur = -1
best_iter = 0
best_rate = 0
acc_best = -1
prec_best = -1
recall_best = -1
f1_best = -1
for it in iters:
    for r in rates:
        lr_t = Log_Reg(max_iter=it, learning_rate=r)
        lr_t.fit(X_train.values, y_train.values)
        y_pr = lr_t.predict(X_test.values)
        acc, precision, recall, f1 = metrics(y_test.values, y_pred)
        print(acc, precision, recall, f1)
        if acc >= acc_cur and precision >= acc_cur and f1 >= f1_cur and recall >= recall_cur:
            best_rate = r
            best_iter = it
            acc_best, prec_best, recall_best, f1_best = acc, precision, recall, f1
            acc_cur, prec_cur, recall_cur, f1_cur = acc, precision, recall, f1


print(best_iter, best_rate)




