import pandas as pd

from lenskit.batch import predict
from lenskit.metrics.predict import user_metric, rmse, mae
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.als import BiasedMF


comoda = pd.read_csv('LDOS-CoMoDa.csv').rename(columns={"userID": "user", 'itemID': 'item'})

comoda = comoda.groupby('item').filter(lambda x: x['item'].count() >= 2)  # movie_filter
comoda = comoda.groupby('user').filter(lambda x: x['user'].count() >= 5)  # user_filter
comoda = comoda[comoda["age"] > 0]


del comoda["budget"]

train = comoda.sample(frac=0.8, random_state=200)
test = comoda.drop(train.index)
print(train)
print(test)
aBiasedMF = BiasedMF(28, bias=True)
aBiasedMF.fit(train)
preds = predict(aBiasedMF, test)
print("RMSE: ",user_metric(preds, metric=rmse))
print("MAE: ",user_metric(preds, metric=mae))