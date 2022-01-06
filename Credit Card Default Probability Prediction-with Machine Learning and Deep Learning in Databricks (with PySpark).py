# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card Default Probability Prediction- with Machine Learning and Deep Learning in Databricks (with PySpark)
# MAGIC 
# MAGIC ## Introduction
# MAGIC In this project I attempted to predict the default probability for clients in a retail portolio with Dataricks in a PySpark environment. I read the input data from an S3 bucket in my AWS account and wrote some of the outputs to the same location in S3. I employed several Machine Learning techniques (and Deep Learning) to see which model has the best AUROC. I used additional feature engineering techniques such as univariate outlier detection together with winsorizing, normalisation (scaling), PCA for dimensionality reduction and the Yeo-Johnson transform. Due to the high correlation of the billing amount variables between each-other at diferent points in time (autocorrelation), I decided to use PCA to de-correlate them.
# MAGIC 
# MAGIC The last step after creating the final training and test datasets was to develop several default probability prediction models (Machine Learning and Deep Learning models), tune them and assess their discriminative power and compare them in a horse race. I also visualised the Shapley values of the Gradient Booster Classifier and the Extreme Gradient Booster Classifier in order to analyse how each of the variables affects the probability that the client defaults next month.
# MAGIC 
# MAGIC ## Overview¶
# MAGIC 1. Data Exploration and Pre-processing
# MAGIC 2. Feature Engineering and Train-Test Split 
# MAGIC 3. Model Building (Baseline Test Performance)¶
# MAGIC 4. Hyperparameter Tuning (Tuned Model Performance)
# MAGIC 5. Conclusion
# MAGIC 
# MAGIC ##Data
# MAGIC This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
# MAGIC 
# MAGIC The dataset (from UCI) used contains the following variables:
# MAGIC 
# MAGIC * ID: client ID (primary key)
# MAGIC * LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# MAGIC * SEX: Gender (1=male, 2=female)
# MAGIC * EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# MAGIC * MARRIAGE: Marital status (1=married, 2=single, 3=others)
# MAGIC * AGE: Age in years
# MAGIC * PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
# MAGIC * PAY_2: Repayment status in August, 2005 (scale same as above)
# MAGIC * PAY_3:  Repayment status in July, 2005 (scale same as above)
# MAGIC * PAY_4:  Repayment status in June, 2005 (scale same as above)
# MAGIC * PAY_5:  Repayment status in May, 2005 (scale same as above)
# MAGIC * PAY_6:  Repayment status in April, 2005 (scale same as above)
# MAGIC * BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# MAGIC * BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# MAGIC * BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# MAGIC * BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# MAGIC * BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# MAGIC * BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# MAGIC * PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# MAGIC * PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# MAGIC * PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# MAGIC * PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# MAGIC * PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# MAGIC * PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# MAGIC * default.payment.next.month: Default payment (1=yes, 0=no)
# MAGIC 
# MAGIC ## Acknowledgements
# MAGIC Any publications based on this dataset should acknowledge the following:
# MAGIC 
# MAGIC Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
# MAGIC 
# MAGIC The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) at the UCI Machine Learning Repository.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Exploration and Pre-Processing

# COMMAND ----------

# MAGIC %md
# MAGIC ###Reading the data from the S3 bucket (AWS)

# COMMAND ----------

ENCODED_SECRET_KEY = SECRET_KEY.replace("/", "%2F")
AWS_BUCKET_NAME = "databricks-workspace-stack-lambdazipsbucket-1cg83ucq5fbw1"
MOUNT_NAME = "databricks-workspace_credit_card_data"

dbutils.fs.mount(f"s3a://{ACCESS_KEY}:{ENCODED_SECRET_KEY}@{AWS_BUCKET_NAME}",f"/mnt/{MOUNT_NAME}")

# COMMAND ----------

# MAGIC %fs mounts

# COMMAND ----------

data=(spark.read.option("sep",",").option("header",True).option("inferschema",True).csv(f"/mnt/{MOUNT_NAME}/UCI_Credit_Card.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring and pre-processing the data

# COMMAND ----------

display(data)

# COMMAND ----------

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

data.toPandas().info()

# COMMAND ----------

np.any(np.isnan(data.toPandas()))

# COMMAND ----------

# MAGIC %md
# MAGIC There are no missing values.

# COMMAND ----------

data.drop_duplicates()
data.toPandas().describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Higher values in each category seem to increase the risk of a default as an increase in the size of the payment installments and billing amount would increase the likelihood of default.
# MAGIC 
# MAGIC In addition to this, as 90 DPD (also depending on the country and the portolio) leads to an R2 default event and the default.payment.next.month corresponds to a repayment status of 8 (8 months delay aferent to PAY_0-PAY_6), I would assume the re-payment status are the strongest predictors in the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC Because the target variable default.payment.next.month has dots in its name and PySpark does not accept dots in the namings of the columns, I had to replace the "." symbol with "_".

# COMMAND ----------

data = data.toDF(*(c.replace('.', '_') for c in data.columns))
data.show(5)

# COMMAND ----------

permanent_table_name = "UCI_Credit_Card_csv"

data.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC select * from `UCI_Credit_Card_csv`

# COMMAND ----------

from pyspark.sql.functions import col

cat_values=data.select(col("SEX"),col("EDUCATION"),col("MARRIAGE"),col("PAY_0"),col("PAY_2"),col("PAY_3"),col("PAY_4"),col("PAY_5"),col("PAY_6"),col("default_payment_next_month"))
num_values=data.select(col("AGE"),col("LIMIT_BAL"),col("BILL_AMT1"),col("BILL_AMT2"),col("BILL_AMT3"),col("BILL_AMT4"),col("BILL_AMT5"),col("BILL_AMT6"),col("PAY_AMT1"),col("PAY_AMT2"),col("PAY_AMT3"),col("PAY_AMT4"),col("PAY_AMT5"),col("PAY_AMT6"))

# COMMAND ----------

import seaborn as sns 

plt.figure(figsize=(12, 6))

mask = np.triu(np.ones_like(num_values.toPandas().corr(), dtype=np.bool))
heatmap=sns.heatmap(num_values.toPandas().corr(), mask=mask, vmin=-1, vmax=1,annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);


# COMMAND ----------

# MAGIC %md
# MAGIC For the time series variales (BILL_AMT1-BILL_AMT6 and PAY_AMT1-PAY_AMT6) the correlation seems to decrease with distance between months. In general, besides the billing amount variables, the level of correlation is acceptale.

# COMMAND ----------

bill_amt=data[["BILL_AMT1","BILL_AMT2", "BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]]
pmt_amt=data[["PAY_AMT1","PAY_AMT2", "PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]]

# COMMAND ----------

mask = np.triu(np.ones_like(bill_amt.toPandas().corr(), dtype=np.bool))
heatmap=sns.heatmap(bill_amt.toPandas().corr(), mask=mask, vmin=-1, vmax=1,annot=True)
heatmap.set_title('Billing amount Correlation Heatmap', fontdict={'fontsize':14}, pad=12);

# COMMAND ----------

# MAGIC %md
# MAGIC Due to the very high correlations, I decided to apply PCA to the billing amount variables.

# COMMAND ----------

mask = np.triu(np.ones_like(pmt_amt.toPandas().corr(), dtype=np.bool))
heatmap=sns.heatmap(pmt_amt.toPandas().corr(), mask=mask, vmin=-1, vmax=1,annot=True)
heatmap.set_title('Payment amount correlation heatmap', fontdict={'fontsize':14}, pad=12);

# COMMAND ----------

data.groupBy('default_payment_next_month').count().orderBy('count', ascending=False).show()

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.window import Window

data.groupBy('default_payment_next_month').count()\
.withColumn('percentage', f.round(f.col('count') / f.sum('count')\
.over(Window.partitionBy()),3)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC There classes in the target variale seem well balanced. There does not seem to be any need for undersampling nor oversampling.

# COMMAND ----------

cat_values.groupBy('PAY_0').count().orderBy('count', ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Because the repayment status is a strong indicator of the risk of default (like the DPD-Days Past Due), I will check the average re-payment status per default flag. 

# COMMAND ----------

# MAGIC %sql
# MAGIC select default_payment_next_month,round(avg(PAY_0),4) as Avg_PAY_0,round(avg(PAY_2),4) as Avg_PAY_2,round(avg(PAY_3),4) as Avg_PAY_3,round(avg(PAY_4),4) as Avg_PAY_4,round(avg(PAY_5),4) as Avg_PAY_5,round(avg(PAY_6),4) as Avg_PAY_6 from `UCI_Credit_Card_csv` group by default_payment_next_month order by default_payment_next_month					

# COMMAND ----------

# MAGIC %md
# MAGIC Let us have a quick look at the distriutions of each of the variales.

# COMMAND ----------

#distributions for all numeric variables 

for i in num_values.columns:
    plt.hist(num_values.toPandas()[i])
    plt.title(i)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The numeric variables have similar distributions (right skewed).

# COMMAND ----------

for i in cat_values.columns:
    plt.hist(cat_values.toPandas()[i])
    plt.title(i)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC In the next chapter I covered the feature engineering.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering and Train-Test Split
# MAGIC 
# MAGIC First I will create the group values via clustering. Afterwards, I will encode the group variables using WOE encoding. In order to successfully create the clusters, I will follow the following steps:
# MAGIC * Detect and treat outliers
# MAGIC * One-hot-encoding, additional pre-processing and train-test split
# MAGIC * PCA applied to the billing amount variables
# MAGIC * Yeo-Johnson transformation applied to every numeric variable
# MAGIC * Feature scaling and assembling feature vector (final Machine Learning Pipelines)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate outlier detection and treatment
# MAGIC 
# MAGIC I used an [adjusted Tukey fence](https://www.researchgate.net/publication/265731183_A_generalized_boxplot_for_skewed_and_heavy-tailed_distributions) to account for the skewness in the distribution of each of the variables together with winsorizing.

# COMMAND ----------

# import numpy and statsmodels
import numpy as np
from statsmodels.stats.stattools import medcouple
import math

col_names=list(num_values.columns)

#rdd_num=num_values.rdd
data_after_winsorising=data.toPandas()

for i in col_names:
    q1, q2, q3 = data_after_winsorising[i].quantile([0.25,0.5,0.75])
    MC=medcouple(data_after_winsorising[i])
    IQR =q3-q1
    lower_cap=q1-1.5*math.exp(-3.5*MC)*IQR
    upper_cap=q3+1.5*math.exp(4*MC)*IQR
    data_after_winsorising[i]=data_after_winsorising[i].apply(lambda x: upper_cap if x>(upper_cap) else (lower_cap if x<(lower_cap) else x))
    #rdd_num2=rdd_num.map(lambda x: upper_cap if x[i]>(upper_cap) else (lower_cap if x[i]<(lower_cap) else x[i]))

data_after_winsorising = spark.createDataFrame(data_after_winsorising)    
#num_values_winsorized=rdd_num2.toDF(col_names)
#num_values_winsorized.show(10)    
#num_values_winsorized.describe() 

# COMMAND ----------

data_after_winsorising.write.option("header",True).csv(f"/mnt/{MOUNT_NAME}/data_after_winsorising.csv")

# COMMAND ----------

data_after_winsorising=(spark.read.option("sep",",").option("header",True).option("inferschema",True).csv(f"/mnt/{MOUNT_NAME}/data_after_winsorising.csv"))

# COMMAND ----------

display(data_after_winsorising)

# COMMAND ----------

col_names=list(num_values.columns)
data_after_winsorising[col_names].toPandas().describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Overall, a significant decrease in the maximum values afferent to the continuous variables can be observed. The impact of winsorising on the mean and standard deviation is very small.

# COMMAND ----------

# MAGIC %md
# MAGIC ###  One-hot-encoding, additional pre-processing and train-test split

# COMMAND ----------

# MAGIC %md
# MAGIC A simple 70%/30% train-test split was used in PCA, feature scaling and model development. First I joined the numerical and categorical variables.

# COMMAND ----------

categoricalColumns =["EDUCATION", "MARRIAGE"]

for column in categoricalColumns:
    data_after_winsorising = data_after_winsorising.withColumn(column, data_after_winsorising['`{}`'.format(column)].cast('string'))
data_after_winsorising = data_after_winsorising.withColumn("default_payment_next_month", data_after_winsorising['`{}`'.format("default_payment_next_month")].cast('string'))

data_after_winsorising.printSchema()

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
 
from distutils.version import LooseVersion
 
stages = [] # stages in Pipeline
for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
        from pyspark.ml.feature import OneHotEncoderEstimator
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    else:
        from pyspark.ml.feature import OneHotEncoder
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

# COMMAND ----------

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="default_payment_next_month", outputCol="label")
stages += [label_stringIdx]

# COMMAND ----------

partialPipeline = Pipeline().setStages(stages)
preproc_pipeline = partialPipeline.fit(data_after_winsorising)
data_after_preproc = preproc_pipeline.transform(data_after_winsorising)
display(data_after_preproc)

# COMMAND ----------

### Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = data_after_preproc.randomSplit([0.7, 0.3], seed=31)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ###  PCA applied to the billing amount variables

# COMMAND ----------

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
import numpy as np
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import RobustScaler
from pyspark.ml.feature import MinMaxScaler
import pyspark.sql.functions as f
import pyspark.sql.types
import pandas as pd
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

PCA_assembler = VectorAssembler(inputCols=bill_amt.columns, outputCol = "BILL_amt_features")
trainingData_pca_output = PCA_assembler.transform(trainingData)
testData_pca_output = PCA_assembler.transform(testData)

# COMMAND ----------

centre = StandardScaler(inputCol="BILL_amt_features", outputCol="centred_BA_features", withStd=False, withMean=True)

# Compute summary statistics by fitting the StandardScaler
scalerModel = centre.fit(trainingData_pca_output)

# Centre each feature to have zero standard deviation.
trainingData_pca_output = scalerModel.transform(trainingData_pca_output)

testData_pca_output = scalerModel.transform(testData_pca_output)

# COMMAND ----------

pca = PCA(k=6, inputCol = centre.getOutputCol(), outputCol="PCA_features")

pca_model = pca.fit(trainingData_pca_output)
trainingData_pca_output = pca_model.transform(trainingData_pca_output)
testData_pca_output = pca_model.transform(testData_pca_output)

trainingData_pca_output[["PCA_features"]].show(5, truncate = False)
testData_pca_output[["PCA_features"]].show(5, truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC The factor loadings afferent to each billing amount variable along with the amount of explained variance per principal component are printed below.

# COMMAND ----------

pcs=np.round(pca_model.pc.toArray(),4)
df_pc = pd.DataFrame(pcs, columns = ['PC1','PC2','PC3','PC4','PC5','PC6'], index = bill_amt.columns)
df_pc

# COMMAND ----------

np.round(100.00*pca_model.explainedVariance.toArray(),4)

# COMMAND ----------

# MAGIC %md
# MAGIC Because the first principal component accounts for more than 90% of the variability, I decided to drop the rest of the principal components.

# COMMAND ----------

from pyspark.ml.functions import vector_to_array

trainingData_after_pca=trainingData_pca_output.withColumn("BILL_AMT_PC1", vector_to_array("PCA_features")).select(trainingData.columns + [col("BILL_AMT_PC1")[0].alias("BILL_AMT_PC1")]).drop("BILL_AMT1","BILL_AMT2", "BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6")
testData_after_pca=testData_pca_output.withColumn("BILL_AMT_PC1", vector_to_array("PCA_features")).select(testData.columns + [col("BILL_AMT_PC1")[0].alias("BILL_AMT_PC1")]).drop("BILL_AMT1","BILL_AMT2", "BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6")

display(testData_after_pca)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Yeo-Johnson transformation applied to every numeric variable

# COMMAND ----------

# MAGIC %md
# MAGIC Because the features are highly skewed, I tried to make it slightly more normal using a Yeo-Johnson transformation. This should improve the performance of the predictive models.
# MAGIC 
# MAGIC The Yeo-Johnson transformation is an extension of Box-Cox transformation that can handle both positive and negative values.

# COMMAND ----------

from sklearn.preprocessing import PowerTransformer

power = PowerTransformer(method='yeo-johnson', standardize=False)
num_var=["AGE","LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT_PC1"]
cat_var=["ID","SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
add_var=["default_payment_next_month","EDUCATIONIndex","EDUCATIONclassVec","MARRIAGEIndex","MARRIAGEclassVec","label"]

pt_model = power.fit(trainingData_after_pca[num_var].toPandas())
num_train_after_yj=pt_model.transform(trainingData_after_pca[num_var].toPandas())
num_test_after_yj=pt_model.transform(testData_after_pca[num_var].toPandas())

num_train_after_yj=pd.DataFrame(num_train_after_yj,columns=num_var).reset_index(drop=True, inplace=False)
num_test_after_yj=pd.DataFrame(num_test_after_yj,columns=num_var).reset_index(drop=True, inplace=False)


cat_train_after_yj=pd.DataFrame(trainingData_after_pca[cat_var].toPandas(),columns=cat_var).reset_index(drop=True, inplace=False)
cat_test_after_yj=pd.DataFrame(testData_after_pca[cat_var].toPandas(),columns=cat_var).reset_index(drop=True, inplace=False)

train_after_yj=spark.createDataFrame(pd.concat([cat_train_after_yj,num_train_after_yj],axis=1))
test_after_yj=spark.createDataFrame(pd.concat([cat_test_after_yj,num_test_after_yj],axis=1))

from pyspark.sql.functions import monotonically_increasing_id

train_after_yj = train_after_yj.withColumn("row_id", monotonically_increasing_id())
test_after_yj = test_after_yj.withColumn("row_id", monotonically_increasing_id())
train_add_cols = trainingData_after_pca[add_var].withColumn("row_id", monotonically_increasing_id())
test_add_cols = testData_after_pca[add_var].withColumn("row_id", monotonically_increasing_id())

train_data_final = train_after_yj.join(train_add_cols, ("row_id")).drop("row_id")
test_data_final = test_after_yj.join(test_add_cols, ("row_id")).drop("row_id")

display(test_data_final)

# COMMAND ----------

#distributions for all numeric variables 

for i in num_test_after_yj.columns:
    plt.hist(num_test_after_yj[i])
    plt.title(i)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Even though still ar away from normally distributed, the numerical data looks less skewed than before.

# COMMAND ----------

plt.figure(figsize=(12, 6))

mask = np.triu(np.ones_like(num_test_after_yj.corr(), dtype=np.bool))
heatmap=sns.heatmap(num_test_after_yj.corr(), mask=mask, vmin=-1, vmax=1,annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);

# COMMAND ----------

# MAGIC %md
# MAGIC The levels of correlation are now acceptable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature scaling and assembling feature vector (final Machine Learning Pipelines)

# COMMAND ----------

final_stages_std = [] # stages in Pipeline
final_stages_rs = []
final_stages_norm = []

# Feature scaling

num_feat_assembler = VectorAssembler(inputCols= num_var, outputCol = "num_features")
std = StandardScaler(inputCol="num_features", outputCol="std_num_features", withMean=True, withStd=True)
rs = RobustScaler(inputCol="num_features", outputCol="rs_num_features", withCentering=True, withScaling=True)
norm = MinMaxScaler(inputCol="num_features", outputCol="norm_num_features")

final_stages_std += [num_feat_assembler, std]
final_stages_rs += [num_feat_assembler, rs]
final_stages_norm += [num_feat_assembler, norm]

# COMMAND ----------

num_var=["AGE","LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT_PC1"]
categoricalColumns =["EDUCATION", "MARRIAGE"]

# Transform all features into a vector using VectorAssembler

std_assemblerInputs = ["std_num_features"] + [c + "classVec" for c in categoricalColumns] + ["SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
rs_assemblerInputs = ["rs_num_features"] + [c + "classVec" for c in categoricalColumns] + ["SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
norm_assemblerInputs = ["norm_num_features"] + [c + "classVec" for c in categoricalColumns] + ["SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

std_assembler = VectorAssembler(inputCols=std_assemblerInputs, outputCol="features")
rs_assembler = VectorAssembler(inputCols=rs_assemblerInputs, outputCol="features")
norm_assembler = VectorAssembler(inputCols=norm_assemblerInputs, outputCol="features")

final_stages_std += [std_assembler]
final_stages_rs += [rs_assembler]
final_stages_norm += [norm_assembler]

# COMMAND ----------

# Final data with standardisation

std_Pipeline = Pipeline().setStages(final_stages_std)
std_pipelineModel = std_Pipeline.fit(train_data_final)
std_train_data = std_pipelineModel.transform(train_data_final)
std_test_data = std_pipelineModel.transform(test_data_final)
display(std_test_data)

# COMMAND ----------

# Final data with robust scaling

rs_Pipeline = Pipeline().setStages(final_stages_rs)
rs_pipelineModel = rs_Pipeline.fit(train_data_final)
rs_train_data = rs_pipelineModel.transform(train_data_final)
rs_test_data = rs_pipelineModel.transform(test_data_final)
display(rs_test_data)

# COMMAND ----------

# Final data with mormalisation (Min-Max scaling)

norm_Pipeline = Pipeline().setStages(final_stages_norm)
norm_pipelineModel = norm_Pipeline.fit(train_data_final)
norm_train_data = norm_pipelineModel.transform(train_data_final)
norm_test_data = norm_pipelineModel.transform(test_data_final)
display(norm_test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Building (Baseline Test Performance)¶
# MAGIC Before going further, I wanted to see how various different models perform with default parameters. I tried to assess each of the model's perormance to get a baseline. With the baseline, we can see how much tuning improves each of the models. 
# MAGIC 
# MAGIC The baseline model performance (in terms of AUROC) can be found in the table below:
# MAGIC 
# MAGIC | Baseline Model | Training AUROC | Test AUROC |
# MAGIC | --- | --- | --- |
# MAGIC     |Baseline gaussian naive bayes with standardised numeric variables | 45.61% | 46.14%|
# MAGIC     |Baseline gaussian naive bayes with robust scaling | 45.61% | 46.14%|
# MAGIC     |Baseline logistic regression with standardised numeric variables | 64.28% |63.37% |
# MAGIC     |Baseline logistic regression with robust scaling | 64.27% |63.37% |
# MAGIC     |Baseline decision tree classifier with standardised numeric variables |39.74% | 41.51% |
# MAGIC     |Baseline decision tree classifier with robust scaling |39.73% | 41.37% |
# MAGIC     |Baseline random forest classifier with standardised numeric variables (with feature importance)| 65.40% | 65.02% |
# MAGIC     |Baseline random forest classifier with robust scaling (with feature importance)| 65.41% | 65.02% |
# MAGIC     |Baseline linear support vector classifier with standardised numeric variables| 55.27% | 54.90% |
# MAGIC     |Baseline linear support vector classifier with robust scaling| 55.73% | 53.98% |
# MAGIC     |Baseline gradient boosting classifier with standardised numeric variables (including Shapley values)| 70.08% | 65.19% |
# MAGIC     |Baseline gradient boosting classifier with robust scaling (including Shapley values)| 70.33% | 65.04% |
# MAGIC     |Baseline extreme gradient boosting classifier with standardised numeric variables (including Shapley values)| 92.13% | 58.36%  |
# MAGIC     |Baseline extreme gradient boosting classifier with robust scaling (including Shapley values)| 92.13% | 62.76%  |
# MAGIC     |Baseline multilayer perceptron classifier with min-max scaling (1 hidden layer, 18 neurons)| 65.49% | 64.68%  |
# MAGIC     |Baseline multilayer perceptron classifier with min-max scaling (2 hidden layers with 18, respectively 9 neurons)| 66.97% | 64.74%  |

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import GBTClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from pyspark.ml.classification import MultilayerPerceptronClassifier

import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#!pip install shap
import shap

evaluator = BinaryClassificationEvaluator()

# COMMAND ----------

#!pip install tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow
#!pip install hyperopt
from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline Gaussian Naive Bayes with standardised numeric variables

# COMMAND ----------

gnb = NaiveBayes(modelType="gaussian",labelCol="label", featuresCol="features")
model = gnb.fit(std_train_data)
dev=model.transform(std_train_data)
preds=model.transform(std_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline gaussian naive bayes with robust scaling

# COMMAND ----------

gnb = NaiveBayes(modelType="gaussian",labelCol="label", featuresCol="features")
model = gnb.fit(rs_train_data)
dev=model.transform(rs_train_data)
preds=model.transform(rs_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline logistic regression with standardised numeric variables

# COMMAND ----------

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
model = lr.fit(std_train_data)
dev=model.transform(std_train_data)
preds=model.transform(std_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline logistic regression with robust scaling

# COMMAND ----------

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
model = lr.fit(rs_train_data)
dev=model.transform(rs_train_data)
preds=model.transform(rs_test_data)


print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline decision tree classifier with standardised numeric variables

# COMMAND ----------

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
model = dt.fit(std_train_data)
dev=model.transform(std_train_data)
preds=model.transform(std_test_data)


print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline decision tree classifier with robust scaling

# COMMAND ----------

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
model = dt.fit(rs_train_data)
dev=model.transform(rs_train_data)
preds=model.transform(rs_test_data)


print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline random forest classifier with standardised numeric variables (with feature importance)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features",seed=31)
model = rf.fit(std_train_data)
dev=model.transform(std_train_data)
preds=model.transform(std_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# In order to plot the feature importance values, I first need to convert the feature vector to a Numpy array.
from sklearn.ensemble import RandomForestClassifier

std_train_data_pd = std_train_data.toPandas()

features = std_train_data_pd['features']
y_train = std_train_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,std_train_data_pd['features'].iloc[0:1])

series = std_train_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_train_std = np.apply_along_axis(lambda x : x[0], 1, series)

rf_skl = RandomForestClassifier()
rf_skl.fit(x_train_std, y_train.ravel())

feat_importances = pd.Series(rf_skl.feature_importances_, index=["AGE","LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT_PC1","Education_1","Education_2","Education_3","Education_4","Education_5","Education_6","Marriage_1","Marriage_2","Marriage_3","SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"])
feat_importances.nlargest(18).plot(kind='barh')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline random forest classifier with robust scaling (with feature importance)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features",seed=31)
model = rf.fit(rs_train_data)
dev=model.transform(rs_train_data)
preds=model.transform(rs_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# In order to plot the feature importance values, I first need to convert the feature vector to a Numpy array.
from sklearn.ensemble import RandomForestClassifier

rs_train_data_pd = rs_train_data.toPandas()

features = rs_train_data_pd['features']
y_train = rs_train_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,rs_train_data_pd['features'].iloc[0:1])

series = rs_train_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_train_std = np.apply_along_axis(lambda x : x[0], 1, series)

rf_skl = RandomForestClassifier()
rf_skl.fit(x_train_rs, y_train.ravel())

feat_importances = pd.Series(rf_skl.feature_importances_, index=["AGE","LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT_PC1","Education_1","Education_2","Education_3","Education_4","Education_5","Education_6","Marriage_1","Marriage_2","Marriage_3","SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"])
feat_importances.nlargest(18).plot(kind='barh')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline linear support vector classifier with standardised numeric variables

# COMMAND ----------

svc= LinearSVC(labelCol="label", featuresCol="features")
model = svc.fit(std_train_data)
dev=model.transform(std_train_data)
preds=model.transform(std_test_data)


print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline linear support vector classifier with robust scaling

# COMMAND ----------

svc= LinearSVC(labelCol="label", featuresCol="features")
model = svc.fit(rs_train_data)
dev=model.transform(rs_train_data)
preds=model.transform(rs_test_data)


print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline gradient boosting classifier with standardised numeric variables (including Shapley values)

# COMMAND ----------

gbc= GBTClassifier(labelCol="label", featuresCol="features")
model = gbc.fit(std_train_data)
dev=model.transform(std_train_data)
preds=model.transform(std_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# In order to plot the Shapley values, I first need to convert the feature vector to Numpy array.
from sklearn.ensemble import GradientBoostingClassifier

std_train_data_pd = std_train_data.toPandas()

features = std_train_data_pd['features']
y_train = std_train_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,std_train_data_pd['features'].iloc[0:1])

series = std_train_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_train_std = np.apply_along_axis(lambda x : x[0], 1, series)

gbc_skl = GradientBoostingClassifier()
gbc_skl.fit(x_train_std, y_train.ravel())

shap.initjs()
explainer = shap.TreeExplainer(gbc_skl)
shap_values = explainer.shap_values(x_train_std)
shap.summary_plot(shap_values, features=x_train_std, feature_names=["AGE","LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT_PC1","Education_1","Education_2","Education_3","Education_4","Education_5","Education_6","Marriage_1","Marriage_2","Marriage_3","SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline gradient boosting classifier with robust scaling (including Shapley values)

# COMMAND ----------

gbc= GBTClassifier(labelCol="label", featuresCol="features")
model = gbc.fit(rs_train_data)
dev=model.transform(rs_train_data)
preds=model.transform(rs_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# In order to plot the Shapley values, I first need to convert the feature vector to Numpy array.
from sklearn.ensemble import GradientBoostingClassifier

rs_train_data_pd = rs_train_data.toPandas()

features = rs_train_data_pd['features']
y_train = rs_train_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,rs_train_data_pd['features'].iloc[0:1])

series = rs_train_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_train_rs = np.apply_along_axis(lambda x : x[0], 1, series)

gbc_skl = GradientBoostingClassifier()
gbc_skl.fit(x_train_rs, y_train.ravel())

shap.initjs()
explainer = shap.TreeExplainer(gbc_skl)
shap_values = explainer.shap_values(x_train_rs)
shap.summary_plot(shap_values, features=x_train_rs, feature_names=["AGE","LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT_PC1","Education_1","Education_2","Education_3","Education_4","Education_5","Education_6","Marriage_1","Marriage_2","Marriage_3","SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline extreme gradient boosting classifier with standardised numeric variables (including Shapley values)

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

std_test_data_pd = std_test_data.toPandas()

features = std_test_data_pd['features']
y_test = std_test_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,std_test_data_pd['features'].iloc[0:1])

series = std_test_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_test_std = np.apply_along_axis(lambda x : x[0], 1, series)

xgb_skl = XGBClassifier()
xgb_skl.fit(x_train_std, y_train.ravel())

dev = xgb_skl.predict_proba(x_train_std)[:,1]
preds = xgb_skl.predict_proba(x_test_std)[:,1]

print('AUROC on training data: ', roc_auc_score(y_train, dev))
print('AUROC on test data: ', roc_auc_score(y_test, preds))


shap.initjs()
explainer = shap.TreeExplainer(xgb_skl)
shap_values = explainer.shap_values(x_train_std)
shap.summary_plot(shap_values, features=x_train_std, feature_names=["AGE","LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT_PC1","Education_1","Education_2","Education_3","Education_4","Education_5","Education_6","Marriage_1","Marriage_2","Marriage_3","SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline extreme gradient boosting classifier with robust scaling (including Shapley values)

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

rs_test_data_pd = rs_test_data.toPandas()

features = rs_test_data_pd['features']
y_test = rs_test_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,rs_test_data_pd['features'].iloc[0:1])

series = rs_test_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_test = np.apply_along_axis(lambda x : x[0], 1, series)

xgb_skl = XGBClassifier()
xgb_skl.fit(x_train_rs, y_train.ravel())

dev = xgb_skl.predict_proba(x_train_rs)[:,1]
preds = xgb_skl.predict_proba(x_test_rs)[:,1]

print('AUROC on training data: ', roc_auc_score(y_train, dev))
print('AUROC on test data: ', roc_auc_score(y_test, preds))

shap.initjs()
explainer = shap.TreeExplainer(xgb_skl)
shap_values = explainer.shap_values(x_train_rs)
shap.summary_plot(shap_values, features=x_train_rs, feature_names=["AGE","LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT_PC1","Education_1","Education_2","Education_3","Education_4","Education_5","Education_6","Marriage_1","Marriage_2","Marriage_3","SEX","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline multilayer perceptron classifier with min-max scaling (1 hidden layer, 18 neurons)

# COMMAND ----------

mpc= MultilayerPerceptronClassifier(labelCol="label", featuresCol="features",layers=[25,18,2], seed=31)
model = mpc.fit(norm_train_data)
dev=model.transform(norm_train_data)
preds=model.transform(norm_test_data)


print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline multilayer perceptron classifier with min-max scaling (2 hidden layers with 18, respectively 9 neurons)

# COMMAND ----------

mpc= MultilayerPerceptronClassifier(labelCol="label", featuresCol="features",layers=[25,18,9,2], seed=31)
model = mpc.fit(norm_train_data)
dev=model.transform(norm_train_data)
preds=model.transform(norm_test_data)


print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning (Tuned Model Performance)
# MAGIC After getting the baselines, let's see if we can improve on the indivdual model results! I mainly used grid search to tune four of the models (with AUROC>63% on test data). 
# MAGIC 
# MAGIC The tuned model performance (in terms of AUROC) can be found in the table below:
# MAGIC 
# MAGIC | Tuned Model |Training AUROC | Test AUROC |
# MAGIC | --- | --- | --- |
# MAGIC     | Tuned logistic regression with standardised numeric variables | 64.27% | 63.34%|
# MAGIC     | Tuned random forest classifier with robust scaling  | 68.96% | 63.33% |
# MAGIC     | Tuned gradient boosting classifier with standardised numeric variables| 70.08% | 65.19%  |
# MAGIC     | Tuned feed-forward neural networks with min-max scaling (2 hidden layers)| 67.14% | 65.61%  |

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tuned logistic regression with standardised numeric variables

# COMMAND ----------

lr = LogisticRegression()

param_grid = (ParamGridBuilder()
             .addGrid(lr.regParam, np.logspace(-4, 4, 20))
             .addGrid(lr.elasticNetParam, [0.0,1.0])
             .addGrid(lr.maxIter, [20])
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5, seed=31)
 
# Run cross validations
cvModel = cv.fit(std_train_data)

dev=cvModel.transform(std_train_data)
preds=cvModel.transform(std_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tuned random forest classifier with robust scaling 

# COMMAND ----------

# MAGIC %md
# MAGIC In order to narrow down the potential number of configurations used in the grid search, I chose to first loop through several values of each of the individual hyperparameters to see which value leads to the highest accuracy. I chose accuracy instead of the AUROC because the latter is somewhat irresponsive to changes in the values of the hyperparameters.

# COMMAND ----------

#Create training data in Numpy array format

rs_train_data_pd = rs_train_data.toPandas()

features = rs_train_data_pd['features']
y_train = rs_train_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,rs_train_data_pd['features'].iloc[0:1])

series = rs_train_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_train_rs = np.apply_along_axis(lambda x : x[0], 1, series)


#Create test data in Numpy array format
rs_test_data_pd = rs_test_data.toPandas()

features = rs_test_data_pd['features']
y_test = rs_test_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,rs_test_data_pd['features'].iloc[0:1])

series = rs_test_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_test_rs = np.apply_along_axis(lambda x : x[0], 1, series)

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pandas import DataFrame

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
val_results = []
for max_depth in max_depths:
    rf = RandomForestClassifier(random_state=31,n_jobs=-1,max_depth=max_depth).fit(x_train_rs, y_train.ravel())
    score_train=rf.score(x_train_rs, y_train)
    score_val=rf.score(x_test_rs, y_test)
    train_results.append(score_train)
    val_results.append(score_val)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Training accuracy")
line2, = plt.plot(max_depths, val_results, 'r', label="Test accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('Tree depth')
plt.show()

max_depths=DataFrame(max_depths,columns=["Tree depth"])
train_results=DataFrame(train_results,columns=["Training accuracy"])
val_results=DataFrame(val_results,columns=["Test accuracy"])
features_grid=max_depths.join([train_results,val_results],on=None, how="left",sort=False)
features_grid_top5=features_grid.sort_values(by='Test accuracy', ascending=False).head(5)
features_grid_top5

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier()

param_grid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [8, 9, 10])
             .addGrid(rf.maxBins, [32,60])
             .addGrid(rf.numTrees, [5,10])
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5, seed=31)
 
# Run cross validations
cvModel = cv.fit(rs_train_data)

dev=cvModel.transform(rs_train_data)
preds=cvModel.transform(rs_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Tuned gradient boosting classifier with standardised numeric variables

# COMMAND ----------

# MAGIC %md
# MAGIC In order to narrow down the potential number of configurations used in the grid search, I chose to first loop through several values of each of the individual hyperparameters to see which value leads to the highest accuracy. I chose accuracy instead of the AUROC because the latter is somewhat irresponsive to changes in the values of the hyperparameters.

# COMMAND ----------

#Create training data in Numpy array format

std_train_data_pd = std_train_data.toPandas()

features = std_train_data_pd['features']
y_train = std_train_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,std_train_data_pd['features'].iloc[0:1])

series = std_train_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_train_std = np.apply_along_axis(lambda x : x[0], 1, series)


#Create test data in Numpy array format
std_test_data_pd = std_test_data.toPandas()

features = std_test_data_pd['features']
y_test = std_test_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,std_test_data_pd['features'].iloc[0:1])

series = std_test_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_test_std = np.apply_along_axis(lambda x : x[0], 1, series)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from pandas import DataFrame

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
val_results = []
for max_depth in max_depths:
    gbc = GradientBoostingClassifier(random_state=31,max_depth=max_depth).fit(x_train_std, y_train.ravel())
    score_train=gbc.score(x_train_std, y_train)
    score_val=gbc.score(x_test_std, y_test)
    train_results.append(score_train)
    val_results.append(score_val)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Training accuracy")
line2, = plt.plot(max_depths, val_results, 'r', label="Test accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('Tree depth')
plt.show()

max_depths=DataFrame(max_depths,columns=["Tree depth"])
train_results=DataFrame(train_results,columns=["Training accuracy"])
val_results=DataFrame(val_results,columns=["Test accuracy"])
features_grid=max_depths.join([train_results,val_results],on=None, how="left",sort=False)
features_grid_top5=features_grid.sort_values(by='Test accuracy', ascending=False).head(5)
features_grid_top5

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

gbc = GBTClassifier()

param_grid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 3, 4])
             .addGrid(rf.maxBins, [32,60])
             .addGrid(rf.numTrees, [5,10])
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=gbc, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5, seed=31)
 
# Run cross validations
cvModel = cv.fit(std_train_data)

dev=cvModel.transform(std_train_data)
preds=cvModel.transform(std_test_data)

print('AUROC on training data: ', evaluator.evaluate(dev))
print('AUROC on test data: ', evaluator.evaluate(preds))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tuned feed-forward neural networks with min-max scaling (2 hidden layers)

# COMMAND ----------

#Create training data in Numpy array format

norm_train_data_pd = norm_train_data.toPandas()

features = norm_train_data_pd['features']
y_train = norm_train_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,norm_train_data_pd['features'].iloc[0:1])

series = norm_train_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_train_norm = np.apply_along_axis(lambda x : x[0], 1, series)


#Create test data in Numpy array format
norm_test_data_pd = norm_test_data.toPandas()

features = norm_test_data_pd['features']
y_test = norm_test_data_pd['label'].values.reshape(-1,1)

map(lambda x : x,norm_test_data_pd['features'].iloc[0:1])

series = norm_test_data_pd['features'].apply(lambda x : np.array(x.toArray())).to_numpy().reshape(-1,1)
x_test_norm = np.apply_along_axis(lambda x : x[0], 1, series)

# COMMAND ----------

from keras.models import Sequential
from keras.layers import Input,Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

# COMMAND ----------

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
 

checkpoint_path = "/FileStore/tables/keras_checkpoint_weights.ckpt"
 
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)
 

# COMMAND ----------

import keras
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adadelta,Adam,RMSprop
import hyperopt
from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

import sys

space = {   'units1': hp.quniform('units1', 0,38,1),
            'units2': hp.quniform('units2', 0,38,1),
            'activation_1':  hp.choice('activation_1',['relu', 'tanh', 'sigmoid','elu','selu','swish']),
            'activation_2':  hp.choice('activation_2',['relu', 'tanh', 'sigmoid','elu','selu','swish']),
            "learning_rate": hp.choice('learning_rate', [0.001,0.005,0.01]),
            'dropout1': hp.choice('dropout1', [0.10,0.25]),
            'dropout2': hp.choice('dropout2',  [0.10,0.25]),
            'optimizer': hp.choice('optimizer',['Adadelta','Adam','RMSprop'])        
        }

def f_nn(params):   
    print ('Params testing: ', params)
      # Log run information with mlflow.tensorflow.autolog()
        # Select optimizer
    tf.random.set_seed(31)
    optimizer_call = getattr(tf.keras.optimizers, params["optimizer"])
    optimizer = optimizer_call(learning_rate=params["learning_rate"])    
        
    model = Sequential()
    model.add(Dense(units=params['units1'],activation=params['activation_1'], input_dim = x_train_norm.shape[1])) 
    model.add(Dropout(params['dropout1']))

    model.add(Dense(units=params['units2'],activation=params['activation_2'])) 
    model.add(Dropout(params['dropout2'])) 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])


    model.fit(x_train_norm, y_train, epochs=20,validation_split=.2, batch_size=32, callbacks=[model_checkpoint, early_stopping], verbose = 2)
    dev=model.predict(x_train_norm, batch_size = 32, verbose = 0)
    preds =model.predict(x_test_norm, batch_size = 32, verbose = 0)
    acc_train=roc_auc_score(y_train, dev)
    acc_test = roc_auc_score(y_test, preds)
    return {'loss': 1-acc_test, 'status': STATUS_OK}


spark_trials = SparkTrials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=30, trials=spark_trials)
print(hyperopt.space_eval(space, best))

# COMMAND ----------

tf.random.set_seed(31)

def tuned_neural_network():
    model = Sequential()
    model.add(Dense(32,activation="relu",input_dim=x_train_norm.shape[1]))
    model.add(Dense(6, activation="elu"))
    model.add(Dense(1, activation="sigmoid"))
    return model

optimiser=tf.keras.optimizers.RMSprop(0.001)

tuned_neural_network=tuned_neural_network()

tuned_neural_network.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=optimiser,metrics= tf.keras.metrics.AUC())
tuned_neural_network.fit(x_train_norm, y_train,epochs=20,batch_size=32,callbacks=[model_checkpoint,early_stopping],validation_split=.2)

# COMMAND ----------

dev=tuned_neural_network.predict(x_train_norm, batch_size = 32, verbose = 0)
preds =tuned_neural_network.predict(x_test_norm, batch_size = 32, verbose = 0)

acc_train=roc_auc_score(y_train, dev)
acc_test = roc_auc_score(y_test, preds)
print('AUROC on training data: ', acc_train)
print('AUROC on test data: ', acc_test) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC The model that performs the best with min-max scaling is the feed-forward neural network with 2 hidden layer after hyperparameter tuning. The tuned Gradient Boosting Classifier with standardised numeric variables is the second best performing algorithm. According to the Shapley values afferent to the baseline Gradient Boosting Classifier, the repayment status in September 2005 (PMT_0; the most recent time series) and the first principal component of the billing amount variables have the strongest impact on whether a client defaults. The effect of the repayment status variable sharply decreases with the distance in time (the earlier months affect the default proability to a lesser extent). The level of education plays an important role in reducing the default probability. So far, the influence each of the features has on the default probability is more or less as I initially expected.
# MAGIC 
# MAGIC Below, I created the final table that contains the default probability (PD) predictions and saved the final predictions in my S3 bucket in AWS.

# COMMAND ----------

final_predictions=pd.DataFrame(preds,columns=["PD"])
final_predictions=spark.createDataFrame(final_predictions)

from pyspark.sql.functions import monotonically_increasing_id

norm_train_data_v2 = norm_train_data.withColumn("row_id", monotonically_increasing_id())
final_predictions = final_predictions.withColumn("row_id", monotonically_increasing_id())

test_data_with_final_predictions = norm_train_data_v2.join(final_predictions, ("row_id")).drop("row_id")

display(test_data_with_final_predictions)

# COMMAND ----------

final_predictions=test_data_with_final_predictions[["ID","PD"]]
final_predictions.write.option("header",True).csv(f"/mnt/{MOUNT_NAME}/final_predictions.csv")