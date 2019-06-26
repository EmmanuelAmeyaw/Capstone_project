
# coding: utf-8

# This is the second assignment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# 
# Just execute all cells one after the other and you are done - just note that in the last one you have to update your email address (the one you've used for coursera) and obtain a submission token, you get this from the programming assignment directly on coursera.
# 
# Please fill in the sections labelled with "###YOUR_CODE_GOES_HERE###"

# In[189]:


get_ipython().system(u'wget https://github.com/IBM/coursera/raw/master/coursera_ml/a2.parquet')


# Now it’s time to have a look at the recorded sensor data. You should see data similar to the one exemplified below….
# 

# In[190]:


df=spark.read.load('a2.parquet')

df.createOrReplaceTempView("df")
spark.sql("SELECT * from df").show()


# Please create a VectorAssembler which consumes columns X, Y and Z and produces a column “features”
# 

# In[191]:


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=['X','Y','Z'],outputCol='features')


# Please instantiate a classifier from the SparkML package and assign it to the classifier variable. Make sure to either
# 1.	Rename the “CLASS” column to “label” or
# 2.	Specify the label-column correctly to be “CLASS”
# 

# In[192]:


from pyspark.ml.classification import LinearSVC

classifier = LinearSVC(maxIter=10, regParam=0.2)

import pyspark.sql.functions as F
df = df.select( '*', F.col('CLASS').alias('label') )


# Let’s train and evaluate…
# 

# In[193]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, classifier])


# In[194]:


model = pipeline.fit(df)


# In[195]:


prediction = model.transform(df)


# In[196]:


prediction.show()


# In[197]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy").setPredictionCol("prediction").setLabelCol("label")
    
binEval.evaluate(prediction) 


# If you are happy with the result (I’m happy with > 0.55) please submit your solution to the grader by executing the following cells, please don’t forget to obtain an assignment submission token (secret) from the Coursera’s graders web page and paste it to the “secret” variable below, including your email address you’ve used for Coursera. (0.55 means that you are performing better than random guesses)
# 

# In[198]:


get_ipython().system(u'rm -Rf a2_m2.json')


# In[199]:


prediction = prediction.repartition(1)
prediction.write.json('a2_m2.json')


# In[200]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/IBM/coursera/master/rklib.py')


# In[201]:


import zipfile

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('a2_m2.json.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('a2_m2.json', zipf)
zipf.close()


# In[202]:


get_ipython().system(u'base64 a2_m2.json.zip > a2_m2.json.zip.base64')


# In[203]:


from rklib import submit
key = "J3sDL2J8EeiaXhILFWw2-g"
part = "G4P6f"
email = 'ameyawemmanuel@rocketmail.com'
secret = 'v254xGFO5XNN4GUg'

with open('a2_m2.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)


# In[207]:


df.show()


# In[208]:


data = df


# In[209]:


from pyspark.ml.linalg import Vectors


# In[210]:


from pyspark.ml.feature import VectorAssembler


# In[211]:


vectorAssembler = VectorAssembler(inputCols=['X','Y','Z'],outputCol='features')


# In[212]:


features_vectorized = vectorAssembler.transform(data)


# In[213]:


features_vectorized.show()


# In[214]:


from pyspark.ml.feature import Normalizer


# In[247]:


normalizer = Normalizer(inputCol='features',outputCol='features_norm', p=2.0)


# In[248]:


normalized_data = normalizer.transform(features_vectorized)


# In[250]:


normalized_data.show()


# In[251]:


#Normalization does not work for NAIVE BAYES .tHIS IMPLIES USE OTHER STANDADIZATION METHODS


# In[218]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=(vectorAssembler, normalizer))

model = pipeline.fit(data)

prediction = model.transform(data)

prediction.show()


# In[219]:


final_data = prediction.drop('CLASS').drop('SENSORID').drop('X').drop('Y').drop('Z').drop('features')


# In[220]:


final_data.show()


# In[223]:


import pyspark.sql.functions as F
final_data = final_data.select( '*', F.col('features_norm').alias('features') ).drop('features_norm')


# In[252]:


final_data.show()


# # Naive Bayes - VIP does not use negative features

# In[225]:


from pyspark.sql.functions import when
final_data2 = final_data.withColumn('label', when(final_data.label == 0, 'zero').otherwise('one')).withColumn('features', final_data.features)


# In[226]:


final_data2.show(3)


# In[270]:


transformed = final_data2[['features','label']]


# In[271]:


# very important step as features cannot be negative in naive bayes
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(transformed)


# In[272]:


# rescale each feature to range [min, max].
scaledData = scalerModel.transform(transformed)
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
scaledData.show(3, False)


# In[273]:


scaledData.show(3, False)


# In[274]:


import pyspark.sql.functions as F
data = scaledData.select( '*').drop('features')


# In[275]:


data.show(2,False)


# In[276]:


import pyspark.sql.functions as F
data = data.select( '*', F.col('scaledFeatures').alias('features') ).drop('scaledFeatures')


# In[277]:


data.show(2, False)


# In[278]:


transformed = data


# In[279]:


from pyspark.ml.linalg import Vectors # !!!!caution: not from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString,StringIndexer, VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[280]:


# Index labels, adding metadata to the label column
labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel').fit(transformed)
labelIndexer.transform(transformed).show(5, False)


# In[265]:


# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =VectorIndexer(inputCol="features",                               outputCol="indexedFeatures",                               maxCategories=4).fit(transformed)
featureIndexer.transform(transformed).show(5, True)


# In[281]:


data.show(2, False)


# In[282]:


# Split the data into training and test sets (40% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

trainingData.show(5,False)
testData.show(5,False)


# In[283]:


from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(featuresCol='indexedFeatures', labelCol='indexedLabel')


# In[284]:


# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


# In[285]:


# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, nb,labelConverter])


# In[286]:


trainingData.show(3,False)


# In[287]:



model = pipeline.fit(trainingData)


# In[288]:


# Make predictions.
predictions = model.transform(testData)
# Select example rows to display.
predictions.select("features","label","predictedLabel").show(5)


# In[289]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("ACCURACY = %g" % (accuracy))


# ## aCCURACY IS POOR FOR NAIVE BAYES AS WELL, WE NEED TO TUNE THE HYPARAMETERS OF THE MODEL
