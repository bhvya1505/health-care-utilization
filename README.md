# health-care-utilization
This is the final project submission for CMPT 732 - Big Data Lab 1

The dataset used is Oh Canada! Sample of Canadian FHIR Data: 124 MB. Synthetic Canadian patients spread across provinces, and can be found here - https://synthea.mitre.org/downloads.

The project aims to create a data pipeline that will achieve the following:
1. The input is semi-structured data - JSON file for each patient. The first step is to transform the data and load it into specific tables in parquet format.
2. Clean the data 
3. Data Analysis and visualization 
4. Data transformation for machine learning 
5. Machine Learning 

### steps -AWS



spark config for running etl
--conf spark.dynamicAllocation.enabled=true
--conf spark.dynamicAllocation.minExecutors=2
--conf spark.dynamicAllocation.maxExecutors=20
--conf spark.hadoop.mapreduce.input.fileinputformat.split.minsize=128MB

--conf spark.yarn.maxAppAttempts=1

s3://c732-health-care-utilization/data-warehouse/etl/ c732-health-care-utilization data-warehouse/analysis/cost_analysis_output
s3://c732-health-care-utilization/data-warehouse/etl/ c732-health-care-utilization data-warehouse/analysis/disparities_analysis
s3://c732-health-care-utilization/data-warehouse/etl/ s3://c732-health-care-utilization/data-warehouse/ml/


You can add a custom step to your EMR cluster to install the required libraries.
Steps:
Go to the EMR Console.
Select your cluster and choose the Steps tab.
Add a new step with the following configurations:
Step type: Custom JAR
JAR location: command-runner.jar
Arguments:
This step will execute the command across the cluster.
bashCopy code
sudo pip install numpy seaborn
sudo pip install sklearn xgboost
sudo pip install boto3 
sudo pip install fsspec