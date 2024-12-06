
# Health-Care-Utilization

This is the final project submission for **CMPT 732 - Big Data Lab 1**.

The dataset used is **Oh Canada! Sample of Canadian FHIR Data** (124 MB), consisting of synthetic Canadian patient data spread across provinces. The dataset can be found here: [Oh Canada! Dataset](https://synthea.mitre.org/downloads).

## Project Overview
The project aims to create a data pipeline that performs the following tasks:
1. ETL to Parquet files
2. Data cleaning
3. Data analysis and visualization
4. Data transformation for machine learning
5. Machine learning

---

## Steps to Set Up and Run on AWS

### S3 Setup
1. Log in with your IAM account.
2. Create an S3 bucket following these instructions: [Assignment 5 Instructions](https://coursys.sfu.ca/2024fa-cmpt-732-g1/pages/Assign5) (if you're not in Greg's course, search for AWS S3 bucket creation steps).

    #### Steps to Create a Bucket:
    - Go to the **S3 Console**.
    - Switch to the **Oregon/us-west-2** region (top-right).
    - In the **Services** menu (top-left), locate **Storage** and click on **S3**.
    - Click **Create bucket** (middle-right), which opens a dialog box.
    - Enter the following in the dialog box:
      - **Bucket name**: Example - `c732-health-care-utilization` (must be unique).
    - Accept all default settings.
    - Scroll to the bottom and click **Create bucket**.
    - A green status bar will confirm successful creation. Close the dialog box by clicking the `X`.

3. Inside the bucket, create the following directories:
   - **data-lake**: To store raw data.
   - **data-warehouse**: To store transformed and analyzed data.
     - Inside `data-warehouse`, create:
       - `etl`: To store transformed Parquet files.
       - `analysis`: To store outputs of data analysis and visualizations.

4. Download the **Oh Canada! Dataset** from [this link](https://synthea.mitre.org/downloads) and upload the `fhir` folder into the `data-lake` directory.

5. Upload the following Python scripts to the `c732-health-care-utilization` bucket:
   - `etl.py`
   - `healthcare_cost_analysis.py`
   - `HealthcareDisparitiesAnalysis.py`
   - `ML.py`

---

### EMR EC2 Cluster Setup
1. Log in with your IAM account.
2. Create an EMR EC2 cluster using the instructions from [Assignment 5 Instructions](https://coursys.sfu.ca/2024fa-cmpt-732-g1/pages/Assign5).

#### Steps to Create a Cluster:
- Switch to the **Oregon/us-west-2** region (top-right).
- Go to the **EMR Console**:
  - Search for **EMR** in the AWS search bar.
  - Select **EMR on EC2** under Clusters.
  - Click **Create cluster**.

#### General Configuration:
- **Cluster name**: Example - `c732-emr-2x-m7.2xl`.
- **Release**: Choose the latest EMR release.
- **Application bundle**: Select **Spark Interactive**.

#### Cluster Configuration:
- **Instance type**: Select `m7a.2xlarge` for both **Primary** and **Core** nodes.
- **Core node count**: 2
- **Task instance group**: Delete it.

#### Networking:
- Select a VPC and Subnet, or use the default options.

#### Security:
- **Key pair for SSH**: No need to specify a key pair.
- **IAM roles**:
  - **Service role**: Create or select `EMR_DefaultRole`.
  - **Instance profile**: Select `EMR_EC2_DefaultRole`.

Click **Create cluster**. It may take 5–15 minutes for the cluster to move from **Starting** to **Running**.

---

### Running the Code
Once the cluster status reads **Waiting** in green, you can submit Spark applications via the **Steps** tab.

---

#### Setting Up the Environment
Add a custom step to install required Python libraries:
- **Step type**: Custom JAR
- **Name**: `load-libraries`
- **JAR location**: `command-runner.jar`
- **Arguments**:
  ```bash
  sudo pip install numpy seaborn sklearn xgboost boto3 fsspec
  ```

This step needs to be executed each time a new cluster is created or cloned.

---

### ETL
To run `etl.py`:
- **Step type**: Spark Application
- **Name**: `etl`
- **Deploy mode**: Client
- **Spark-submit options**:
  ```bash
  --conf spark.dynamicAllocation.enabled=true
  --conf spark.dynamicAllocation.minExecutors=2
  --conf spark.dynamicAllocation.maxExecutors=20
  --conf spark.hadoop.mapreduce.input.fileinputformat.split.minsize=128MB
  ```
- **Application location**: `s3://c732-health-care-utilization/etl.py`
- **Arguments**: input (to read fhir from), output (to store parquet to)
  ```bash
  s3://c732-health-care-utilization/data-lake/fhir/ 
  s3://c732-health-care-utilization/data-warehouse/etl/
  ```

**Expectation**: Resource-type folders (e.g., `patient`, `observation`) should be created, with files saved in the output location.

---

### Healthcare Cost Analysis
To run `healthcare_cost_analysis.py`:
- **Step type**: Spark Application
- **Name**: `cost-analysis`
- **Deploy mode**: Client
- **Spark-submit options**:
  ```bash
  --conf spark.yarn.maxAppAttempts=1
  ```
- **Application location**: `s3://c732-health-care-utilization/healthcare_cost_analysis.py`
- **Arguments**: input (to read parquet from), s3-bucket, output (to store visualizations to)
  ```bash
  s3://c732-health-care-utilization/data-warehouse/etl/ 
  c732-health-care-utilization 
  data-warehouse/analysis/cost_analysis_output
  ```

**Expectation**: Visualizations saved in `cost_analysis_output`. (Saved samples are under `analysis_output/cost_analysis` in this repository.)

---

### Healthcare Disparities Analysis
To run `HealthcareDisparitiesAnalysis.py`:
- **Step type**: Spark Application
- **Name**: `disparities-analysis`
- **Deploy mode**: Client
- **Spark-submit options**:
  ```bash
  --conf spark.yarn.maxAppAttempts=1
  ```
- **Application location**: `s3://c732-health-care-utilization/HealthcareDisparitiesAnalysis.py`
- **Arguments**: input (to read parquet from), s3-bucket, output (to store visualizations to)
  ```bash
  s3://c732-health-care-utilization/data-warehouse/etl/ 
  c732-health-care-utilization 
  data-warehouse/analysis/disparities_analysis
  ```

**Expectation**: Visualizations saved in `disparities_analysis`. (Saved samples are under `analysis_output/disparities_analysis` in this repository.)

---

### Machine Learning
To run `ML.py`:
- **Step type**: Spark Application
- **Name**: `ml`
- **Deploy mode**: Client
- **Spark-submit options**:
  ```bash
  --conf spark.dynamicAllocation.enabled=true
  --conf spark.dynamicAllocation.minExecutors=2
  --conf spark.dynamicAllocation.maxExecutors=20
  --conf spark.hadoop.mapreduce.input.fileinputformat.split.minsize=128MB
  ```
- **Application location**: `s3://c732-health-care-utilization/ML.py`
- **Arguments**: input (to read parquet from)
  ```bash
  s3://c732-health-care-utilization/data-warehouse/etl/
  ```

**Expectation**: Machine learning models are trained successfully. 
