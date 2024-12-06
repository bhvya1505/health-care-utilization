#!/usr/bin/env python
# coding: utf-8

# In[307]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Disparities in Healthcare Analysis").getOrCreate()
from pyspark.sql.functions import year, current_date, concat_ws, when, col, trim, lower,count, mean, stddev, to_timestamp, unix_timestamp, regexp_replace, sum


# In[308]:


#paths=['etl/patient','etl/observation']
input = sys.argv[1]

observation_df = spark.read.parquet("input/observation")
immunization_df = spark.read.parquet("input/immunization")
diagnostic_report_df = spark.read.parquet("input/diagnostic_report")
procedure_df = spark.read.parquet("input/procedure")
careteam_df = spark.read.parquet("input/careteam")
careplan_df = spark.read.parquet("input/careplan")
explanation_of_benefit_df = spark.read.parquet("input/explanation_of_benefit")
claim_df = spark.read.parquet("input/claim")
medication_request_df = spark.read.parquet("input/medication_request")
condition_df = spark.read.parquet("input/condition")
encounter_df = spark.read.parquet("input/encounter")
patient_df = spark.read.parquet("input/patient")


#observation_df.show(5)
# immunization_df.show(5)
# diagnostic_report_df.show(5)
# procedure_df.show(5)
# careteam_df.show(5)
# careplan_df.show(5)
# explanation_of_benefit_df.show(5)
# claim_df.show(5)
# medication_request_df.show(5)
# condition_df.show(5)
# encounter_df.show(5)



# In[309]:


patient_df.show(5)
patient_df.printSchema()


# In[310]:


cities_count = patient_df.select("city").distinct().count()
print(f"Number of cities: {cities_count}")
patient_df.select("city").distinct().show()
patient_df.select("state").distinct().show()

patient_df.select("languages").show()
patient_df.select("marital_status").distinct().show()
patient_df.select("gender").distinct().show()


# In[311]:


patient_df_cleaned = patient_df.select(
    "patient_id",
    "gender",
    "birth_date",
    "state",
    "postal_code",
    "marital_status",
    "disability_adjusted_life_years",
    "quality_adjusted_life_years",
)


# In[312]:


#Creating an age column and populating it for our patient population.
patient_df_cleaned = patient_df_cleaned.withColumn(
    "age",
    year(current_date()) - year(col("birth_date"))
)

patient_df_cleaned.show()


# In[313]:


# Clean and Standardize 'marital_status'

patient_df_cleaned = patient_df_cleaned.withColumn(
    "marital_status",
    trim(lower(col("marital_status")))  # Remove extra spaces and convert to lowercase
)

patient_df_cleaned = patient_df_cleaned.withColumn(
    "marital_status",
    when(col("marital_status") == "m", "Married")
    .when(col("marital_status") == "s", "Single")
    .when(col("marital_status") == "never married", "Never Married")
    .otherwise("Unknown")  # Handle missing or invalid values
)

patient_df_cleaned.select("marital_status").distinct().show()


# In[314]:


null_patient_id = patient_df_cleaned.filter(col("patient_id").isNull()).count()

print(f"null 'patient_id': {null_patient_id}")


# In[315]:


patient_df_cleaned.groupBy("gender").agg(count("*").alias("count")).show()


# In[316]:


age_stats = patient_df_cleaned.select(
    mean("age").alias("average_age"),
    stddev("age").alias("stddev_age")
)
age_stats.show()


# In[317]:


patient_df_cleaned.groupBy("marital_status").agg(count("*").alias("count")).show()


# In[318]:


encounter_df.printSchema()
encounter_df.show(10, truncate=False)


# In[319]:


encounter_df.select("class_code").distinct().show(truncate=False)


# In[320]:


encounter_df.select("status").distinct().show(truncate=False)


# In[321]:


encounter_df_cleaned = encounter_df
encounter_df_cleaned = encounter_df.dropna(subset=["encounter_id", "patient_reference", "start_time"])
encounter_df_cleaned = encounter_df_cleaned.withColumn("start_time", to_timestamp("start_time"))
encounter_df_cleaned = encounter_df_cleaned.withColumn("end_time", to_timestamp("end_time"))
encounter_df_cleaned = encounter_df_cleaned.withColumn(
    "encounter_duration",
    (unix_timestamp("end_time") - unix_timestamp("start_time")) / 60  # Duration in minutes
)


# In[322]:


encounter_df_cleaned = encounter_df_cleaned.withColumn(
    "status",
    when(col("status") == "finished", "Completed")
    .otherwise("In-Progress")
)


# In[323]:


encounter_df_cleaned.select("status").distinct().show() #Probably going to drop if only finished values


# In[324]:


encounter_df_cleaned = encounter_df_cleaned.withColumn(
    "class_code",
    when(col("class_code") == "IMP", "Inpatient")
    .when(col("class_code") == "AMB", "Ambulatory")
    .when(col("class_code") == "EMER", "Emergency")
    .otherwise("Unknown")  # For any unexpected values
)

# Verify the transformation
encounter_df_cleaned.select("class_code").distinct().show()


# In[325]:


encounter_df_cleaned = encounter_df_cleaned.withColumn(
    "patient_reference",
    regexp_replace("patient_reference", "urn:uuid:", "")
)

# Verify the transformation
encounter_df_cleaned.select("patient_reference").distinct().show(truncate=False)
encounter_df_cleaned.show(10, truncate=False)


# In[326]:


encounter_df_cleaned.show(10, truncate=False)


# In[327]:


encounter_df_cleaned.select("patient_reference").distinct().count()
encounter_df_cleaned.select("patient_reference").distinct().show(truncate=False)


# In[328]:


encounter_patient_df = encounter_df_cleaned.join(
    patient_df_cleaned,
    encounter_df_cleaned.patient_reference == patient_df_cleaned.patient_id,
    how="inner"
)


# In[329]:


encounter_patient_df.printSchema()
encounter_patient_df.show(10, truncate=False)


# In[330]:


encounter_patient_df = encounter_patient_df.drop(
    "service_provider_id",
    "service_provider_display",
    "participant_individual_display",
    "participant_individual_reference",
    "participant_period_start",
    "participant_period_end",
    "participant_type_code",
    "participant_type_display"
)


# In[331]:


overlap_count = encounter_df_cleaned.join(
    patient_df_cleaned,
    encounter_df_cleaned.patient_reference == patient_df_cleaned.patient_id,
    how="inner"
).count()
print(f"Number of overlapping records: {overlap_count}")


# In[332]:


encounter_patient_df.groupBy("gender", "class_code").count().show()


# In[333]:


encounter_patient_df.groupBy("gender").agg(mean("encounter_duration").alias("avg_duration")).show()


# In[334]:


from pyspark.sql.functions import when

# Create age groups
encounter_patient_df = encounter_patient_df.withColumn(
    "age_group",
    when(col("age") < 18, "Child")
    .when((col("age") >= 18) & (col("age") < 40), "Young Adult")
    .when((col("age") >= 40) & (col("age") < 65), "Middle Aged")
    .otherwise("Senior")
)

# Average encounter duration by age group
encounter_patient_df.groupBy("age_group").agg(mean("encounter_duration").alias("avg_duration")).show()


# In[335]:


# Display rows with high encounter_duration
encounter_patient_df.orderBy(col("encounter_duration").desc()).show(10, truncate=False)


# In[336]:


# Filter for Inpatient encounters with long durations
inpatient_long_stay = encounter_patient_df.filter((col("class_code") == "Inpatient") & (col("encounter_duration") > 1440))

# Show a few long-stay inpatient records
inpatient_long_stay.orderBy(col("encounter_duration").desc()).show(10, truncate=False)


# In[337]:


# Calculate average and max duration by encounter type
from pyspark.sql.functions import max

encounter_patient_df.groupBy("class_code").agg(
    mean("encounter_duration").alias("avg_duration"),
    max("encounter_duration").alias("max_duration")
).show()



# In[338]:


# Filter out encounters with unrealistic durations
filtered_encounter_patient_df = encounter_patient_df.filter(
    (col("encounter_duration") <= 86400) | (col("class_code") != "Inpatient")
)

filtered_encounter_patient_df.show(10, truncate=False)

#86400 is equal to 60 days in the hospital 


# In[339]:


outliers_df = encounter_patient_df.filter(col("encounter_duration") > 86400)
outliers_df.show()


# In[340]:


# Recalculate average duration by class_code
filtered_encounter_patient_df.groupBy("class_code").agg(
    mean("encounter_duration").alias("avg_duration")
).show()


# In[341]:


# Group by age_group and class_code with filtered data
filtered_encounter_patient_df.groupBy("class_code", "age_group").agg(
    mean("encounter_duration").alias("avg_duration")
).orderBy("class_code", "age_group").show()


# In[342]:


# Group by class_code, age_group, and gender to calculate average duration
filtered_encounter_patient_df.groupBy("class_code", "age_group", "gender").agg(
    mean("encounter_duration").alias("avg_duration")
).orderBy("class_code", "age_group", "gender").show(truncate=False)


# In[343]:


filtered_encounter_patient_df.groupBy("marital_status").agg(
    mean("encounter_duration").alias("avg_duration")
).show()


# In[344]:


filtered_encounter_patient_df.groupBy("marital_status", "gender").agg(
    mean("encounter_duration").alias("avg_duration")
).orderBy("marital_status", "gender").show()


# In[345]:


filtered_encounter_patient_df.groupBy("marital_status", "gender", "age_group").agg(
    mean("encounter_duration").alias("avg_duration")
).orderBy("marital_status", "gender", "age_group").show(truncate=False)


# In[360]:


explanation_of_benefit_df.printSchema()
explanation_of_benefit_df.show(10, truncate=False)


# In[361]:


explanation_of_benefit_df.select("eob_id").distinct().count()


# In[362]:


explanation_of_benefit_df.select(
    mean("total_amount").alias("avg_total_amount"),
    mean("payment_amount").alias("avg_payment_amount"),
    sum("total_amount").alias("total_claimed"),
    sum("payment_amount").alias("total_paid")
).show()


# In[363]:


# explanation_of_benefit_df.select(
#     sum(col("total_amount").isNull().cast("int")).alias("null_total_amount"),
#     sum(col("payment_amount").isNull().cast("int")).alias("null_payment_amount")
# ).show()


# In[365]:


explanation_of_benefit_df.select("total_amount", "payment_amount").show(truncate=False)


# In[367]:


explanation_of_benefit_df.filter(col("total_amount").isNotNull()).select("total_amount").show(truncate=False)


# In[368]:


explanation_of_benefit_df.select("adjudication").show(truncate=False)


# In[369]:


# Print the schema of the adjudication column
explanation_of_benefit_df.select("adjudication").printSchema()


# In[370]:


from pyspark.sql.functions import explode

# Explode the adjudication array
adjudication_exploded_df = explanation_of_benefit_df.select(
    "eob_id",
    "patient_reference",
    explode("adjudication").alias("adjudication_item")
)

# Show some data
adjudication_exploded_df.show(truncate=False)


# In[371]:


# Extract amount and category fields
adjudication_details_df = adjudication_exploded_df.select(
    "eob_id",
    "patient_reference",
    "adjudication_item.amount.value",
    "adjudication_item.amount.currency",
    "adjudication_item.category.coding"
)

# Show the data
adjudication_details_df.show(truncate=False)


# In[374]:


adjudication_details_df.printSchema()


# In[375]:


# Extract the first element of the 'coding' array and its fields
adjudication_details_df = adjudication_details_df.withColumn(
    "category_code",
    col("coding").getItem(0).getField("code")  # Access the 'code' field of the first element
).withColumn(
    "category_display",
    col("coding").getItem(0).getField("display")  # Access the 'display' field of the first element
)

# Show the resulting DataFrame
adjudication_details_df.select("category_code", "category_display").show(truncate=False)


# In[378]:


adjudication_with_claims_df = adjudication_details_df.join(
    explanation_of_benefit_df.select("eob_id", "insurance_coverage_display", "billable_period_start"),
    ["eob_id"],
    how="inner"
)

# Inspect the joined DataFrame
adjudication_with_claims_df.show(truncate=False)


# In[379]:


adjudication_with_claims_cleaned_df = adjudication_with_claims_df.drop("coding", "category_code", "insurance_coverage_display")
adjudication_with_claims_cleaned_df.show(truncate=False)
adjudication_with_claims_cleaned_df.printSchema()


# In[393]:


eob_final_df_cleaned = adjudication_with_claims_cleaned_df.withColumn(
    "patient_reference",
    regexp_replace("patient_reference", "urn:uuid:", "")
)

# Verify the transformation
eob_final_df_cleaned.select("patient_reference").distinct().show(truncate=False)
eob_final_df_cleaned.show(10, truncate=False)


# In[394]:


# Join adjudication data with demographic information
eob_with_demographics_df = eob_final_df_cleaned.join(
    filtered_encounter_patient_df.select("patient_reference", "gender", "age", "marital_status", "age_group"),
    ["patient_reference"],
    how="inner"
)

# Verify the joined DataFrame
eob_with_demographics_df.printSchema()
eob_with_demographics_df.show(truncate=False)


# In[395]:


eob_with_demographics_df.select("category_display").distinct().show(truncate=False)


# In[396]:


eob_with_demographics_df = eob_with_demographics_df.withColumn(
    "category_display_normalized",
    trim(lower(col("category_display")))
)


# In[397]:


eob_with_demographics_df.select("category_display", "category_display_normalized").distinct().show(truncate=False)


# In[398]:


eob_with_demographics_df = eob_with_demographics_df.withColumn(
    "category_display",
    trim(lower(col("category_display")))
)

eob_with_demographics_df = eob_with_demographics_df.withColumn(
    "category_display",
    when(col("category_display") == "line submitted charge amount", "Submitted Charge")
    .when(col("category_display") == "line beneficiary coinsurance amount", "Coinsurance Amount")
    .when(col("category_display") == "line allowed charge amount", "Allowed Charge")
    .when(col("category_display") == "line provider payment amount", "Provider Payment")
    .when(col("category_display") == "line beneficiary part b deductible amount", "Part B Deductible")
    .when(col("category_display") == "line processing indicator Code", "Processing Indicator")
    .otherwise("Unknown")  # Handle unexpected values
)

eob_with_demographics_df.select("category_display").distinct().show(truncate=False)


# In[399]:


eob_with_demographics_df.groupBy("category_display").agg(
    sum("value").alias("total_value"),
    mean("value").alias("avg_value")
).orderBy("total_value", ascending=False).show(truncate=False)


# In[400]:


eob_with_demographics_df.groupBy("category_display", "gender").agg(
    mean("value").alias("avg_value"),
    sum("value").alias("total_value")
).orderBy("category_display", "gender").show(truncate=False)


# In[401]:


eob_with_demographics_df.groupBy("category_display", "age_group").agg(
    mean("value").alias("avg_value"),
    sum("value").alias("total_value")
).orderBy("category_display", "age_group").show(truncate=False)


# In[402]:


eob_with_demographics_df.groupBy("category_display", "age_group").agg(
    mean("value").alias("avg_value"),
    sum("value").alias("total_value")
).orderBy("category_display", "age_group").show(truncate=False)


# In[404]:


eob_with_demographics_df.groupBy("category_display", "marital_status").agg(
    mean("value").alias("avg_value"),
    sum("value").alias("total_value")
).orderBy("category_display", "marital_status").show(truncate=False)


# In[405]:


eob_with_demographics_df.groupBy("category_display", "gender", "age_group", "marital_status").agg(
    mean("value").alias("avg_value"),
    sum("value").alias("total_value")
).orderBy("category_display", "gender", "age_group", "marital_status").show(truncate=False)


# In[ ]:




