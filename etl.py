#%% md
# The input data used for this project is FHIR patient data from the Oh Canada dataset found here - https://synthea.mitre.org/downloads
# 
# The input files are created for each patient, and consists of all relevant entries wrt that patient. The goal of this ETL is to read the json files, transform to create multiple dataframes, and load the same. 
# 
# We assume that the files are present in a Data Lake and would be stored in a Data Warehouse after the ETL. 
#%%
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, expr

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FHIR Data Pipeline") \
    .getOrCreate()

#%%

# File path
file_path = sys.argv[1]

#Output path
output_path = sys.argv[2]

# Read the JSON file and repartition
data = spark.read.json(file_path, multiLine=True).repartition(numPartitions=16)

# Explode the entry array
entries = data.select(explode(col("entry")).alias("entry"))

# Cache entries because it is used to derive all subsequent dataframes
entries.cache()

#%% md
# ### Patient
#%%
# Filter patient resources
patients = entries.filter(col("entry.resource.resourceType") == "Patient") \
                  .select(col("entry.resource.*"))

# Extract the "official" name
official_name = expr("""
    filter(name, x -> x.use = 'official')[0]
""")

# Extract geolocation fields
geolocation = expr("""
    filter(address[0].extension, x -> x.url = 'http://hl7.org/fhir/StructureDefinition/geolocation')[0].extension
""")

# Extract all languages from communication as a list
communication_languages = expr("""
    transform(communication, x -> x.language.text)
""")

# Extract identifier types
identifier_types = expr("""
    transform(identifier, x -> x.type.coding[0].display)
""")

# Create the enhanced patient DataFrame
patient_df = patients.select(
    col("id").alias("patient_id"),
    identifier_types.alias("identifier_types"),
    col("name").alias("name"),
    col("gender").alias("gender"),
    col("birthDate").alias("birth_date"),
    col("address").getItem(0).getField("city").alias("city"),
    col("address").getItem(0).getField("state").alias("state"),
    col("address").getItem(0).getField("country").alias("country"),
    col("address").getItem(0).getField("postalCode").alias("postal_code"),
    geolocation.getItem(0).getField("valueDecimal").alias("latitude"),
    geolocation.getItem(1).getField("valueDecimal").alias("longitude"),
    col("telecom").getItem(0).getField("value").alias("phone"),
    col("maritalStatus.text").alias("marital_status"),
    col("extension").getItem(0).getField("valueString").alias("mothers_maiden_name"),
    col("extension").getItem(1).getField("valueAddress").getField("city").alias("birthplace_city"),
    col("extension").getItem(1).getField("valueAddress").getField("state").alias("birthplace_state"),
    col("extension").getItem(1).getField("valueAddress").getField("country").alias("birthplace_country"),
    col("extension").getItem(2).getField("valueDecimal").alias("disability_adjusted_life_years"),
    col("extension").getItem(3).getField("valueDecimal").alias("quality_adjusted_life_years"),
    col("multipleBirthBoolean").alias("multiple_birth"),
    communication_languages.alias("languages")
)

# Show the resulting DataFrame
patient_df.show(truncate=False)

#%%
# Save the DataFrame to Parquet format in the "patients" folder
patient_df.write.mode("overwrite").parquet(f"{output_path}/patient")

print("Patient DataFrame has been successfully saved in Parquet format in the 'patient' folder.")
#%% md
# ### Encounter
#%%
from pyspark.sql.functions import from_json
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

# Define the schema for the 'type' field
type_schema = ArrayType(
    StructType([
        StructField("coding", ArrayType(
            StructType([
                StructField("system", StringType(), True),
                StructField("code", StringType(), True),
                StructField("display", StringType(), True)
            ])
        ), True),
        StructField("text", StringType(), True)
    ])
)


# Filter encounter resources
encounters = entries.filter(col("entry.resource.resourceType") == "Encounter") \
                    .select(col("entry.resource.*"))

# Parse the 'type' field from JSON string to structured format
encounters = encounters.withColumn("type_parsed", from_json(col("type"), type_schema))

# Explode the participant array to create one row per participant
encounters_with_participants = encounters.select(
    col("id").alias("encounter_id"),
    col("status").alias("status"),
    col("class.code").alias("class_code"),
    col("type_parsed").getItem(0).getField("text").alias("type_text"),  
    col("subject.reference").alias("patient_reference"),
    col("period.start").alias("start_time"),
    col("period.end").alias("end_time"),
    col("serviceProvider.reference").alias("service_provider_id"),
    col("serviceProvider.display").alias("service_provider_display"),
    explode(col("participant")).alias("participant")
)

# Extract participant details
encounter_with_participant_df = encounters_with_participants.select(
    col("encounter_id"),
    col("status"),
    col("class_code"),
    col("type_text"),
    col("start_time"),
    col("end_time"),
    col("patient_reference"),
    col("service_provider_id"),
    col("service_provider_display"),
    col("participant.individual.display").alias("participant_individual_display"),
    col("participant.individual.reference").alias("participant_individual_reference"),
    col("participant.period.start").alias("participant_period_start"),
    col("participant.period.end").alias("participant_period_end"),
    col("participant.type").getItem(0).getField("coding").getItem(0).getField("code").alias("participant_type_code"),
    col("participant.type").getItem(0).getField("coding").getItem(0).getField("display").alias("participant_type_display")
)

# Show the resulting DataFrame
encounter_with_participant_df.show(truncate=False)

#%%
# Save the DataFrame to Parquet
encounter_with_participant_df.write.mode("overwrite").parquet(f"{output_path}/encounter")

print("Participant details have been successfully extracted and saved in the 'encounter' folder.")

#%% md
# ### Condition
#%%
# Filter for Condition resources
conditions = entries.filter(col("entry.resource.resourceType") == "Condition") \
                    .select(col("entry.resource.*"))

# Extract relevant fields from Condition
condition_df = conditions.select(
    col("id").alias("condition_id"),
    col("clinicalStatus.coding").getItem(0).getField("code").alias("clinical_status"),
    col("verificationStatus.coding").getItem(0).getField("code").alias("verification_status"),
    col("code.text").alias("condition_code_display"),
    col("subject.reference").alias("patient_reference"),
    col("encounter.reference").alias("encounter_reference"),
    col("onsetDateTime").alias("onset_datetime"),
    col("abatementDateTime").alias("abatement_datetime"),
    col("recordedDate").alias("recorded_date")
)

# Show the resulting DataFrame
condition_df.show(truncate=False)
#%%
# Save the DataFrame to Parquet
condition_df.write.mode("overwrite").parquet(f"{output_path}/condition")

print("Condition DataFrame has been successfully saved in Parquet format in the 'condition' folder.")
#%% md
# ### Medication Request
#%%
# Filter for MedicationRequest resources
medication_requests = entries.filter(col("entry.resource.resourceType") == "MedicationRequest") \
                             .select(col("entry.resource.*"))

# Explode the dosageInstruction array to handle multiple dosage instructions
medication_requests_exploded = medication_requests.withColumn("dosageInstruction", explode(col("dosageInstruction")))

# Extract relevant fields from MedicationRequest
medication_request_df = medication_requests_exploded.select(
    col("id").alias("medication_request_id"),
    col("status").alias("status"),
    col("intent").alias("intent"),
    col("medicationCodeableConcept.coding").getItem(0).getField("display").alias("medication_display"),
    col("subject.reference").alias("patient_reference"),
    col("requester.reference").alias("requester_reference"),
    col("requester.display").alias("requester_display"),
    col("encounter.reference").alias("encounter_reference"),
    col("authoredOn").alias("authored_on"),
    col("dosageInstruction.text").alias("dosage_text"),
    # col("dosageInstruction.timing.repeat.frequency").alias("dosage_frequency"),
    # col("dosageInstruction.timing.repeat.period").alias("dosage_period"),
    # col("dosageInstruction.timing.repeat.periodUnit").alias("dosage_period_unit")
)

# Show the resulting DataFrame
medication_request_df.show(truncate=False)
#%%
# Save the DataFrame to Parquet
medication_request_df.write.mode("overwrite").parquet(f"{output_path}/medication_request")

print("MedicationRequest DataFrame with multiple dosage instructions has been successfully saved in the 'medication_request' folder.")
#%% md
# ### Claim
#%%
from pyspark.sql.types import DoubleType

# Filter for Claim resources
claims = entries.filter(col("entry.resource.resourceType") == "Claim") \
                .select(col("entry.resource.*"))

# Define the schema for the 'type' field
type_schema = ArrayType(
    StructType([
        StructField("coding", ArrayType(
            StructType([
                StructField("system", StringType(), True),
                StructField("code", StringType(), True)
            ])
        ), True)
    ])
)

# Define the schema for the 'total' field
total_schema = StructType([
    StructField("value", DoubleType(), True),      
    StructField("currency", StringType(), True)  
])


claims = (claims
          .withColumn("type_parsed", from_json(col("type"), type_schema))
          .withColumn("total_parsed", from_json(col("total"), total_schema))
          .withColumn("supportingInfo", explode(col("supportingInfo")))
          .withColumn("insurance", explode(col("insurance")))
          .withColumn("claim_item", explode(col("item")))
          .withColumn("diagnosis", explode(col("diagnosis")))
          )  

# Extract relevant fields from Claim
claim_df = claims.select(
    col("id").alias("claim_id"),
    col("status").alias("status"),
    col("type_parsed").getItem(0).getField("coding").getItem(0).getField("code").alias("type_code"),
    col("use").alias("use"),
    col("patient.reference").alias("patient_reference"),
    col("patient.display").alias("patient_display"),
    col("created").alias("created_date"),
    col("billablePeriod.start").alias("billable_period_start"),
    col("billablePeriod.end").alias("billable_period_end"),
    col("provider.reference").alias("provider_reference"),
    col("priority.coding").getItem(0).getField("code").alias("priority"),
    col("supportingInfo.category.coding").getItem(0).getField("code").alias("supporting_info_code"),
    col("supportingInfo.valueReference.reference").alias("supporting_info_value_reference"),
    col("insurance.coverage.display").alias("insurance_coverage_display"),
    col("insurance.focal").alias("insurance_focal"),
    col("total_parsed").getField("value").alias("total_amount"), 
    col("total_parsed").getField("currency").alias("currency"),
    col("diagnosis.diagnosisReference.reference").alias("diagnosis_reference"), 
    col("claim_item.productOrService.coding").getItem(0).getField("code").alias("item_code"),
    col("claim_item.productOrService.coding").getItem(0).getField("display").alias("item_description"),
    col("claim_item.category.coding").getItem(0).getField("display").alias("item_category"),
    col("claim_item.net.value").alias("item_net_value"),
    col("claim_item.net.currency").alias("item_net_currency"),
    col("claim_item.encounter").getItem(0).getField("reference").alias("encounter_reference"),
    col("claim_item.locationCodeableConcept.coding").getItem(0).getField("display").alias("location_description"),
    col("claim_item.servicedPeriod.start").alias("service_period_start"),
    col("claim_item.servicedPeriod.end").alias("service_period_end"),
    col("claim_item.adjudication").alias("adjudication")
)

claim_df.show(truncate=False)

#%%
# Save the DataFrame to Parquet
claim_df.write.mode("overwrite").parquet(f"{output_path}/claim")

print("Claims DataFrame has been successfully saved in the 'claim' folder.")
#%% md
# ### Explanation of Benefit
#%%
# Filter for ExplanationOfBenefit resources
eobs = entries.filter(col("entry.resource.resourceType") == "ExplanationOfBenefit") \
              .select(col("entry.resource.*"))

eobs = (eobs.withColumn("type_parsed", from_json(col("type"), type_schema))
        .withColumn("total_parsed", from_json(col("total"), total_schema))
        .withColumn("contained", explode(col("contained")))
        .withColumn("careTeam", explode(col("careTeam")))
        .withColumn("insurance", explode(col("insurance")))
        .withColumn("item", explode(col("item")))
        )
# Extract relevant fields from ExplanationOfBenefit
eob_df = eobs.select(
    col("id").alias("eob_id"),
    col("identifier").getItem(0).getField("value").alias("identifier_claim_id"),  
    col("identifier").getItem(1).getField("value").alias("identifier_claim_group"),  
    col("status").alias("status"),
    col("type_parsed.coding").getItem(0).getField("code").alias("code"),
    col("use").alias("use"),
    col("patient.reference").alias("patient_reference"),
    col("billablePeriod.start").alias("billable_period_start"),
    col("billablePeriod.end").alias("billable_period_end"),
    col("insurer.display").alias("insurer_display"),
    col("provider.reference").alias("provider_reference"),
    col("referral.reference").alias("referral_reference"),
    col("claim.reference").alias("claim_reference"),
    col("outcome").alias("outcome"),
    col("careTeam.provider.reference").alias("care_team_provider_reference"),
    col("careTeam.role.coding").getItem(0).getField("display").alias("care_team_role"), 
    col("insurance.coverage.reference").alias("insurance_coverage_reference"),
    col("insurance.coverage.display").alias("insurance_coverage_display"),
    col("insurance.focal").alias("insurance_focal"),
    col("item.category.coding").getItem(0).getField("display").alias("item_category_display"),
    col("item.productOrService.coding").getItem(0).getField("code").alias("item_service_code"),
    col("item.productOrService.coding").getItem(0).getField("display").alias("item_service_description"),
    col("item.net.value").alias("item_net_value"),
    col("item.net.currency").alias("item_net_currency"),
    col("item.locationCodeableConcept.coding").getItem(0).getField("display").alias("location_description"),
    col("item.servicedPeriod.start").alias("service_period_start"),
    col("item.servicedPeriod.end").alias("service_period_end"),
    col("item.adjudication").alias("adjudication"),
    col("total_parsed.value").alias("total_amount"),
    col("total_parsed.currency").alias("total_currency"),
    col("payment.amount.value").alias("payment_amount"),
    col("payment.amount.currency").alias("payment_currency"),
    col("created").alias("created_date")  
)

# Show the resulting DataFrame
eob_df.show(truncate=False)
#%%
# Save the DataFrame to Parquet
eob_df.write.mode("overwrite").parquet(f"{output_path}/explanation_of_benefit")

print("Explanation of Benefit DataFrame has been successfully saved in the 'explanation_of_benefit' folder.")
#%% md
# ### Care Plan
#%%
# Filter for CarePlan resources
careplans = entries.filter(col("entry.resource.resourceType") == "CarePlan") \
                   .select(col("entry.resource.*"))

# Define schema for the 'category' field
category_schema = StructType([
        StructField("coding", ArrayType(
            StructType([
                StructField("system", StringType(), True),
                StructField("code", StringType(), True),
                StructField("display", StringType(), True),
            ])
        ), True),
        StructField("text", StringType(), True)
    ])


careplans = (careplans
             .withColumn("careTeam", explode(col("careTeam")))
             .withColumn("activity", explode(col("activity")))
             )

# Extract relevant fields from CarePlan
careplan_df = careplans.select(
    col("id").alias("careplan_id"),
    col("status").alias("status"),
    col("intent").alias("intent"),
    col("category").getItem(0).alias("category"),  
    col("subject.reference").alias("patient_reference"),
    col("encounter.reference").alias("encounter_reference"),
    col("period.start").alias("period_start"),
    col("period.end").alias("period_end"),
    col("created").alias("created_date"),
    col("careTeam.reference").alias("care_team_reference"), 
    col("activity.detail.code.coding").getItem(0).getField("display").alias("activity_display"),
    col("activity.detail.code.coding").getItem(0).getField("code").alias("activity_code"),
    col("activity.detail.status").alias("activity_status"),
    col("activity.detail.location.display").alias("activity_location"),
    col("addresses.reference").alias("addresses")
)

# Since category is an array of string, separately process the column
careplan_df = careplan_df.withColumn("category_parsed", from_json(col("category"), category_schema)).withColumn("category_text", col("category_parsed.text"))

# Dropping the redundant columns
careplan_df = careplan_df.select(
    [column for column in careplan_df.columns if column not in ["category", "category_parsed"]]
)

# Show the resulting DataFrame
careplan_df.show(truncate=False)

#%%
# Save the DataFrame to Parquet
careplan_df.write.mode("overwrite").parquet(f"{output_path}/careplan")

print(" CarePlan DataFrame has been successfully saved in the 'careplan' folder.")
#%% md
# ### Care Team
#%%
# Filter for CareTeam resources
care_team_resources = entries.filter(col("entry.resource.resourceType") == "CareTeam") \
                             .select(col("entry.resource.*"))

# Explode the participants array if it exists
care_team_resources = (care_team_resources.withColumn("participant", explode(col("participant")))
                       .withColumn("managingOrganization", explode(col("managingOrganization"))))

# Extract relevant fields
care_team_df = care_team_resources.select(
    col("id").alias("care_team_id"),
    col("status").alias("status"),
    col("subject.reference").alias("subject_reference"),
    col("encounter.reference").alias("encounter_reference"),
    col("period.start").alias("period_start"),
    col("period.end").alias("period_end"),
    col("managingOrganization.display").alias("managing_organization"),
    col("participant.role.text").getItem(0).alias("participant_role"),
    col("participant.member.reference").alias("participant_reference"),
    col("participant.member.display").alias("participant_display"),
)

# Show the resulting DataFrame
care_team_df.show(truncate=False)

#%%
# Save the DataFrame to Parquet
care_team_df.write.mode("overwrite").parquet(f"{output_path}/careteam")

print(" CareTeam DataFrame has been successfully saved in the 'careteam' folder.")
#%% md
# ### Procedure
#%%
# Filter for Procedure resources
procedures = entries.filter(col("entry.resource.resourceType") == "Procedure") \
                    .select(col("entry.resource.*"))

procedures = procedures.withColumn("reasonReference", explode(col("reasonReference")))

# Extract relevant fields
procedure_df = procedures.select(
    col("id").alias("procedure_id"),
    col("status").alias("status"),
    col("code.coding").getItem(0).getField("display").alias("procedure_code_display"),
    col("code.coding").getItem(0).getField("code").alias("procedure_code"),
    col("subject.reference").alias("subject_reference"),
    col("performedPeriod.start").alias("performed_period_start"),
    col("performedPeriod.end").alias("performed_period_end"),
    col("encounter.reference").alias("encounter_reference"),
    col("reasonReference.reference").alias("reason_reference"),
    col("reasonReference.display").alias("reason_reference_display")
)

# Show the resulting DataFrame
procedure_df.show(truncate=False)
#%%
# Save the DataFrame to Parquet
procedure_df.write.mode("overwrite").parquet(f"{output_path}/procedure")

print("Procedure DataFrame has been successfully saved in the 'procedure' folder.")
#%% md
# ### Diagnostic Report
#%%
# Filter for DiagnosticReport resources
diagnostic_reports = entries.filter(col("entry.resource.resourceType") == "DiagnosticReport") \
                            .select(col("entry.resource.*"))

# Explode results array if present
diagnostic_reports = diagnostic_reports.withColumn("result", explode(col("result")))

# Extract relevant fields
diagnostic_report_df = diagnostic_reports.select(
    col("id").alias("diagnostic_report_id"),
    col("status").alias("status"),
    col("category").getItem(0).alias("category"),
    col("code.coding").getItem(0).getField("display").alias("code"),
    col("code.coding").getItem(0).getField("code").alias("code_system"),
    col("subject.reference").alias("subject_reference"),
    col("encounter.reference").alias("encounter_reference"),
    col("effectiveDateTime").alias("effective_date_time"),
    col("issued").alias("issued_date"),
    col("result.display").alias("result_display")
)

diagnostic_report_df = diagnostic_report_df.withColumn("category_parsed", from_json(col("category"), category_schema)).withColumn("category_display", col("category_parsed.coding").getItem(0).getField("display"))

diagnostic_report_df = diagnostic_report_df.select(
    [column for column in diagnostic_report_df.columns if column not in ["category", "category_parsed"]]
)
# Show the resulting DataFrame
diagnostic_report_df.show(truncate=False)
#%%
# Save the Dataframe to Parquet
diagnostic_report_df.write.mode("overwrite").parquet(f"{output_path}/diagnostic_report")

print("Diagnostic report DataFrame has been successfully saved in the 'diagnostic_report' folder.")
#%% md
# ### Immunization
#%%
# Filter for Immunization resources
immunizations = entries.filter(col("entry.resource.resourceType") == "Immunization") \
                       .select(col("entry.resource.*"))

# Extract relevant fields
immunization_df = immunizations.select(
    col("id").alias("immunization_id"),
    col("status").alias("status"),
    col("vaccineCode.coding").getItem(0).getField("display").alias("vaccine_display"),
    col("vaccineCode.coding").getItem(0).getField("code").alias("vaccine_code"),
    col("patient.reference").alias("patient_reference"),
    col("encounter.reference").alias("encounter_reference"),
    col("primarySource").alias("primary_source"),
    col("occurrenceDateTime").alias("occurrence_date_time"),
)

# Show the resulting DataFrame
immunization_df.show(truncate=False)
#%%
# Save the Dataframe to Parquet
immunization_df.write.mode("overwrite").parquet(f"{output_path}/immunization")

print("Immunization DataFrame has been successfully saved in the 'immunization' folder.")
#%% md
# ### Observation
#%%
# Filter for Observation resources
observations = entries.filter(col("entry.resource.resourceType") == "Observation") \
                      .select(col("entry.resource.*"))


# Extract relevant fields
observation_df = observations.select(
    col("id").alias("observation_id"),
    col("status").alias("status"),
    col("category").getItem(0).alias("category"),
    col("code.coding").getItem(0).getField("display").alias("observation_code_display"),
    col("code.coding").getItem(0).getField("code").alias("observation_code"),
    col("subject.reference").alias("subject_reference"),
    col("encounter.reference").alias("encounter_reference"),
    col("effectiveDateTime").alias("effective_date_time"),
    col("issued").alias("issued_date"),
    col("valueQuantity.value").alias("value_quantity_value"),
    col("valueQuantity.unit").alias("value_quantity_unit")
    
)

observation_df = observation_df.withColumn("category_parsed", from_json(col("category"), category_schema)).withColumn("category_display", col("category_parsed.coding").getItem(0).getField("display"))

observation_df = observation_df.select(
    [column for column in observation_df.columns if column not in ["category", "category_parsed"]]
)
# Show the resulting DataFrame
observation_df.show(truncate=False)

#%%
# Save the Dataframe to Parquet
observation_df.write.mode("overwrite").parquet(f"{output_path}/observation")

print("Observation DataFrame has been successfully saved in the 'observation' folder.")
#%%
