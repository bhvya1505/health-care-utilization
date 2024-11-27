import sys
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, max, min, year
from HealthcareReportGeneration import generate_report


def main(input_file, output_file):
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Healthcare Data Analysis").config("spark.sql.shuffle.partitions",
                                                                            "50").config("spark.executor.memory",
                                                                                         "4g").config(
        "spark.driver.memory", "4g").getOrCreate()

    # Load and Cache the Data
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    df.cache()

    # Analysis Functions with Immediate Conversion to Pandas
    def patient_demographics_analysis(df):
        return df.groupBy("GENDER", "RACE", "ETHNICITY").count().toPandas()

    def age_distribution(df):
        df = df.withColumn("Age", 2023 - year("BIRTHDATE"))
        return df.groupBy("Age").count().orderBy("Age").toPandas()

    def marital_status_distribution(df):
        return df.groupBy("MARITAL").count().toPandas()

    def gender_based_encounters(df):
        return df.groupBy("GENDER", "ENCOUNTER_DESCRIPTION").count().toPandas()

    def average_encounter_costs(df):
        return df.groupBy("ENCOUNTER_DESCRIPTION").agg(
            avg(col("ENCOUNTER_BASE_COST")).alias("Average_Base_Cost"),
            avg(col("ENCOUNTER_TOTAL_CLAIM_COST")).alias("Average_Claim_Cost"),
            avg(col("ENCOUNTER_PAYER_COVERAGE")).alias("Average_Coverage")
        ).toPandas()

    def cost_by_procedure(df):
        return df.groupBy("PROCEDURE_DESCRIPTION").agg(
            avg(col("PROCEDURE_BASE_COST")).alias("Average_Procedure_Cost")
        ).toPandas()

    def insurance_coverage_by_procedure(df):
        return df.groupBy("PROCEDURE_DESCRIPTION").agg(
            avg(col("ENCOUNTER_PAYER_COVERAGE")).alias("Average_Coverage")
        ).toPandas()

    def most_frequent_procedures(df):
        return df.groupBy("PROCEDURE_DESCRIPTION").count().orderBy(col("count").desc()).limit(10).toPandas()

    def frequent_encounter_reasons(df):
        return df.groupBy("ENCOUNTER_REASON_DESCRIPTION").count().orderBy(col("count").desc()).limit(10).toPandas()

    def procedures_per_encounter(df):
        return df.groupBy("ENCOUNTER_DESCRIPTION", "PROCEDURE_DESCRIPTION").count().orderBy(
            "ENCOUNTER_DESCRIPTION").toPandas()

    def age_based_encounter_analysis(df):
        df = df.withColumn("Age", 2023 - year("BIRTHDATE"))
        return df.groupBy("Age", "ENCOUNTER_DESCRIPTION").count().orderBy("Age").toPandas()

    def age_and_cost_relationship(df):
        df = df.withColumn("Age", 2023 - year("BIRTHDATE"))
        return df.groupBy("Age").agg(
            avg(col("ENCOUNTER_TOTAL_CLAIM_COST")).alias("Average_Claim_Cost")
        ).toPandas()

    def coverage_vs_cost(df):
        return df.groupBy("ENCOUNTER_DESCRIPTION").agg(
            avg(col("ENCOUNTER_TOTAL_CLAIM_COST")).alias("Average_Claim_Cost"),
            avg(col("ENCOUNTER_PAYER_COVERAGE")).alias("Average_Coverage")
        ).toPandas()

    def encounter_cost_variation(df):
        return df.groupBy("ENCOUNTER_DESCRIPTION").agg(
            min("ENCOUNTER_BASE_COST").alias("Min_Base_Cost"),
            max("ENCOUNTER_BASE_COST").alias("Max_Base_Cost")
        ).toPandas()

    def marital_status_encounters(df):
        return df.groupBy("MARITAL", "ENCOUNTER_DESCRIPTION").count().toPandas()

    def ethnicity_cost_disparity(df):
        return df.groupBy("ETHNICITY").agg(
            avg("ENCOUNTER_TOTAL_CLAIM_COST").alias("Average_Cost")
        ).toPandas()

    def encounter_cost_coverage_relation(df):
        return df.groupBy("ENCOUNTER_DESCRIPTION").agg(
            avg("ENCOUNTER_PAYER_COVERAGE").alias("Average_Coverage"),
            avg("ENCOUNTER_TOTAL_CLAIM_COST").alias("Average_Total_Cost")
        ).toPandas()

    def top_expensive_encounters(df):
        return df.orderBy(col("ENCOUNTER_TOTAL_CLAIM_COST").desc()).limit(10).toPandas()

    # Collect Results as Pandas DataFrames
    results = {
        "Patient Demographics Analysis": patient_demographics_analysis(df),
        "Age Distribution": age_distribution(df),
        "Marital Status Distribution": marital_status_distribution(df),
        "Gender-Based Encounters": gender_based_encounters(df),
        "Average Encounter Costs": average_encounter_costs(df),
        "Cost by Procedure": cost_by_procedure(df),
        "Insurance Coverage by Procedure": insurance_coverage_by_procedure(df),
        "Most Frequent Procedures": most_frequent_procedures(df),
        "Frequent Encounter Reasons": frequent_encounter_reasons(df),
        "Procedures per Encounter": procedures_per_encounter(df),
        "Age-Based Encounter Analysis": age_based_encounter_analysis(df),
        "Age and Cost Relationship": age_and_cost_relationship(df),
        "Coverage vs. Cost": coverage_vs_cost(df),
        "Encounter Cost Variation": encounter_cost_variation(df),
        "Marital Status and Encounter Types": marital_status_encounters(df),
        "Ethnicity Cost Disparities": ethnicity_cost_disparity(df),
        "Encounter Cost-Coverage Relationship": encounter_cost_coverage_relation(df),
        "Top Expensive Encounters": top_expensive_encounters(df)
    }

    # Serialize results using pickle
    with open("analysis_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Call the report generation program
    generate_report("analysis_results.pkl", output_file)
    spark.stop()


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
