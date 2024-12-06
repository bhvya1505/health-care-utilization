import pickle
import sys


def generate_report(input_file, output_file):
    # Load serialized analysis results
    with open(input_file, "rb") as f:
        results = pickle.load(f)

    # Detailed descriptions for each section
    analysis_descriptions = {
        "PATIENT DEMOGRAPHICS ANALYSIS": (
            "This analysis breaks down patient demographics by gender, race, and ethnicity. "
            "The insights provide a foundational understanding of the patient population's diversity. \n\n"
            "In healthcare, demographic information is crucial for tailoring programs, identifying at-risk groups, "
            "and ensuring equity in healthcare delivery. By understanding the demographic makeup, healthcare organizations "
            "can develop targeted care initiatives and allocate resources efficiently, addressing disparities and improving "
            "patient outcomes."
        ),

        "AGE DISTRIBUTION": (
            "The age distribution analysis provides insights into the distribution of patients by age group. "
            "This information is vital for designing age-specific healthcare services and understanding which age groups may have "
            "higher healthcare needs. \n\nIt can help in anticipating future demands in healthcare services, tailoring preventive measures, "
            "and managing resources. Identifying age trends in healthcare utilization can also help reduce age-related health risks."
        ),

        "MARITAL STATUS DISTRIBUTION": (
            "This analysis examines the distribution of marital status among patients, offering insights into social support structures. "
            "Marital status often influences healthcare outcomes, as married individuals may have additional family support that impacts recovery times and mental health. \n\n"
            "Understanding this distribution allows healthcare providers to consider social support factors in treatment plans, aiming to improve overall patient well-being."
        ),

        "GENDER-BASED ENCOUNTERS": (
            "This analysis categorizes healthcare encounters by gender and type, helping identify gender-specific health trends and resource needs. "
            "Gender-based encounter data can be used to tailor health programs to address specific gender-related health concerns, ensuring equitable healthcare delivery. \n\n"
            "It addresses the need for gender sensitivity in healthcare, which is crucial for improving patient satisfaction and outcomes."
        ),

        "AVERAGE ENCOUNTER COSTS": (
            "This analysis reveals average costs across various encounter types, including base costs, claim costs, and insurance coverage. \n\n"
            "By examining these costs, healthcare administrators can evaluate financial efficiency, forecast budgets, and identify high-cost encounter types that may require cost-reduction strategies. "
            "This analysis is essential for optimizing resource allocation and controlling healthcare expenses."
        ),

        "COST BY PROCEDURE": (
            "This analysis displays the average cost associated with each procedure. By identifying procedures with high associated costs, healthcare providers can prioritize cost-effective practices and seek ways to reduce expenses for costly procedures. \n\n"
            "This data-driven insight supports cost management initiatives and helps achieve financial sustainability in healthcare operations."
        ),

        "INSURANCE COVERAGE BY PROCEDURE": (
            "This analysis shows average insurance coverage for each procedure, aiding in understanding which procedures are more likely to be covered by insurance. \n\n"
            "This information helps healthcare providers identify gaps in insurance coverage and advocate for policy changes to ensure comprehensive patient coverage."
        ),

        "MOST FREQUENT PROCEDURES": (
            "This analysis highlights the most common procedures conducted in the healthcare setting. Frequent procedure data helps healthcare administrators allocate resources effectively, streamline processes, and improve efficiency in service delivery. \n\n"
            "By focusing on commonly performed procedures, providers can optimize workflows and enhance patient care quality for high-demand services."
        ),

        "FREQUENT ENCOUNTER REASONS": (
            "Identifies common reasons for patient encounters. Understanding encounter reasons allows healthcare providers to focus on preventative care and patient education initiatives. \n\n"
            "Frequent reasons for encounters can indicate public health trends and guide resource allocation for patient education on prevalent health issues, ultimately reducing avoidable visits."
        ),

        "PROCEDURES PER ENCOUNTER": (
            "This analysis displays the frequency of procedures conducted within each encounter type. By understanding procedure patterns, healthcare providers can better manage scheduling, reduce patient wait times, and optimize care coordination. \n\n"
            "This analysis helps ensure the efficient use of medical resources and staff time."
        ),

        "AGE-BASED ENCOUNTER ANALYSIS": (
            "This analysis categorizes encounter types by age, helping identify age-specific healthcare needs. Age-related data is valuable for tailoring health services, ensuring age-appropriate care, and allocating resources effectively for different age groups. \n\n"
            "Addressing age-specific needs can enhance patient satisfaction and improve health outcomes."
        ),

        "AGE AND COST RELATIONSHIP": (
            "Explores cost variations by age group, showing how encounter costs differ with patient age. This data supports financial planning for age-specific healthcare services and helps identify cost drivers among age groups. \n\n"
            "Understanding the age-cost relationship is essential for budgeting and resource allocation."
        ),

        "COVERAGE VS. COST": (
            "This analysis compares average insurance coverage against actual encounter costs by type, highlighting gaps in coverage. \n\n"
            "This information is beneficial for both patients and providers, as it supports advocacy for improved insurance plans that cover the true cost of healthcare services. It addresses the challenge of underinsurance and promotes equitable access to necessary care."
        ),

        "ENCOUNTER COST VARIATION": (
            "Shows minimum and maximum costs for each encounter type, providing insights into cost variances. \n\n"
            "By examining cost variability, healthcare providers can develop strategies to standardize costs, making healthcare services more predictable and affordable for patients."
        ),

        "MARITAL STATUS AND ENCOUNTER TYPES": (
            "Analyzes encounter types based on marital status, helping providers understand how social support structures impact healthcare utilization. \n\n"
            "This information can guide care planning, especially for single or elderly patients who may require additional support, contributing to a holistic approach to patient care."
        ),

        "ETHNICITY COST DISPARITIES": (
            "Identifies disparities in healthcare costs across different ethnic groups. This data-driven insight is crucial for ensuring equitable healthcare and addressing potential systemic inequalities. \n\n"
            "By highlighting cost disparities, healthcare providers can work toward equitable resource allocation and treatment."
        ),

        "ENCOUNTER COST-COVERAGE RELATIONSHIP": (
            "Analyzes the relationship between encounter costs and insurance coverage, providing insight into how effectively insurance covers healthcare expenses. \n\n"
            "This information can guide policy changes and support advocacy efforts to improve patient coverage, especially for high-cost encounters."
        ),

        "TOP EXPENSIVE ENCOUNTERS": (
            "Lists the most expensive encounters, identifying cases with significant financial impact. \n\n"
            "This information can help healthcare administrators focus on cost-containment strategies and optimize care for high-cost cases, ensuring sustainable financial management."
        )
    }

    with open(output_file, 'w') as f:
        # Title Page
        f.write("HEALTHCARE DATA ANALYSIS PROJECT REPORT\n")
        f.write("=" * 100 + "\n")
        f.write("Comprehensive Analysis of Patient Encounters and Procedures\n\n")

        # Table of Contents
        f.write("TABLE OF CONTENTS\n")
        f.write("=" * 25 + "\n")
        for i, title in enumerate(results.keys(), start=1):
            f.write(f"{i}. {title}\n")
        f.write("\n" + "=" * 100 + "\n\n")

        # Main Analysis Sections with dynamic widths capped at 250 characters
        max_column_width = 250
        for i, (title, data) in enumerate(results.items(), start=1):
            f.write(f"{i}. {title.upper()}\n")
            f.write("=" * 100 + "\n")
            description = analysis_descriptions.get(title.upper(), "Detailed analysis information not available.")
            f.write(f"{description}\n\n")

            # Calculate column widths dynamically, capped at 250 characters
            columns = data.columns.tolist()
            column_widths = []
            for col in columns:
                max_width = max(len(str(val)) for val in data[col].astype(str).tolist() + [col])
                column_widths.append(min(max_width, max_column_width))

            # Header Border
            f.write("+" + "+".join("-" * (width + 2) for width in column_widths) + "+\n")

            # Column Headers
            header = "| " + " | ".join(col.upper().ljust(width) for col, width in zip(columns, column_widths)) + " |\n"
            f.write(header)

            # Header Bottom Border
            f.write("+" + "+".join("-" * (width + 2) for width in column_widths) + "+\n")

            # Row Data with truncation for values longer than 250 characters
            for _, row in data.iterrows():
                row_line = "| " + " | ".join(
                    str(val)[:width].ljust(width) for val, width in zip(row, column_widths)) + " |\n"
                f.write(row_line)

            # Table Bottom Border and Section Separator
            f.write("+" + "+".join("-" * (width + 2) for width in column_widths) + "+\n\n")
            f.write("=" * 100 + "\n\n")

        # Conclusion Section
        f.write("CONCLUSION AND SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write("This report presents a comprehensive analysis of healthcare encounters and procedures.\n")
        f.write(
            "Key insights include demographic patterns, cost variations, and insurance coverage analysis. Further investigation and detailed analysis can drive actionable improvements in patient care.")


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    generate_report
