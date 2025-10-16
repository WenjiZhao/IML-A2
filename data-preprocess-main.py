import DataPreprocess as dp
import pandas as pd

def main():
    df = pd.read_csv('Dataset/alzheimers_prediction_dataset.csv')

    maps = {
        "smoking_stack.pop();status" : {"Never": 0, "Former": 1, "Current": 2},
        "physical_activity_level" : {"Low": 2, "Medium": 1, "High": 0},
        "alcohol_consumption" : {"Never": 0, "Occasionally": 1, "Regularly": 2},
        "depression_level" : {"Low": 0, "Medium": 1, "High": 2},
        "sleep_quality" : {"Poor": 2, "Average": 1, "Good": 0},
        "dietary_habits" : {"Unhealthy": 2, "Average": 1, "Healthy": 0},
        "air_pollution_exposure" : {"Low": 2, "Medium": 1, "High": 0},
        "social_engagement_level" : {"Low": 2, "Medium": 1, "High": 0},
        "income_level" : {"Low": 2, "Medium": 1, "High": 0},
        "stress_levels" : {"Low": 2, "Medium": 1, "High": 0}
    }

    column_nominal = {"gender", "diabetes", "hypertension",
                      "cholesterol_level", "family_history_of_alzheimer’s",
                      "employment_status", "marital_status",
                      "genetic_risk_factor_(apoe-ε4_allele)", "urban_vs_rural_living"}

    column_scale = {"age", "education_level", "bmi", "cognitive_test_score", "physical_activity_level", "smoking_status", "alcohol_consumption",
                    "depression_level", "sleep_quality", "dietary_habits",
                    "air_pollution_exposure", "social_engagement_level",
                    "income_level", "stress_levels"}

    pre = dp.DataPreprocess(df)

    pre.clean_column_names()

    for col, mapping in maps.items():
        pre.encode_column(col, mapping)

    for col in column_nominal:
        pre.encode_nominal(col)

    pre.class_process("alzheimer’s_diagnosis")

    pre.min_max_scaler(column_scale)


    clean_df = pre.get_data()

    pre.save_csv("after_processing.csv")

if __name__ == "__main__":
    main()