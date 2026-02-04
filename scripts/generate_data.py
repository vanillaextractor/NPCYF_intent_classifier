import pandas as pd
import random

# Define intents
INTENT_GENERAL = 0
INTENT_SQL = 1

crops = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Bajra", "Jowar", "Soybean"]
states = ["Punjab", "Haryana", "UP", "Uttar Pradesh", "Bihar", "West Bengal", "Maharashtra", "Madhya Pradesh", "Karnataka", "Tamil Nadu"]
seasons = ["Kharif", "Rabi", "Zaid"]
years = [str(y) for y in range(2010, 2025)]
weather_params = ["rainfall", "temperature", "humidity", "precipitation", "wind speed"]
agri_metrics = ["yield", "production", "sowing area", "harvested area"]

sql_templates = [
    "What was the {metric} of {crop} in {state} in {year}?",
    "Show me the {weather} data for {state} in {year}",
    "How much {crop} was produced in {state} during {season} season?",
    "Give me the average {weather} in {state} for {year}",
    "Which state had the highest {crop} {metric} in {year}?",
    "Compare the {metric} of {crop} between {state} and {state}",
    "List all climatic data for {state} where {weather} > 100mm",
    "Find the {crop} yield for {year}",
    "Did it rain in {state} in {year}?",
    "Max temperature in {state} last year",
    "Rice production data for {state}",
    "Show yield trend for {crop}",
]

general_templates = [
    "What is feature engineering?",
    "Explain Random Forest algorithm",
    "How does a decision tree work?",
    "Tell me about Indian agriculture",
    "Challenges faced by Indian farmers",
    "What is the difference between supervised and unsupervised learning?",
    "Define precision and recall",
    "How to improve model accuracy?",
    "What is deep learning?",
    "Explain the concept of overfitting",
    "Best practices for sustainable farming",
    "Introduction to machine learning",
    "What is Green Revolution?",
    "Types of soil in India",
    "What is crop rotation?",
    "Explain organic farming methods",
    "What is a neural network?",
    "How to handle missing data in pandas?",
    "What is dimensionality reduction?",
    "Explain PCA",
    "How to normalize data?",
    "Data cleaning techniques in Python",
    "Handling imbalanced data for classification",
    "Best pesticides for Cotton",
    "Soil requirements for Wheat",
    "Common diseases in Rice",
    "Harvesting methods for Sugarcane",
    "Organic fertilizers for Maize",
    "Explain cross-validation on data",
    "Data visualization with Matplotlib",
    "What is data augmentation?",
    "History of Cotton farming in India"
]

def generate_sql_query():
    template = random.choice(sql_templates)
    return template.format(
        metric=random.choice(agri_metrics),
        crop=random.choice(crops),
        state=random.choice(states),
        year=random.choice(years),
        season=random.choice(seasons),
        weather=random.choice(weather_params)
    )

def generate_dataset(num_samples=500):
    data = []
    
    # Generate balanced dataset
    for _ in range(num_samples // 2):
        data.append({"text": generate_sql_query(), "label": INTENT_SQL})
        data.append({"text": random.choice(general_templates), "label": INTENT_GENERAL}) # Sampling with replacement for general
    
    # Add some variations to general queries to avoid exact duplicates dominating too much if num_samples is large,
    # but for 500 samples and ~20 templates, duplicates are inevitable. 
    # Let's add specific hardcoded variety to general queries list to make it robust.
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_dataset(600)
    output_path = "data/dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path} with {len(df)} samples.")
    print(df.head())
