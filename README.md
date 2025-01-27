# Weather Data Analysis Project



## Project Overview
This project focuses on weather data analysis with the objective of predicting precipitation (rainfall). The dataset sourced from NASA POWER includes various weather parameters such as temperature, humidity, wind speed, and precipitation. The primary goal is to preprocess the dataset to make it suitable for predictive modeling, identifying patterns and trends critical for accurate precipitation prediction.

---

## Dataset Overview

The dataset consists of **1094 rows** and **11 columns**, providing weather-related conditions. Below is a list of the columns:

| Column Name                             | Description                           |
|-----------------------------------------|---------------------------------------|
| `YEAR`                                  | Year of observation                   |
| `MO`                                    | Month of observation                  |
| `DY`                                    | Day of observation                    |
| `Temperature at 2 Meters (°C)`          | Observed temperature at 2 meters      |
| `Dew/Frost Point at 2 Meters (°C)`      | Dew or frost point temperature        |
| `Temperature at 2 Meters Maximum (°C)` | Maximum temperature observed          |
| `Temperature at 2 Meters Minimum (°C)` | Minimum temperature observed          |
| `Specific Humidity at 2 Meters (g/kg)` | Specific humidity at 2 meters         |
| `Relative Humidity at 2 Meters (%)`    | Relative humidity at 2 meters         |
| `Precipitation Corrected (mm/day)`     | Corrected precipitation measurement   |
| `Wind Speed at 10 Meters (m/s)`        | Wind speed at 10 meters               |

---

## Steps Performed

### 1. Data Loading and Exploration
The dataset was loaded into the environment using `pandas`:
```python
import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv')

# Inspecting the dataset
df.info()
df.describe()
df.head()
df.tail()
```
Key insights include:
- Total entries: **1094**
- No missing values were found.
- Data types include 8 numerical (`float64`) and 3 integer (`int64`) columns.

### 2. Data Cleaning
- **Missing Data:** Verified no missing values using `df.isna().sum()`.
- **Duplicate Records:** No duplicates found, confirmed by checking the dataset's shape before and after removing duplicates.
- **Date Column Creation:** Combined `YEAR`, `MO`, and `DY` into a single `Date` column for time-based analysis.
- **Outlier Detection and Handling:**
  - Skewness was checked for all numerical columns.
  - Used Z-score or IQR methods based on skewness to detect outliers.
  - Outliers in `Wind Speed at 10 Meters` were removed.
  - Outliers in `Precipitation Corrected` were retained as they represent valid, rare events.

### 3. Data Transformation
- **Scaling:**
  - Applied standardization (Z-score normalization) to numerical features to handle varying feature scales.
  - Standardization ensures consistency across features and is robust against outliers.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
scaled_data = scaler.fit_transform(df[numerical_columns])
```

---

## Key Libraries Used
- **Pandas**: For data loading, exploration, and cleaning.
- **NumPy**: For numerical computations.
- **SciPy**: For statistical analysis.
- **Scikit-learn**: For data preprocessing and normalization.

---

## Next Steps
- Perform feature engineering to extract additional insights.
- Implement machine learning models to predict precipitation.
- Evaluate model performance and refine based on results.

---

## Contact
For any questions or contributions, feel free to contact:
- **Muhammad Qasim**: mq08460@gmail.com

