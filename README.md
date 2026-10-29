# Araling Panlipunan Student Data: KMeans Clustering Analysis

## Project Overview
This repository contains a data analysis project exploring the correlations between various student demographics, language skills, and their academic performance (grades) in Araling Panlipunan (Social Studies). 

Data was gathered from 151 students. To maintain privacy, all student identities were obfuscated. The analysis utilizes the **KMeans** clustering algorithm from the Python `scikit-learn` library to identify patterns within the student data, and the results are visualized using scatterplots generated with `matplotlib.pyplot`.

## Repository Contents

*   **`AP.csv`**: The obfuscated and pre-processed dataset containing the student survey responses. Categorical text data (such as ethnicity and qualitative grades) has been mapped to numerical values for clustering.
*   **`AralPan.py`**: The main Python script that handles data loading, model training, clustering, and data visualization. 
*   **`KmeansAnalysisVideo.mp4`**: A video walkthrough and demonstration of the code execution, data processing steps, and resulting scatterplot visualizations.

## Methodology

### 1. Data Pre-processing
The raw survey data originally contained text-based responses for fields such as Ethnicity, Attitude, Written Filipino Skills, Spoken Filipino Skills, and Grades. Before running the clustering model, this data was cleaned and converted into numerical formats to allow for mathematical distance calculations required by KMeans. Duplicates were also removed.

### 2. Features Selected
The K-Means clustering model uses the following features to form clusters:
*   `ethnicity`
*   `attitude` (Attitude towards other Filipino cultures)
*   `writtenfil` (Ability in Written Filipino)
*   `spokenfil` (Ability in Spoken Filipino)
*   `grade` (Overall Grade in Araling Panlipunan)

### 3. Clustering
The `scikit-learn` library's KMeans algorithm was configured to group the student data into **4 distinct clusters**. Test data points (e.g., simulating a Bisaya or Ilonggo student with specific attitudes and language skills) were also injected to see how the model categorizes hypothetical student profiles based on the generated clusters.

### 4. Visualization
The `matplotlib.pyplot` library is used to generate scatterplots to visualize the clusters across different dimensions. Key relationships analyzed include:
*   Attitude towards other cultures vs. Grade
*   Skills in Written Filipino vs. Grade
*   Skills in Spoken Filipino vs. Grade

## Requirements
To run the `AralPan.py` script locally, you will need Python installed along with the following libraries:
*   `pandas` (for reading and manipulating the CSV data)
*   `scikit-learn` (for the KMeans clustering algorithm)
*   `matplotlib` (for generating the scatterplot visualizations)

You can install these dependencies using pip:
```bash
pip install pandas scikit-learn matplotlib
```