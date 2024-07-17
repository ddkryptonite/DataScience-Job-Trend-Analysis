# DataScience Job Trend Analysis

## Overview
This repository contains a data analysis project focused on trends in data science job postings across various locations, industries, and skill requirements. The analysis provides insights into the types of roles available, required skills, popular locations, and more.

## Contents
1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Data Cleaning](#data-cleaning)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Predictive Analysis](#predictive-analysis)
6. [Project Structure](#project-structure)
7. [Technologies Used](#technologies-used)
8. [Setup Instructions](#setup-instructions)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction
The project aims to analyze data science job postings to understand trends such as popular locations, skills in demand, and industry preferences. This analysis can provide valuable insights for job seekers, recruiters, and researchers in the field of data science.

## Data Overview
The dataset used contains information on various job postings including job title, company, required skills, minimum experience, location, industry type, and more. It includes data from multiple sources to provide a comprehensive view of the data science job market.

## Data Cleaning
Initial data cleaning steps involved handling missing values, renaming columns for clarity, and standardizing location names. This ensured the dataset was ready for analysis without compromising data integrity.

## Exploratory Data Analysis
- **Top Locations**: Analyzed the distribution of job postings across different cities.
- **Minimum Experience**: Examined the minimum experience required for different roles.
- **Role Categories**: Explored the distribution of job roles across various categories.
- **Top Skills**: Identified the most in-demand skills based on job postings.
- **Top Companies**: Investigated the companies with the highest number of job postings.

## Predictive Analysis'
Utilized Decision Trees algorithm to predict Role Category (Job Category) based on various features with a 70% accuracy and generated a confusion matrix to show accuracy of prediction.
Utilized K Nearest Neighbor (KNN) algorithm to predict job locations based on role categories, functional areas, and industry types. Achieved an accuracy of approximately 42%, indicating potential insights into predicting job locations based on job attributes.

## Project Structure
- `Data_Science_Jobs.csv`: Dataset containing the raw data used for analysis.
- `AnalysisDataScienceJobs.ipynb`: Jupyter Notebook containing the Python code for data analysis.
- `README.md`: This file, providing an overview of the project.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

## Setup Instructions
To run the analysis notebook locally:
1. Clone this repository.
2. Install the required dependencies (`pip install -r requirements.txt`).
3. Launch Jupyter Notebook and open `AnalysisDataScienceJobs.ipynb`.

## Contributing
Contributions to enhance the analysis or add new features are welcome. Please fork the repository, create a new branch, and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
