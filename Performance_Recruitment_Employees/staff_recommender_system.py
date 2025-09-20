
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Read the dataset
data = pd.read_csv('Performance_Recruitment_Employees/fau_clinic_recommender_system.csv')

# check columns data
print(data.columns)
#  concatenate columns, and we need spaces between cells for tfidf to work
def concatenate_features(x):
    return ' '.join([
        str(x['teams']) if pd.notna(x['teams']) else '',
        str(x['previous_experience']) if pd.notna(x['previous_experience']) else '',
        str(x['hobbies']) if pd.notna(x['hobbies']) else '',
        str(x['sports']) if pd.notna(x['sports']) else ''
    ])


# new concatenated column
data['joined_columns'] = data.apply(concatenate_features, axis=1)

# print for debugging
for index, value in data['joined_columns'].items():
    print(f"Row {index}: {value}")

# fit and transform
tfidf_vector = TfidfVectorizer(stop_words='english')
# combine TF and IDF to assign a weight to each term
tfidf_matrix = tfidf_vector.fit_transform(data['joined_columns'])

cosine_func_kernel = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['id']).drop_duplicates()

# recommender method
def recommender(id_x, cosine_func_similarity):

    # store indices
    idx = indices[id_x]
    score_similarity = list(enumerate(cosine_func_similarity[idx]))

    # sorting
    score_similarity = sorted(score_similarity, key=lambda x: x[1], reverse=True)

    # the three most similar employees excluding self
    score_similarity = score_similarity[1:4]

    # return the info about them
    indices_nurses = [i[0] for i in score_similarity]
    recommended_nurses = data.iloc[indices_nurses].copy()
    recommended_nurses['similarity_score'] = [score[1] for score in score_similarity]

    return recommended_nurses[['id', 'similarity_score']]

print(recommender('emp_050', cosine_func_kernel))

#  ================= Visualization =======================

import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap for visualizing the tokens
def check_emp_050(similarity_matrix, data_ids):
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='coolwarm', annot=False, xticklabels=data_ids, yticklabels=data_ids)
    plt.title("Similarity Heatmap")
    plt.xlabel("Employee ID")
    plt.ylabel("Employee ID")
    plt.tight_layout()
    plt.show()

similarity_m = pd.DataFrame(
    cosine_func_kernel,
    index=data['id'],
    columns=data['id']
)

# Plotting
check_emp_050(similarity_m, data['id'])

