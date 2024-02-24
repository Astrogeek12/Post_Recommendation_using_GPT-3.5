# Capture start time
import time
start_time = time.time()
import openai
import pandas as pd
import nltk
#from nltk.tokenize import sent_tokenize, word_tokenize


openai.api_key = "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"

import json
import pandas as pd

def preprocess_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    preprocessed_data = []

    for username, posts in data.items():
        for post in posts:
            text = post.get('text', '')
            preprocessed_data.append({'username': username, 'text': text})

    return preprocessed_data


input_file = 'train_sample.json'
output_csv = 'preprocessed_data.csv'

# Preprocess the JSON data
preprocessed_data = preprocess_json(input_file)

# Convert the preprocessed data into a Pandas DataFrame
df = pd.DataFrame(preprocessed_data)
user_texts = df.groupby('username')['text'].apply(list).to_dict()


# Save the DataFrame to a CSV file
df.to_csv(output_csv, index=False)

print(f"Preprocessing complete. Data saved to '{output_csv}'.")
df = pd.read_csv('trial_output.csv')

############################################################################

topics_per_user = {}

# Loop through each row in the DataFrame
for username, text in zip(df['username'], df['text']):
    # your personalized prompt for ChatGPT-3
    prompt = f"topics that the post come under.{text}"

    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},  # Placeholder for assistant's response
        ],
        temperature=0.7,
        max_tokens=200,
    )

    # Extract and filter potential topics from ChatGPT-3's response
    topics = [topic.strip() for topic in response['choices'][0]['message']['content'].split('\n') if topic.strip()]

    # Store identified topics for each user
    topics_per_user[username] = topics

# Convert topics_per_user dictionary to DataFrame
user_profiling_df = pd.DataFrame(topics_per_user.items(), columns=['username', 'topics'])

# Save the DataFrame to a CSV file
user_profiling_csv = 'user_profiling.csv'
user_profiling_df.to_csv(user_profiling_csv, index=False)

print(f"User profiling complete. Data saved to '{user_profiling_csv}'.")



#####################################################################################

responses = []

# Loop through each post
post_topics_list = []

for _, row in df.iterrows():
    username = row['username']
    post_text = row['text']

    # Assume you already have topics_per_user as a dictionary with usernames as keys and topics as values
    user_topics = topics_per_user.get(username, [])

    # Define your personalized prompt for ChatGPT-3
    prompt = f"topics that the text come under.{text}"

    # Make API call to ChatGPT-3
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt} {post_text}"},
            {"role": "assistant", "content": ""},  # Placeholder for assistant's response
        ],
        temperature=0.7,
        max_tokens=200,
    )
    
    # Append each response to the list
    responses.append(response)

    # Extract and filter potential topics from ChatGPT-3 response
    post_topics = response['choices'][0]['message']['content']
    #print(f"Post topics: {post_topics}")

    

   

    

    post_topics_list.append({'post': post_text, 'post_topics': post_topics})

# Create a new DataFrame for post-level topics
post_topics_df = pd.DataFrame(post_topics_list)

# Save the DataFrame to a CSV file
post_topics_output = "post_topics.csv"
post_topics_df.to_csv(post_topics_output, index=False)

print(f"Post-level topic analysis complete. Data saved to '{post_topics_output}'.")

##########################################################################################3

user_profiling_csv = 'user_profiling.csv'
post_topics_csv = 'post_topics.csv'

# Read the user and interested topics data
user_interests_df = pd.read_csv(user_profiling_csv)

# Read the post and post topics data
post_topics_df = pd.read_csv(post_topics_csv)

# Create a dictionary to map posts to their topics
post_topics_dict = post_topics_df.set_index('post')['post_topics'].to_dict()

# Create a dictionary to store recommendations for each post
post_recommendations_dict = {}

# Iterate through each post
for post, post_topics in post_topics_dict.items():
    # Calculate similarity with each user
    user_similarity_scores = {}

    # Iterate through each user
    for _, user_row in user_interests_df.iterrows():
        username = user_row['username']
        user_interests = user_row['topics']  

        # Calculate similarity as the number of common interests
        similarity_score = len(set(user_interests) & set(post_topics))
        user_similarity_scores[username] = similarity_score

    # Get the top 10 users with the highest similarity scores
    top_users = sorted(user_similarity_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # Store the top 10 user recommendations for the post
    post_recommendations_dict[post] = top_users

# Convert the recommendations dictionary to a DataFrame
post_recommendations_df = pd.DataFrame(post_recommendations_dict.items(), columns=['post', 'top_users'])

# Save the DataFrame to a CSV file
recommendations_csv = 'post_recommendations.csv'
post_recommendations_df.to_csv(recommendations_csv, index=False)

print(f"Post recommendations complete. Data saved to '{recommendations_csv}'.")


##########################################################################################

from sklearn.metrics import jaccard_score


post_recommendations_csv = 'post_recommendations.csv'
trial_output_csv = 'trial_output.csv'

# Read post recommendations and trial output data
post_recommendations_df = pd.read_csv(post_recommendations_csv)
trial_output_df = pd.read_csv(trial_output_csv)

# Initialize lists to store Jaccard similarities and NDCG scores
jaccard_similarities = []
ndcg_scores = []

# Iterate through each post in the trial output
for _, trial_row in trial_output_df.iterrows():
    post_text = trial_row['text']
    actual_user = trial_row['username']

    # Find the corresponding row in the recommendation file
    recommendation_row = post_recommendations_df[post_recommendations_df['post'] == post_text]

    if not recommendation_row.empty:
        # Extract the recommended users from the recommendation file
        recommended_users = [user for user, _ in eval(recommendation_row['top_users'].iloc[0])]

        # Calculate Jaccard similarity
        intersection = len(set([actual_user]) & set(recommended_users))
        union = len(set([actual_user]) | set(recommended_users))
        jaccard_similarity = intersection / union if union > 0 else 0
        jaccard_similarities.append(jaccard_similarity)

        # Calculate NDCG
        relevance_scores = [1 if user == actual_user else 0 for user in recommended_users]
        dcg = sum([(2 ** score - 1) / (i + 1) for i, score in enumerate(relevance_scores)])

        # Sort recommended users by relevance to calculate IDCG
        sorted_recommended_users = sorted(recommended_users, key=lambda x: relevance_scores[recommended_users.index(x)], reverse=True)
        idcg = sum([(2 ** 1 - 1) / (i + 1) for i in range(len(sorted_recommended_users))])

        # Avoid division by zero
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    else:
        print(f"No recommendations found for post: {post_text}")

# average Jaccard similarity and NDCG
average_jaccard_similarity = sum(jaccard_similarities) / len(jaccard_similarities)
average_ndcg = sum(ndcg_scores) / len(ndcg_scores)

print(f"Average Jaccard Similarity: {average_jaccard_similarity}")
print(f"Average NDCG: {average_ndcg}")



end_time = time.time()
for _ in range(1000000):
    _ = _ + 1
# elapsed time
elapsed_time = end_time - start_time

# Print elapsed time
print(f"Elapsed Time: {elapsed_time} seconds")


