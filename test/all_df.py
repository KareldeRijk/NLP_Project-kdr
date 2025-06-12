import os
import pandas as pd
import joblib
import openai
from dotenv import load_dotenv

# ========== 1️⃣ Load environment & API key ==========
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

# ========== 2️⃣ Load saved models and mappings ==========
tfidf = joblib.load("Joblib_files/tfidf_vectorizer.joblib")
label_encoder = joblib.load("Joblib_files/label_encoder.joblib")
rf_clf = joblib.load("Joblib_files/random_forest_model.joblib")
unique_categories_dict = joblib.load("Clustering_model/unique_categories_dict.pkl")

# ========== 3️⃣ Load and combine datasets ==========

# Define file paths
file_paths = [
    "datasets/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    "datasets/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv",
    "datasets/1429_1.csv"
]

dataframes = []

for file in file_paths:
    df_temp = pd.read_csv(file)

    # If imageURLs column is missing, add it with NaN
    if 'imageURLs' not in df_temp.columns:
        df_temp['imageURLs'] = pd.NA
    
    dataframes.append(df_temp)

# Concatenate all dataframes
df = pd.concat(dataframes, ignore_index=True)

# Drop rows where 'reviews.text' is missing (since we need text for sentiment)
df = df.dropna(subset=['reviews.text'])

# ========== 4️⃣ Predict sentiment ==========
X_tfidf = tfidf.transform(df['reviews.text'])
predictions = rf_clf.predict(X_tfidf)
predicted_labels = label_encoder.inverse_transform(predictions)
df['predicted_sentiment'] = predicted_labels

# ========== 5️⃣ Map categories ==========
df['category_cluster'] = df['categories'].map(unique_categories_dict).fillna("Unknown")

# ========== 6️⃣ Get top N products by positive sentiment per category ==========
def get_top_products_by_category(df, sentiment_label="positive", top_n=3):
    positive_df = df[df['predicted_sentiment'] == sentiment_label]
    product_counts = (
        positive_df.groupby(['category_cluster', 'name'])
        .size()
        .reset_index(name='positive_count')
    )
    top_products = (
        product_counts
        .sort_values(['category_cluster', 'positive_count'], ascending=[True, False])
        .groupby('category_cluster')
        .head(top_n)
    )
    return top_products

top_products = get_top_products_by_category(df)

# ========== 7️⃣ Generate summaries using OpenAI ==========
def generate_summary(product, reviews):
    formatted_reviews = "\n\n".join(reviews)
    prompt = f"""
    You are an expert product reviewer.
    Summarize the following customer reviews for the product, please write a maximum 5 lines summary, should be very tech-oriented '{product}':
    
    {formatted_reviews}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a content manager summarizer of product reviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

summaries = []
for idx, row in top_products.iterrows():
    product_name = row['name']
    product_reviews = df[
        (df['name'] == product_name) & 
        (df['predicted_sentiment'] == 'positive')
    ]['reviews.text'].head(20).tolist()
    summary = generate_summary(product_name, product_reviews)
    summaries.append(summary)

top_products['summary'] = summaries

# ========== 8️⃣ Extract first or second image URL ==========
def extract_image_urls(url_string):
    placeholder = "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"
    
    if pd.isna(url_string) or not url_string.strip():
        return placeholder
    
    urls = [url.strip() for url in url_string.split(",") if url.strip()]
    
    if len(urls) > 0 and urls[0]:
        return urls[0]
    if len(urls) > 1 and urls[1]:
        return urls[1]
    
    return placeholder

# ========== 9️⃣ Merge additional info and apply image extraction ==========
df_final = pd.merge(
    top_products,
    df.drop_duplicates(subset=['name'])[['name', 'category_cluster', 'imageURLs', 'reviews.rating']],
    on=['name', 'category_cluster'],
    how='left'
)

df_final['image_url'] = df_final['imageURLs'].apply(extract_image_urls)

# ========== 🔟 Compute avg positive rating ==========
def compute_avg_rating(product_name):
    positive_reviews = df[
        (df['name'] == product_name) & 
        (df['predicted_sentiment'] == 'positive')
    ]['reviews.rating']
    return round(positive_reviews.mean(), 2) if not positive_reviews.empty else None

df_final['avg_positive_rating'] = df_final['name'].apply(compute_avg_rating)

# Drop columns no longer needed
df_final = df_final.drop(columns=['imageURLs', 'reviews.rating'])

# ========== 1️⃣1️⃣ Save final output ==========
# Ensure output directory exists
os.makedirs("output", exist_ok=True)

df_final.to_csv("output/test.csv", index=False)
print("✅ Full pipeline completed. Output saved to output/test.csv")
