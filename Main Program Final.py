# Import all necessary libraries for this program
from pathlib import Path
import kagglehub
import pandas as pd
import os
import pickle
from pathlib import Path
import joblib
import re
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai
from datetime import datetime


# Load the OPENAI API key from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_TEST_KEY_KdR")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_TEST_KEY_KdR environment variable.")

client = openai.OpenAI(api_key=api_key)

# Load the preprocessed reviews data from our Joblib folder
df_combined_path = Path.cwd() / "Joblib_files" / "amazon_reviews_with_embeddings.csv"
if not df_combined_path.exists():
    raise FileNotFoundError(f"Missing dataframe file: {df_combined_path}")
df_combined = pd.read_csv(df_combined_path)

# Import the random forest classifier model
rf_classifier_model_path = Path.cwd() / "Joblib_files" / "classifier_random_forest_model.joblib"
if not rf_classifier_model_path.exists():
    raise FileNotFoundError(f"Missing classifier model file: {rf_classifier_model_path}")
rf_classifier_model = joblib.load(rf_classifier_model_path)
# Import the clustering-classification model
clustering_model_path = Path.cwd() / "Joblib_files" / "Clustering_logistic_regression_model_cv.joblib"
if not clustering_model_path.exists():
    raise FileNotFoundError(f"Missing clustering model file: {clustering_model_path}")
clustering_model = joblib.load(clustering_model_path)

# Define a Class below to score the top 3 products for a given meta-category. The Class contains functions that use the rf_classifier_model to create sentiment analysis 
# on the embeddings of the reviews
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Enhanced SimpleProductScorer class that handles low-data categories
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleProductScorer:
    def __init__(self, df, model_path='random_forest_model.joblib', min_reviews_threshold=5):
        self.df = df.copy()
        self.product_scores = None
        self.model = self._load_model(model_path)
        self.min_reviews_threshold = min_reviews_threshold
        self.low_data_categories = ['Pet Products', 'Kitchen Storage']  # Categories with limited data
    
    def _load_model(self, model_path):
        """Load the trained model if available"""
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except:
            print("No model found - using basic sentiment scoring")
            return None
    
    def _get_sentiment_score(self, row):
        """Get sentiment score from model or fallback to rating-based score"""
        if self.model and 'embeddings' in row and row['embeddings'] is not None:
            try:
                # Use model prediction if available
                prediction = self.model.predict([row['embeddings']])[0]
                # Convert to -1 to 1 scale (adjust based on your model output)
                if prediction in [0, 1, 2]:  # Class labels
                    return {0: -1, 1: 0, 2: 1}[prediction]
                else:
                    return max(-1, min(1, prediction))  # Clamp to [-1, 1]
            except:
                pass
        
        # Fallback: convert rating to sentiment (-1 to 1)
        rating = row.get('reviews.rating', 3)
        return (rating - 3) / 2  # 1->-1, 3->0, 5->1
    
    def _is_low_data_category(self, category):
        """Check if category has low data and should use simplified scoring"""
        return category in self.low_data_categories
    
    def _calculate_simplified_score(self, product_data, category):
        """Calculate simplified score for low-data categories based primarily on ratings"""
        review_count = len(product_data)
        avg_rating = product_data['reviews.rating'].mean()
        
        # For low-data categories, use simpler scoring
        if pd.isna(avg_rating):
            avg_rating = 3.0  # Default neutral rating
        
        # Base score on rating (0-1 scale)
        rating_score = (avg_rating - 1) / 4  # Convert 1-5 to 0-1
        
        # Apply small bonus for having multiple reviews
        review_bonus = min(review_count / 10, 0.1)  # Max 0.1 bonus for 10+ reviews
        
        # Simple composite score for low-data categories
        composite_score = rating_score + review_bonus
        
        # Get other metrics with defaults
        recommend_ratio = product_data['reviews.doRecommend'].mean() if 'reviews.doRecommend' in product_data.columns else 0.5
        helpfulness = product_data['reviews.numHelpful'].fillna(0).mean()
        
        # Simple sentiment based on rating
        avg_sentiment = (avg_rating - 3) / 2  # Convert 1-5 rating to -1 to 1 sentiment
        
        return {
            'composite_score': composite_score,
            'avg_rating': avg_rating,
            'avg_sentiment': avg_sentiment,
            'recommend_ratio': recommend_ratio,
            'helpfulness': helpfulness,
            'scoring_method': 'simplified_rating_based'
        }
    
    def _calculate_full_score(self, product_data):
        """Calculate full score for categories with sufficient data"""
        review_count = len(product_data)
        avg_rating = product_data['reviews.rating'].mean()
        
        # Calculate sentiment scores for each review
        sentiment_scores = []
        for _, row in product_data.iterrows():
            sentiment_scores.append(self._get_sentiment_score(row))
        
        avg_sentiment = np.mean(sentiment_scores)
        
        # Helpfulness score (higher is better)
        helpfulness = product_data['reviews.numHelpful'].fillna(0).mean()
        
        # Recommendation ratio
        recommend_ratio = product_data['reviews.doRecommend'].mean() if 'reviews.doRecommend' in product_data.columns else 0.5
        
        # Full composite score (0-1 scale)
        # Weight: 40% rating, 30% sentiment, 20% recommendations, 10% helpfulness
        rating_score = (avg_rating - 1) / 4  # Convert 1-5 to 0-1
        sentiment_score = (avg_sentiment + 1) / 2  # Convert -1,1 to 0-1
        helpfulness_score = min(helpfulness / 10, 1.0)  # Cap at 1.0
        
        composite_score = (
            0.4 * rating_score + 
            0.3 * sentiment_score + 
            0.2 * recommend_ratio + 
            0.1 * helpfulness_score
        )
        
        # Apply review count penalty for products with very few reviews
        if review_count < 5:
            composite_score *= 0.8
        elif review_count < 10:
            composite_score *= 0.9
        
        return {
            'composite_score': composite_score,
            'avg_rating': avg_rating,
            'avg_sentiment': avg_sentiment,
            'recommend_ratio': recommend_ratio,
            'helpfulness': helpfulness,
            'scoring_method': 'full_model_based'
        }
    
    def calculate_scores(self):
        """Calculate product scores using appropriate method based on data availability"""
        print("Calculating product scores...")
        
        # Add sentiment scores to reviews (for full scoring method)
        if 'sentiment_score' not in self.df.columns:
            self.df['sentiment_score'] = self.df.apply(self._get_sentiment_score, axis=1)
        
        # Group by product and calculate aggregate scores
        product_scores = []
        
        # Get category statistics to determine scoring method
        category_stats = self.df.groupby('meta_category').agg({
            'asins': 'nunique',
            'reviews.rating': 'count'
        }).rename(columns={'asins': 'unique_products', 'reviews.rating': 'total_reviews'})
        
        print("Category statistics:")
        print(category_stats)
        
        for asin in self.df['asins'].dropna().unique():
            product_data = self.df[self.df['asins'] == asin]
    
            if product_data.empty:
                continue  # Skip empty group

            product_name = product_data['name'].iloc[0]
            if pd.isna(product_name):
                print(f"Skipping ASIN {asin} due to missing product name.")
                continue  # Skip products with no name

            category = product_data['meta_category'].iloc[0]
            review_count = len(product_data)
            
            # Determine scoring method based on category and data availability
            if self._is_low_data_category(category):
                print(f"Using simplified scoring for {category}: {product_name[:50]}...")
                scores = self._calculate_simplified_score(product_data, category)
            else:
                scores = self._calculate_full_score(product_data)
            
            product_scores.append({
                'asin': asin,
                'product_name': product_name,
                'category': category,
                'review_count': review_count,
                'avg_rating': scores['avg_rating'],
                'avg_sentiment': scores['avg_sentiment'],
                'recommend_ratio': scores['recommend_ratio'],
                'helpfulness': scores['helpfulness'],
                'composite_score': scores['composite_score'],
                'scoring_method': scores['scoring_method']
            })
        
        self.product_scores = pd.DataFrame(product_scores)
        self.product_scores = self.product_scores.sort_values('composite_score', ascending=False)
        
        print(f"Scored {len(self.product_scores)} products")
        
        # Print scoring method breakdown
        scoring_methods = self.product_scores['scoring_method'].value_counts()
        print("\nScoring methods used:")
        for method, count in scoring_methods.items():
            print(f"  {method}: {count} products")
        
        return self.product_scores
    
    def get_top_products(self, category=None, top_k=3):
        """Get top K products overall or by category"""
        if self.product_scores is None:
            raise ValueError("Run calculate_scores() first")
        
        if category:
            filtered_scores = self.product_scores[self.product_scores['category'] == category]
            
            # For low-data categories, ensure we return available products even if less than top_k
            if len(filtered_scores) < top_k:
                print(f"Warning: Only {len(filtered_scores)} products available in {category} category")
                return filtered_scores
        else:
            filtered_scores = self.product_scores
        
        return filtered_scores.head(top_k)
    
    def get_top_products_by_category(self, category, top_k=3):
        """Alias for get_top_products for backward compatibility"""
        return self.get_top_products(category, top_k)
    
    def get_category_summary(self, category):
        """Get summary statistics for a specific category"""
        if self.product_scores is None:
            return "No scores calculated yet"
        
        category_data = self.product_scores[self.product_scores['category'] == category]
        
        if category_data.empty:
            return f"No products found in category: {category}"
        
        return {
            'category': category,
            'total_products': len(category_data),
            'avg_composite_score': category_data['composite_score'].mean(),
            'avg_rating': category_data['avg_rating'].mean(),
            'avg_sentiment': category_data['avg_sentiment'].mean(),
            'total_reviews': category_data['review_count'].sum(),
            'scoring_method': category_data['scoring_method'].iloc[0],
            'is_low_data_category': self._is_low_data_category(category)
        }
    
    def get_summary(self):
        """Get simple summary of results"""
        if self.product_scores is None:
            return "No scores calculated yet"
        
        return {
            'total_products': len(self.product_scores),
            'avg_score': self.product_scores['composite_score'].mean(),
            'avg_rating': self.product_scores['avg_rating'].mean(),
            'avg_sentiment': self.product_scores['avg_sentiment'].mean(),
            'categories': self.product_scores['category'].value_counts().to_dict(),
            'scoring_methods': self.product_scores['scoring_method'].value_counts().to_dict()
        }
    
    def get_scoring_summary(self):
        """Get detailed summary (for backward compatibility)"""
        if self.product_scores is None:
            return "No scores calculated yet"
        
        return {
            'total_products': len(self.product_scores),
            'avg_composite_score': self.product_scores['composite_score'].mean(),
            'avg_sentiment_score': self.product_scores['avg_sentiment'].mean(),
            'categories': self.product_scores['category'].value_counts().to_dict(),
            'score_distribution': self.product_scores['composite_score'].describe().to_dict(),
            'scoring_methods': self.product_scores['scoring_method'].value_counts().to_dict()
        }
    
    def get_sentiment_analysis_summary(self):
        """Get sentiment analysis summary (for backward compatibility)"""
        if self.df is None:
            return "No data available"
        
        total_reviews = len(self.df)
        valid_sentiments = self.df['sentiment_score'].notna().sum() if 'sentiment_score' in self.df.columns else 0
        
        return {
            'total_reviews': total_reviews,
            'valid_predictions': valid_sentiments,
            'coverage': valid_sentiments / total_reviews if total_reviews > 0 else 0,
            'sentiment_distribution': {
                'positive': (self.df['sentiment_score'] > 0.1).sum() if 'sentiment_score' in self.df.columns else 0,
                'negative': (self.df['sentiment_score'] < -0.1).sum() if 'sentiment_score' in self.df.columns else 0,
                'neutral': ((self.df['sentiment_score'] >= -0.1) & 
                           (self.df['sentiment_score'] <= 0.1)).sum() if 'sentiment_score' in self.df.columns else 0
            }
        }

    def print_top_products_by_category(self, top_k=3):
        """Print top products for each category"""
        print(f"\nTop {top_k} products by category:")
        print("=" * 50)
        
        for category in self.product_scores['category'].unique():
            top_products = self.get_top_products(category, top_k)
            category_summary = self.get_category_summary(category)
            
            print(f"\n{category.upper()} ({category_summary['scoring_method']}):")
            print(f"Total products: {category_summary['total_products']}, Total reviews: {category_summary['total_reviews']}")
            
            for i, (_, product) in enumerate(top_products.iterrows(), 1):
                print(f"{i}. {product['product_name'][:60]}")
                print(f"   Score: {product['composite_score']:.3f} | "
                      f"Rating: {product['avg_rating']:.1f} | "
                      f"Sentiment: {product['avg_sentiment']:.2f} | "
                      f"Reviews: {product['review_count']}")


# Enhanced generate_summary function that handles low-data categories
# Enhanced generate_summary function that handles low-data categories and NaN values
# The function takes as input the dataframe of top products generated by the SimpleProductScorer class, and
# the combined reviews dataframe (df_combined, for extraction of the embeddings of the reviews). As output,
# the function generates a summary of the top products reviews, including a comparison of the top products, and saves it to a CSV file.
def generate_summary(df, df_combined, output_filename=None, min_reviews_for_full_summary=5):
    """
    Generate product summaries and a comparison for top products, with enhanced handling for low-data categories.
    
    Args:
        df (pd.DataFrame): Top products dataframe
        df_combined (pd.DataFrame): Combined reviews dataframe
        output_filename (str, optional): Custom filename for CSV output
        min_reviews_for_full_summary (int): Minimum reviews needed for full AI summary
    
    Returns:
        str: Path to the generated CSV file
    """
    summaries_data = []
    comparison_data = []
    low_data_categories = ['Pet Products', 'Kitchen Storage']
    
    # Check if the input dataframe is empty
    if df.empty:
        print("Warning: No products to summarize")
        return None

    for idx, row in df.iterrows():
        asin = row['asin']
        product_name = row['product_name']
        avg_score = row.get('avg_rating', 0.0)  # Use .get() with default
        review_count = row.get('review_count', 0)
        avg_sentiment = row.get('avg_sentiment', 0.0)
        recommend_ratio = row.get('recommend_ratio', 0.0)
        helpfulness = row.get('helpfulness', 0.0)
        category = row.get('category', 'Unknown')
        composite_score = row.get('composite_score', 0.0)
        scoring_method = row.get('scoring_method', 'unknown')
        
        # Handle NaN values explicitly
        if pd.isna(avg_score):
            avg_score = 0.0
        if pd.isna(review_count):
            review_count = 0
        if pd.isna(avg_sentiment):
            avg_sentiment = 0.0
        if pd.isna(recommend_ratio):
            recommend_ratio = 0.0
        if pd.isna(helpfulness):
            helpfulness = 0.0
        if pd.isna(composite_score):
            composite_score = 0.0
        
        # Get reviews for this specific product from df_combined
        product_reviews = df_combined[df_combined['asins'] == asin]['cleaned_text'].tolist()
        
        # Get image URL for this product
        product_images = df_combined[df_combined['asins'] == asin]['imageURLs'].tolist()
        image_url = product_images[0] if product_images else "No image available"
        
        # Determine if this is a low-data category or product
        is_low_data = category in low_data_categories or len(product_reviews) < min_reviews_for_full_summary
        
        # Handle case where no reviews are found
        if not product_reviews:
            summary_text = f"Limited review data available for {product_name}. Product rated {avg_score:.1f}/5.0 based on available ratings."
            review_count_used = 0
            review_note = "No reviews available for detailed analysis"
            sentiment_desc = "N/A"
        else:
            # Convert sentiment score to descriptive text
            if avg_sentiment >= 0.6:
                sentiment_desc = "very positive"
            elif avg_sentiment >= 0.2:
                sentiment_desc = "positive"
            elif avg_sentiment >= -0.2:
                sentiment_desc = "neutral"
            elif avg_sentiment >= -0.6:
                sentiment_desc = "negative"
            else:
                sentiment_desc = "very negative"
            
            # For low-data categories or products with few reviews, use simplified summary
            if is_low_data:
                # Create a simple summary based on available data
                if len(product_reviews) > 0:
                    # Use first few reviews for basic summary
                    sample_reviews = product_reviews[:min(3, len(product_reviews))]
                    formatted_reviews = "\n".join(sample_reviews)
                    review_count_used = len(sample_reviews)
                    review_note = f"Limited data: {len(product_reviews)} reviews available, showing {review_count_used}"
                    
                    # Simplified prompt for low-data scenarios
                    prompt = f"""
                    You are a product analyst. Create a brief summary for this product with limited review data:
                    Product: '{product_name}'
                    Category: {category}
                    Rating: {avg_score:.1f}/5.0 ({review_count} reviews)
                    Sentiment: {sentiment_desc}
                    
                    Available customer feedback:
                    {formatted_reviews}
                    
                    Provide a concise 2-3 line summary focusing on key product features and customer satisfaction.
                    Note that this summary is based on limited review data.
                    """
                else:
                    # No reviews available, create basic summary
                    summary_text = f"{product_name} is a {category.lower()} product with an average rating of {avg_score:.1f}/5.0. Limited review data available for detailed analysis."
                    review_count_used = 0
                    review_note = "Summary based on rating data only"
            else:
                # Full summary for products with sufficient data
                if len(product_reviews) > 10:
                    formatted_reviews = "\n\n".join(product_reviews[:10])
                    review_note = f"Comprehensive analysis: showing 10 out of {len(product_reviews)} reviews"
                    review_count_used = 10
                else:
                    formatted_reviews = "\n\n".join(product_reviews)
                    review_note = f"Complete analysis: all {len(product_reviews)} reviews included"
                    review_count_used = len(product_reviews)

                prompt = f"""
                You are an expert product reviewer.
                Summarize the following customer reviews for the product: '{product_name}'
                Write a maximum 5 lines summary, very tech-oriented.

                Product Details:
                - Product name: {product_name}
                - Category: {category}
                - Average rating: {avg_score:.2f}/5.0 ({review_count} reviews)
                - Average sentiment: {sentiment_desc} ({avg_sentiment:.2f})
                - Recommendation ratio: {recommend_ratio:.1%}
                - Review helpfulness: {helpfulness:.2f}
                
                Customer Reviews:
                {formatted_reviews}
                
                Focus on technical aspects, build quality, performance, and value proposition.
                """

            # Generate AI summary only if we have a meaningful prompt
            if 'prompt' in locals():
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a technical product analyst. Be concise and focus on practical insights."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200 if is_low_data else 300,
                        temperature=0.3
                    )
                    summary_text = response.choices[0].message.content.strip()
                    
                    # Add note about limited data if applicable
                    if is_low_data and len(product_reviews) > 0:
                        summary_text += f"\n\nNote: Summary based on limited review data ({len(product_reviews)} reviews)."
                        
                except Exception as e:
                    print(f"Error generating AI summary for {product_name}: {e}")
                    summary_text = f"Error generating summary for {product_name}: {e}"
                    if is_low_data:
                        summary_text = f"{product_name} - {category} product rated {avg_score:.1f}/5.0. Limited review data available for detailed analysis."

        # Store individual product summary data
        summaries_data.append({
            'asin': asin,
            'product_name': product_name,
            'category': category,
            'image_url': image_url,
            'avg_rating': avg_score,
            'review_count': review_count,
            'avg_sentiment': avg_sentiment,
            'sentiment_description': sentiment_desc,
            'recommend_ratio': recommend_ratio,
            'helpfulness': helpfulness,
            'composite_score': composite_score,
            'scoring_method': scoring_method,
            'reviews_used_for_summary': review_count_used,
            'summary': summary_text,
            'review_note': review_note,
            'is_low_data_category': is_low_data
        })
        
        # Store comparison data
        comparison_data.append({
            "name": product_name,
            "rating": avg_score,
            "reviews": review_count,
            "sentiment": avg_sentiment,
            "recommend": recommend_ratio,
            "helpfulness": helpfulness,
            "composite": composite_score,
            "scoring_method": scoring_method
        })

    # Create DataFrame from summaries data
    summary_df = pd.DataFrame(summaries_data)
    
    # Handle empty dataframe
    if summary_df.empty:
        print("Warning: No valid products found for summary generation")
        return None
    
    # Add ranking information with proper NaN handling
    try:
        # Fill NaN values before ranking to avoid conversion errors
        summary_df['avg_rating_clean'] = summary_df['avg_rating'].fillna(0)
        summary_df['composite_score_clean'] = summary_df['composite_score'].fillna(0)
        summary_df['review_count_clean'] = summary_df['review_count'].fillna(0)
        
        # Create rankings
        summary_df['rating_rank'] = summary_df['avg_rating_clean'].rank(ascending=False, method='min').fillna(0).astype(int)
        summary_df['composite_rank'] = summary_df['composite_score_clean'].rank(ascending=False, method='min').fillna(0).astype(int)
        summary_df['review_count_rank'] = summary_df['review_count_clean'].rank(ascending=False, method='min').fillna(0).astype(int)
        
        # Drop the temporary clean columns
        summary_df.drop(['avg_rating_clean', 'composite_score_clean', 'review_count_clean'], axis=1, inplace=True)
        
    except Exception as e:
        print(f"Warning: Could not create rankings due to error: {e}")
        # Add default rankings if ranking fails
        summary_df['rating_rank'] = list(range(1, len(summary_df) + 1))
        summary_df['composite_rank'] = list(range(1, len(summary_df) + 1))
        summary_df['review_count_rank'] = list(range(1, len(summary_df) + 1))
    
    # Add key insights
    if comparison_data:
        try:
            # Handle potential NaN values in comparison data
            valid_ratings = [item for item in comparison_data if not pd.isna(item['rating']) and item['rating'] > 0]
            valid_reviews = [item for item in comparison_data if not pd.isna(item['reviews']) and item['reviews'] > 0]
            valid_composite = [item for item in comparison_data if not pd.isna(item['composite']) and item['composite'] > 0]
            
            insights_data = []
            
            if valid_ratings:
                best_rated = max(valid_ratings, key=lambda x: x['rating'])
                insights_data.append({
                    'metric': 'Highest Rated Product', 
                    'product_name': best_rated['name'], 
                    'value': f"{best_rated['rating']:.2f}/5.0"
                })
            
            if valid_reviews:
                most_reviewed = max(valid_reviews, key=lambda x: x['reviews'])
                insights_data.append({
                    'metric': 'Most Reviewed Product', 
                    'product_name': most_reviewed['name'], 
                    'value': f"{most_reviewed['reviews']} reviews"
                })
            
            if valid_composite:
                highest_composite = max(valid_composite, key=lambda x: x['composite'])
                insights_data.append({
                    'metric': 'Best Overall Score', 
                    'product_name': highest_composite['name'], 
                    'value': f"{highest_composite['composite']:.2f}"
                })
            
            # Add data quality information
            scoring_methods = set(item.get('scoring_method', 'unknown') for item in comparison_data)
            simplified_count = sum(1 for item in comparison_data if 'simplified' in item.get('scoring_method', ''))
            
            insights_data.append({
                'metric': 'Data Quality Note', 
                'product_name': f"Scoring methods: {', '.join(scoring_methods)}", 
                'value': f"{simplified_count} products with limited data"
            })
            
            insights_df = pd.DataFrame(insights_data) if insights_data else pd.DataFrame()
            
        except Exception as e:
            print(f"Warning: Could not generate insights due to error: {e}")
            insights_df = pd.DataFrame()
    else:
        insights_df = pd.DataFrame()

    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_name = summary_df['category'].iloc[0].replace(' ', '_') if not summary_df.empty else 'products'
        output_filename = f"product_summary_{category_name}_{timestamp}.csv"
    
    # Ensure .csv extension
    if not output_filename.endswith('.csv'):
        output_filename += '.csv'
    
    try:
        # Save main summary to CSV
        summary_df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"Product summaries saved to: {output_filename}")
        
        # Save insights to separate CSV if there are insights
        if not insights_df.empty:
            insights_filename = output_filename.replace('.csv', '_insights.csv')
            insights_df.to_csv(insights_filename, index=False, encoding='utf-8')
            print(f"Key insights saved to: {insights_filename}")
        
        return output_filename
        
    except Exception as e:
        print(f"Error saving files: {e}")
        return None

# Initialize with enhanced scorer
scorer = SimpleProductScorer(df_combined, model_path=rf_classifier_model_path)
product_scores = scorer.calculate_scores()

# Get top products for low-data categories

top_batteries = scorer.get_top_products_by_category('Connected Home Electronics', top_k=3)


# Generate summaries (will automatically handle low-data scenarios)
office_supplies_summary_file = generate_summary(top_batteries, df_combined, "Connected_Home_Electronics.csv")

