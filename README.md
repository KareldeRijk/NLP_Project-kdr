# Product Scorer and Review Generator project
<<<<<<< HEAD

The goal of this project is to build a model, deployed through a website, that finds the top 3 products per product category. The dataset used is the Amazon product reviews from kaggle found at https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data
 
The final file setup is as follows:
- DataFrame Loading.ipynb loads the datasets and preprocesses via cleaning and embedding. The output is a dataframe containing the relevant information for the rest of the program. We separate loading and preprocessing to make the main program run faster. 
- Main Program Final takes the DataFrame from the above notebook, and uses Sentiment Analysis + a scoring algorithm to find the top 3 products per meta-category. The output is a .csv file which contains the top 3 products for a given meta-category plus relevant metrics. We run this program over each meta-category, then concatenate these into one .csv file (Summaries_Combined) the DataFrame Loading and Concatenating notebook.
- Gradio Final Model deploys the results from the Summaries_Combined DataFrame on a website via Gradio. 

Folder Structure
- .gradio contains draft notebooks for gradio web deployment
- Classifier_model contains drafts and the final version of the Sentiment Classifier Model
- Clustering_model contains notebooks used to extract the meta-categories out of the dataset (Extracting_Meta_Categories) and drafts (in the drafts folder) of trained classifier models and a notebook used to create a training data set with the meta-categories and the original dataset.  
-


Project Pipeline

First, we tried out different vectorizers and supervised learning algorithms to create a sentiment analysis. We landed on a combination of TF-IDF and RandomForest as that gave the best overall results. We evaluated using precision, recall, f1 score and accuracy plus some confusion matrixes. 

Second, we used a combination of transformers (to find semantic embeddings) and unsupervised learning (K-Means) to narrow all product categories down into 6 meta-categories (as per the assignment instructions). We then mapped these meta-categories on the dataset.
    We attempted to train a classification model on the meta-category mappings, so that we could use that to map the meta-categories on the original dataset.  However, this did not yiel any meaningfully better results, so we stuck with the simpler solution to use a dictionary mapping (with key: original categories , list : meta-category)

Third, with some help from Claude AI, we designed a product scoring system to take the most relevant factors into account when finding the top 3 products pet meta-category. We believe that this is where the key added value and challenge lies in this project: how do you find the best reviewed product in a massive pile of data? Which data do you use? After some trial and Error, we settled on an algorithm that creates a weighted composite score of: average rating (40%), review sentiment (analysed via the RandomForest model created earlier) (30%), Recommendations by other users (20%) and helpfullness score by other users (10%). This system creates a composite score from all factors that we considered relevant in the data, such as: amount of reviews, review sentiment, average review score, recommendations etc. 

Fourth, we call chatgpt-3.5-turbo with some prompt engineering (provide example, temperature low, specific instructions) to create a summary of a sample of 10 reviews of the top 3 products. Steps three and four are done in the Final Model python file. 
    We tried evaluating the generated summaries using ROUGE. However, due to a combination of time constraints and a lack of meaningful reference data, we were not able to generate meaningful evaluation scores. As reference text, we used the user reviews from the original dataset. However, the quality of most user reviews is not great so this provides a challenge in evaluating the model summaries.  

Lastly, we deployed our results via a Gradio website hoster. To make the website run smoothly, we load a dataframe created as output from the scoring and summary model. 
