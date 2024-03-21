import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
with open('knn_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the movie features dataframe
movie_features_df = pd.read_csv('movie_features.csv', index_col=0)

# Function to get movie recommendations
def get_movie_recommendations(movie_name, model, movie_features_df, num_recommendations):
    # Check if the movie exists in the dataframe
    if movie_name in movie_features_df.index:
        # Find the index of the movie
        movie_index = movie_features_df.index.get_loc(movie_name)

        # Perform recommendation using the loaded model
        distances, indices = model.kneighbors(movie_features_df.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=num_recommendations + 1)

        # Get recommendations
        recommendations = []
        for i in range(0, len(distances.flatten())):
            if i != 0:
                recommendations.append((movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))

        return recommendations
    else:
        return "Movie not found in the database"

# Function to plot recommendations
def plot_recommendations(recommendations,movie_features_df):
    # Extract movie names and distances
    movie_names = [rec[0] for rec in recommendations]
    distances = [rec[1] for rec in recommendations]

    # Create a horizontal bar plot for recommendations
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=distances, y=movie_names, hue=movie_names, palette='viridis', ax=ax, legend=False)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Movie Title')
    ax.set_title('Top Recommendations')
    ax.grid(axis='x')
    plt.tight_layout()
    st.pyplot(fig)

    # Plot number of user ratings for each recommended movie
    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.barplot(x=movie_features_df.loc[movie_names].sum(axis=1), y=movie_names, palette='viridis', ax=ax)
    # ax.set_xlabel('Number of Ratings')
    # ax.set_ylabel('Movie Title')
    # ax.set_title('Number of User Ratings for Recommended Movies')
    # plt.tight_layout()
    # st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_labels = [f"{movie_name} (Distance: {distance:.2f})" for movie_name, distance in recommendations]

    count_ratings = []
    avg_ratings = []
    for movie_name, _ in recommendations:
        rating_info = movie_features_df.loc[movie_name]
        count_ratings.append(rating_info[rating_info != 0].count())
        avg_ratings.append(rating_info[rating_info != 0].mean())

    width = 0.35
    ind = np.arange(len(recommendations))
    rects1 = ax.barh(ind - width/2, count_ratings, width, label='Count of Ratings', color='skyblue')
    rects2 = ax.barh(ind + width/2, avg_ratings, width, label='Average Rating', color='orange')

    ax.set_yticks(ind)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Value')
    ax.set_title('Recommendations with Count of Ratings and Average Rating')
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate(f'{width:.2f}', xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    st.pyplot(fig)
# Streamlit app interface
def main():
    st.title('Movie Recommendation System')

    # Create an empty container for the sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        'This is a movie recommendation system based on collaborative filtering using nearest neighbors.'
        'It provides recommendations based on user-selected movies.'
        'The model was trained on the MovieLens dataset.'
    )

    # Display dataset statistics
    st.sidebar.title("Dataset Statistics")
    # st.sidebar.write('Total training data:', len(movie_features_df))
    st.sidebar.write('Number of available movies:', movie_features_df.shape[0])
    st.sidebar.write('Number of Users involved in rating :', movie_features_df.shape[1])

    # Movie input with autocomplete
    movie_name = st.selectbox('Enter a movie name:', [''] + movie_features_df.index.tolist())

    # If a movie is selected, display the sidebar
    
          
    # Slider to select the number of recommendations
    num_recommendations = st.slider('Select number of recommendations:', 1, 10, 5)

    # Get recommendations if movie name is provided
    if movie_name:
        if movie_name != '':
            recommendations = get_movie_recommendations(movie_name, loaded_model, movie_features_df, num_recommendations)
            if isinstance(recommendations, str):    
                st.write(recommendations)
            else:
                st.write(f"Top {num_recommendations} Recommendations for {movie_name}:")
                for i, (recommended_movie, distance) in enumerate(recommendations):
                    st.write(f"{i+1}. {recommended_movie} (Distance: {distance:.2f})")

                # Plot recommendations
                st.write("\n")
                st.write("### Visual Representation of Recommendations")
                # plot_recommendations(recommendations)
                plot_recommendations(recommendations, movie_features_df)

# Run the app
if __name__ == '__main__':
    main()
