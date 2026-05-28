import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

try:
    movies=pd.read_csv('IMDB-Movie-Data.csv')
    st.write("✓ Movie Dataset loaded successfully!")
    st.write(f"Total movies: {len(movies)}")
    
    movies['Genre']=movies['Genre'].fillna('')
    tfidf=TfidfVectorizer(stop_words='english')
    tfidf_matrix=tfidf.fit_transform(movies['Genre'])
    cos_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
    indices=pd.Series(movies.index,index=movies['Title']).drop_duplicates()
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    import traceback
    st.write(traceback.format_exc())
    st.stop()
def recommend_movies(Title,num_recommendations=5):
    if Title not in indices:
        return ["Movie not found"]
    idx=indices[Title]
    sim_scores=list(enumerate(cos_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:num_recommendations+1]
    movie_indices=[i[0] for  i in sim_scores]
    return movies['Title'].iloc[movie_indices]

st.write("---")
st.subheader("Find Movie Recommendations")
movie_name=st.text_input("Enter a movie name:")
if st.button('Recommend'):
    if movie_name:
        recommendations=recommend_movies(movie_name)
        st.write("### Recommended movies:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("Please enter a movie name")

