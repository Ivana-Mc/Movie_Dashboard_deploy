
import streamlit as st
import pandas as pd
import plotly.express as px
import random

# Page config
st.set_page_config(page_title="Movie Recommender Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("🔎 Navigation")
st.sidebar.markdown("**👉 Choose a section to explore:**")
section = st.sidebar.radio(
    "Go to:",
    ["📊 Data Overview", "🧠 Clustering", "🤝 Recommendations"],
    label_visibility="collapsed"
)

# Load data
df = pd.read_csv("merged_df_clustered_KMeans.csv")
cluster_df = pd.read_csv("clustered_pca.csv")
cluster_summary_df = pd.read_csv("movies_with_clusters_summary.csv")
user_user_df = pd.read_csv("user_user_recommendations.csv")
item_item_df = pd.read_csv("item_item_recommendations.csv")
cluster_rec_df = pd.read_csv("cluster_recommendations.csv")

# Title
st.title("🎥 Movie Recommender Dashboard")
st.caption("Discover movies based on your unique taste")
st.markdown("Built using real user rating behaviour and machine learning clustering insights.")

# =========================================
# DATA OVERVIEW SECTION
# =========================================
if section == "📊 Data Overview":
    st.markdown("### 🔍 Dataset Overview")
    num_users = df['userId'].nunique()
    num_movies = df['movieId'].nunique()
    avg_rating = round(df['rating'].mean(), 2)
    num_clusters = df['cluster'].nunique()
    avg_ratings_per_user = round(df.groupby('userId')['movieId'].nunique().mean(), 1)

    col1, col2, col3, col4, col5 = st.columns(5)

    box_style = """
        background-color:{bg}; 
        padding:2px; 
        border-radius:6px; 
        text-align:center; 
        border: 1px solid #ccc; 
        height: 65px; 
        display: flex; 
        flex-direction: column; 
        justify-content: center;
      """

    text_style = "margin:0; font-size:16px;"
    number_style = "margin:0; font-size:16px; font-weight:bold;"

    with col1:
        st.markdown(f"""
        <div style='{box_style.format(bg="#f0f8ff")}'>
            <p style='{number_style} color:#1f77b4;'>👥 {num_users}</p>
            <p style='{text_style}'>Users</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='{box_style.format(bg="#fff5e6")}'>
            <p style='{number_style} color:#ff7f0e;'>🎬 {num_movies}</p>
            <p style='{text_style}'>Movies</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='{box_style.format(bg="#e8f5e9")}'>
            <p style='{number_style} color:#2ca02c;'>⭐ {avg_rating}</p>
            <p style='{text_style}'>Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='{box_style.format(bg="#fbe9e7")}'>
            <p style='{number_style} color:#d62728;'>🔢 {num_clusters}</p>
            <p style='{text_style}'>Clusters</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div style='{box_style.format(bg="#ede7f6")}'>
            <p style='{number_style} color:#9467bd;'>📊 {avg_ratings_per_user}</p>
            <p style='{text_style}'>Avg Movies/User</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Ratings Distribution Overview")
    col_user, col_movie = st.columns(2)

    with col_user:
        st.markdown("####👤 Ratings per User")
        ratings_per_user = df.groupby('userId')['movieId'].count().reset_index(name='rating_count')
        fig_user = px.histogram(ratings_per_user, x='rating_count', nbins=50, 	template='plotly_white')
        fig_user.update_traces(marker_color='#6C5CE7')
        fig_user.update_layout(height=300)
        st.plotly_chart(fig_user, use_container_width=True)

    with col_movie:
        st.markdown("####🎬 Ratings per Movie")
        ratings_per_movie = df.groupby('movieId')['rating'].count().reset_index(name='rating_count')
        fig_movie = px.histogram(ratings_per_movie, x='rating_count', nbins=50, 	template='plotly_white')
        fig_movie.update_traces(marker_color='#6C5CE7')
        fig_movie.update_layout(height=300)
        st.plotly_chart(fig_movie, use_container_width=True)

    st.markdown("Most users rate fewer than 500 movies, while many movies receive fewer than 10 	ratings. This highlights the **data sparsity** problem in collaborative filtering.")

    if 'genres' in df.columns:
        st.markdown("### 🎭 Ratings by Genre")
        genre_df = df.copy()
        genre_df['genre'] = genre_df['genres'].str.split('|').str[0]
        avg_genre_rating = genre_df.groupby('genre')['rating'].mean().reset_index()
        fig_genre = px.bar(avg_genre_rating, x='genre', y='rating', template='plotly_white')
        fig_genre.update_traces(marker_color='#6C5CE7')
        fig_genre.update_layout(title="Average Rating by Genre",title_x=0.5, height=350)
        
        st.plotly_chart(fig_genre, use_container_width=True)
        st.markdown("Genres reveal user preference trends. Genres like **Film-Noir**, **Mystery**, 	and **Crime** tend to receive higher ratings.")

    if 'title' in df.columns:
        st.markdown("### 🏆 Top 10 Most Rated Movies")
        top_movies = df['movieId'].value_counts().head(10).index.tolist()
        top_movies_df = df[df['movieId'].isin(top_movies)]
        top_titles = top_movies_df.groupby('title')	['rating'].count().sort_values(ascending=True).reset_index()
        fig_top = px.bar(top_titles, x='rating', y='title', orientation='h', template='plotly_white')
        fig_top.update_traces(marker_color='#6C5CE7')
        fig_top.update_layout(title="Top 10 Most Rated Movies", title_x=0.5, height=400)
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("### 👥 User Activity vs Average Rating")
    user_stats = df.groupby('userId')['rating'].agg(['count', 'mean']).reset_index()
    fig_user_behaviour = px.scatter(user_stats, x='count', y='mean', template='plotly_white',
        labels={'count': 'Number of Ratings', 'mean': 'Average Rating'})
    fig_user_behaviour.update_traces(marker=dict(color='#6C5CE7'))
    fig_user_behaviour.update_layout(title="User Rating Count vs Average Score", height=350, title_x=    	0.5)
    st.plotly_chart(fig_user_behaviour, use_container_width=True)

    st.markdown("---")
    st.caption("🎓 Created by Ivana McFadden | CCT College Dublin | Lecturer: David McQuaid | 2025 	")

# =========================================
# CLUSTERING SECTION
# =========================================
elif section == "🧠 Clustering":
    st.markdown("### 🎯 Explore Movie Clusters")
    st.write("Use the filter to explore a specific cluster. Each point is a movie, positioned using PCA of user ratings.")

    left_col, right_col = st.columns([4, 1])

    with right_col:
        st.markdown("""
        <style>
        .animated-underline {
            display: inline-block;
            font-size: 16px;
            font-weight: bold;
            color: #e377c2;
            position: relative;
        }
        .animated-underline::after {
            content: '';
            position: absolute;
            left: 0; bottom: -3px;
            height: 2px;
            width: 100%;
            background-color: #e377c2;
            transform: scaleX(0);
            transition: transform 0.3s ease;
            transform-origin: left;
        }
        .animated-underline:hover::after {
            transform: scaleX(1);
        }
        .banner-label {
            background-color: #e377c2;
            color: white;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            margin-top: 8px;
            text-align: center;
        }
        </style>
        <div class='animated-underline'>🎯 Select Cluster</div>
        <div class='banner-label'>Filter Active</div>
        """, unsafe_allow_html=True)

        selected_cluster = st.selectbox(
            "Cluster Selection (hidden label)",
            ["All Clusters"] + sorted(cluster_df['Cluster'].unique().tolist()),
            label_visibility="collapsed"
        )
  # Filter data
    if selected_cluster != "All Clusters":
        filtered_df = cluster_df[cluster_df['Cluster'] == selected_cluster]
    else:
        filtered_df = cluster_df

    # Scatterplot in left column
    with left_col:
        fig = px.scatter(
            filtered_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data={'Title': True, 'Cluster': True},
            title="🎬 Movie Clusters (PCA View)",
             template='plotly_white',
            opacity=0.8
        )

        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(
            title_x=0.5,
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
            legend_title_text='Cluster'
        )

        st.plotly_chart(fig, use_container_width=True)
        
#--------------------------------------------------------------------------------------------------------
    # --- Summary Stats for Selected Cluster ---
   # --- Summary Stats for Selected Cluster ---

  
        st.markdown("### 📌 Cluster Summary")

     
        # Get selected cluster
        selected_cluster = st.selectbox(
            "Cluster Selection (hidden label)",
            ["All Clusters"] + sorted(cluster_summary_df['cluster'].unique().tolist()),
            label_visibility="collapsed",
            key="cluster_summary_selectbox"
        )
        
        
        # Filter the summary dataframe
        if selected_cluster != "All Clusters":
            cluster_data = cluster_summary_df[cluster_summary_df['cluster'] == selected_cluster]
        else:
            cluster_data = cluster_summary_df

# Compute metrics
        cluster_num_movies = cluster_data['movieId'].nunique()
        cluster_avg_rating = round(cluster_data['avg_rating'].mean(), 2)

# Genre extraction
        cluster_data['Main Genre'] = cluster_data['genres'].str.split('|').str[0]
        top_genre = cluster_data['Main Genre'].value_counts().idxmax()

# Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🎬 Movies in Cluster", cluster_num_movies)
        col2.metric("⭐ Avg Rating", cluster_avg_rating)
        col3.metric("🎭 Top Genre", top_genre)
# --- Top 5 Most Rated Movies in Selected Cluster ---
        if selected_cluster != "All Clusters":
            st.markdown("### 🎬 Top 5 Most Rated Movies")

    # Sort and select top 5
        top_movies = (
            cluster_data.sort_values(by='rating_count', ascending=False)
                    .head(5)
        )

    # Plot horizontal bar chart
        fig_top_movies = px.bar(
            top_movies,
            x='rating_count',
            y='title',
            orientation='h',
            template='plotly_white',
            title="Most Rated Movies in This Cluster"
        )
        fig_top_movies.update_traces(marker_color='#6C5CE7')
        fig_top_movies.update_layout(title_x=0.5, height=400)

    # Show chart
        st.plotly_chart(fig_top_movies, use_container_width=True)

#-----------------------------------------------------------------------------------    


            
        st.markdown("---")
        st.caption("🎓 Created by Ivana McFadden | CCT College Dublin | Lecturer: David McQuaid | 2025 ")

# =========================================
# RECOMMENDATION 
# =========================================
elif section == "🤝 Recommendations":
    st.markdown("## 🤝 Movie Recommendations")
    st.write("🎬 Discover movies you’ll love — based on your ratings, your vibe, and what similar users enjoyed.")

    
    # Prepare user list
    available_users = sorted(user_user_df['userId'].unique())

    # first run
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = available_users[0]

    # Surprise Me button
    if st.button("🎲 Surprise Me", help="Feeling lucky? We'll pick a user for you!"):
        st.session_state.selected_user = random.choice(available_users)
        st.rerun()

    # Dropdown to select user
    selected_user = st.selectbox("Select a user ID", available_users,
                                 index=available_users.index(st.session_state.selected_user),
                                 key="selected_user")

#-----------------------------------------------------------------------------------------------------
    # Tabs for recommendation methods
    tab1, tab2, tab3 = st.tabs(["👥 User-Based CF", "🎯 Item-Based CF", "🧠 Cluster-Based Rec"])

    with tab1:
        st.markdown("### 👥 What people like you also enjoyed")
        st.markdown("Movies loved by users who rate like you. It’s collaborative filtering at work!")

        user_recs = user_user_df[user_user_df['userId'] == selected_user]
        if not user_recs.empty:
            top_recs = user_recs.sort_values(by='adjusted_rating', ascending=False).head(10)
            st.dataframe(top_recs[['title', 'genres', 'rating', 'adjusted_rating']])
        else:
            st.warning("No user-user recommendations available for this user.")

    with tab2:
        st.markdown("### 🎯 If you liked it, you’ll probably love these too")
        st.markdown("Suggestions based on the movies you've rated 4⭐ or higher. Basically: 'More like that one'.")

        item_recs = item_item_df[item_item_df['userId'] == selected_user]
        if not item_recs.empty:
            top_recs = item_recs.sort_values(by='score', ascending=False)
            st.dataframe(top_recs[['title', 'genres', 'score']])
        else:
            st.warning("No item-item recommendations available for this user.")

    with tab3:
        st.markdown("### 🧠 Discover more from your favourite genre cluster")
        st.markdown("We analysed your top-rated genres and clustered them to find movies that match your style.")

        cluster_recs = cluster_rec_df[cluster_rec_df['userId'] == selected_user]
        if not cluster_recs.empty:
            st.dataframe(cluster_recs[['title', 'genres', 'cluster', 'mean', 'count']].sort_values(by='mean', ascending=False))
        else:
            st.warning("No cluster-based recommendations available for this user.")

    st.markdown("---")
    st.caption("🎓 Created by Ivana McFadden | CCT College Dublin | Lecturer: David McQuaid | 2025 ")
