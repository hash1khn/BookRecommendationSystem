import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt


# Load datdhdhdbdbdbfbd
def load_data();
    books = pd.read_csv(
    'Books.csv',
    encoding='latin1',
    delimiter=',',
    dtype={
        'ISBN': 'str',
        'Book-Title': 'str',
        'Book-Author': 'str',
        'Year-Of-Publication': 'str',  # Force as string to handle mixed types
        'Publisher': 'str',
        'Image-URL-S': 'str',
        'Image-URL-M': 'str',
        'Image-URL-L': 'str'
    }
)
    ratings = pd.read_csv('Ratings.csv', encoding='latin1', delimiter=',')
    users = pd.read_csv('Users.csv', encoding='latin1', delimiter=',')
    return books, ratings, users


# Preprocess the data
def preprocess_data(books, ratings, users):
    # Filter out books and users with too few ratings/interactions
    user_counts = ratings['User-ID'].value_counts()
    book_counts = ratings['ISBN'].value_counts()

    active_users = user_counts[user_counts >= 50].index
    popular_books = book_counts[book_counts >= 100].index

    filtered_ratings = ratings[
        (ratings['User-ID'].isin(active_users)) & 
        (ratings['ISBN'].isin(popular_books))
    ]

    return filtered_ratings


# Create a bipartite graph
def create_bipartite_graph(ratings):
    B = nx.Graph()

    # Add user nodes and book nodes
    users = ratings['User-ID'].unique()
    books = ratings['ISBN'].unique()

    B.add_nodes_from(users, bipartite='user')
    B.add_nodes_from(books, bipartite='book')

    # Add edges between users and books
    edges = list(ratings[['User-ID', 'ISBN']].itertuples(index=False, name=None))
    B.add_edges_from(edges)

    return B


# Create a book projection graph
def create_book_projection(B):
    book_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 'book'}
    book_graph = bipartite.weighted_projected_graph(B, book_nodes)
    return book_graph


# Create a user projection graph
def create_user_projection(B):
    user_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 'user'}
    user_graph = bipartite.weighted_projected_graph(B, user_nodes)
    return user_graph


# Recommend books based on hybrid logic
def hybrid_recommend_books(user_id, B, book_graph, user_graph, ratings, top_n=5):
    # Get books the user has already rated
    user_books = set(B.neighbors(user_id))

    # Step 1: Book-based recommendations
    book_scores = {}
    for book in book_graph.nodes:
        if book not in user_books:
            score = sum(
                book_graph[book][neighbor]['weight']
                for neighbor in user_books
                if book_graph.has_edge(book, neighbor)
            )
            book_scores[book] = score

    # Sort book-based scores
    book_recommendations = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)

    # If book-based recommendations are sufficient, return them
    if len(book_recommendations) >= top_n:
        return [book for book, score in book_recommendations[:top_n]]

    # Step 2: User-based recommendations
    similar_users = user_graph.neighbors(user_id)
    for sim_user in similar_users:
        sim_weight = user_graph[user_id][sim_user]['weight']
        sim_user_books = set(B.neighbors(sim_user)) - user_books

        for book in sim_user_books:
            # Use similarity weight as the score
            book_scores[book] = book_scores.get(book, 0) + sim_weight

    # Combine and sort final scores
    hybrid_recommendations = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [book for book, score in hybrid_recommendations]


# Visualize graphs
def visualize_graphs(B, book_graph, user_graph):
    # Bipartite graph visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(B)
    nx.draw(B, pos, node_size=10, node_color='skyblue', with_labels=False)
    plt.title('User-Book Bipartite Graph')
    plt.show()

    # Book projection visualization
    plt.figure(figsize=(12, 8))
    pos_proj = nx.spring_layout(book_graph)
    nx.draw(book_graph, pos_proj, node_size=10, node_color='lightgreen', with_labels=False)
    plt.title('Book Similarity Projection')
    plt.show()

    # User projection visualization
    plt.figure(figsize=(12, 8))
    pos_user = nx.spring_layout(user_graph)
    nx.draw(user_graph, pos_user, node_size=10, node_color='coral', with_labels=False)
    plt.title('User Similarity Projection')
    plt.show()


# Main function
def main():
    # Load datasets
    books, ratings, users = load_data()

    # Preprocess ratings data
    filtered_ratings = preprocess_data(books, ratings, users)

    # Create the bipartite graph
    B = create_bipartite_graph(filtered_ratings)

    # Create the book projection graph
    book_graph = create_book_projection(B)

    # Create the user projection graph
    user_graph = create_user_projection(B)

    # Test recommendations for a user
    test_user = filtered_ratings['User-ID'].iloc[0]
    recommendations = hybrid_recommend_books(test_user, B, book_graph, user_graph, filtered_ratings)
    print(f"Hybrid Recommendations for User {test_user}:", recommendations)

    # Visualize the graphs
    visualize_graphs(B, book_graph, user_graph)


# Execute the script
if __name__ == "__main__":
    main()
