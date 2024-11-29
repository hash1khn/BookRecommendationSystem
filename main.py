import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import random  # For random user selection

# Load datasets
def load_data():
    books = pd.read_csv(
        'Books.csv',
        encoding='latin1',
        delimiter=',',
        dtype={
            'ISBN': 'str',
            'Book-Title': 'str',
            'Book-Author': 'str',
            'Year-Of-Publication': 'str',
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

    users = ratings['User-ID'].unique()
    books = ratings['ISBN'].unique()

    B.add_nodes_from(users, bipartite='user')
    B.add_nodes_from(books, bipartite='book')

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
def hybrid_recommend_books(
    user_id, B, book_graph, user_graph, users, books, ratings, min_user_ratings=50, top_n=5
):
    user_books = set(B.neighbors(user_id))

    # Step 1: Check if user meets the minimum ratings requirement
    user_rating_count = len(user_books)
    if user_rating_count < min_user_ratings:
        print(
            f"User {user_id} has only {user_rating_count} ratings, "
            f"which is below the minimum threshold of {min_user_ratings}. "
            f"Falling back to user similarity-based recommendations."
        )
    else:
        print(
            f"User {user_id} meets the minimum threshold with {user_rating_count} ratings. "
            f"Attempting book similarity-based recommendations."
        )

    # Step 2: Book-based recommendations
    book_scores = {}
    if user_rating_count >= min_user_ratings:
        for book in book_graph.nodes:
            if book not in user_books:
                score = sum(
                    book_graph[book][neighbor]['weight']
                    for neighbor in user_books
                    if book_graph.has_edge(book, neighbor)
                )
                book_scores[book] = score

        book_recommendations = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)

        # If sufficient book-based recommendations, return them
        if len(book_recommendations) >= top_n:
            recommendations = [
                {"type": "Book Similarity", "book": book, "score": score}
                for book, score in book_recommendations[:top_n]
            ]
            print(f"Recommendations based on book similarity.")
            return recommendations

    # Step 3: User-based recommendations (fallback)
    similar_users = user_graph.neighbors(user_id)
    for sim_user in similar_users:
        sim_weight = user_graph[user_id][sim_user]['weight']
        sim_user_books = set(B.neighbors(sim_user)) - user_books
        for book in sim_user_books:
            book_scores[book] = book_scores.get(book, 0) + sim_weight

    hybrid_recommendations = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = [
        {"type": "User Similarity", "book": book, "score": score}
        for book, score in hybrid_recommendations
    ]

    # Provide fallback reason
    if user_rating_count < min_user_ratings:
        print(
            f"User {user_id} did not have enough ratings ({user_rating_count}) to make "
            f"book similarity-based recommendations. Recommendations are based on user similarity."
        )
    return recommendations

# Test recommendations for a user
def test_recommendations(user_id, books, users, recommendations):
    # Fetch user details
    user_details = users[users['User-ID'] == user_id].iloc[0]
    user_location = user_details['Location']
    user_age = user_details.get('Age', 'Unknown')

    # Display recommendations
    print(f"\nRecommendations for User {user_id} (Location: {user_location}, Age: {user_age}):")
    for rec in recommendations:
        book_title = books.loc[books['ISBN'] == rec['book'], 'Book-Title'].values[0]
        print(
            f"- {book_title} (Score: {rec['score']:.2f}) [Based on {rec['type']}]"
        )

def visualize_graphs(B, book_graph, user_graph):
    # Create a custom layout for bipartite graph (vertical arrangement)
    pos = {}
    
    # Split the nodes into users and books
    user_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 'user']
    book_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 'book']
    
    # Assign vertical positions for users (left side)
    user_height = len(user_nodes)
    for i, user in enumerate(user_nodes):
        pos[user] = (0, i)  # X=0 for users, Y=sequential for vertical layout
    
    # Assign vertical positions for books (right side)
    book_height = len(book_nodes)
    for i, book in enumerate(book_nodes):
        pos[book] = (1, i)  # X=1 for books, Y=sequential for vertical layout
    
    # Plot the bipartite graph with custom layout
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(B, pos, nodelist=user_nodes, node_color='skyblue', node_size=50)
    nx.draw_networkx_nodes(B, pos, nodelist=book_nodes, node_color='orange', node_size=50)
    nx.draw_networkx_edges(B, pos, alpha=0.5)
    plt.title('User-Book Bipartite Graph')
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

    # Book projection visualization (optional, for further clarity)
    plt.figure(figsize=(12, 8))
    pos_proj = nx.spring_layout(book_graph)
    nx.draw(book_graph, pos_proj, node_size=10, node_color='lightgreen', with_labels=False)
    plt.title('Book Similarity Projection')
    plt.show()

    # User projection visualization (optional, for further clarity)
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

    # Randomly select a user from the filtered ratings
    random_user_id = random.choice(filtered_ratings['User-ID'].unique())

    # Test recommendations for the randomly selected user
    recommendations = hybrid_recommend_books(
        random_user_id, B, book_graph, user_graph, users, books, filtered_ratings
    )

    test_recommendations(random_user_id, books, users, recommendations)

    # Visualize the graphs
    visualize_graphs(B, book_graph, user_graph)

# Execute the script
if __name__ == "__main__":
    main()
