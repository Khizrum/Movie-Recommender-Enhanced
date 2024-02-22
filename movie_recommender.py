
import tensorflow as tf


def data_initialization():
    global users
    global movies
    global features
    global language
    global num_users
    global num_movies
    global num_features
    global num_recommendations
    global num_languages

    users = ['Ryan', 'Javier', 'Rajat', 'Ibrahim']
    movies = ['Star wars', 'The Dark Knight', 'Shrek', 'The Incredibles', 'Bleu', 'Momento', 'Ra one',
              'Kuch Kuch Hota Hai', 'Welcome', 'Chhota Bheem', 'Andaz Apna Apna',
              'Matagheye Bohrani', 'The Pig', 'Badigard', 'The Salesman',
              'Los cronocrímenes', 'Nocturna', 'Primos', 'Mar adentro', 'Maixabel']

    features = ['Action', 'Sci-Fi', 'Comedy', 'Cartoon', 'Drama']
    language = ['English', 'Hindi', 'Persian', 'Spanish']

    num_users = len(users)
    num_movies = len(movies)
    num_features = len(features)
    num_languages = len(language)

    num_recommendations = 5


def dataset_char():
    global user_ratings
    global movie_feats
    global language_feats

    # user_ratings are the ratings for each movie by the 4 users
    # 0 means unrated movie by the user
    user_ratings = tf.constant([
        [4, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 3, 0, 0, 0, 6, 0, 0, 0, 0, 8, 0, 10, 7, 0, 0],
        [0, 6, 0, 0, 3, 7, 9, 7, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 5, 0, 0, 0, 0, 0, 0, 7, 0, 8, 6, 0, 10, 0, 0, 0, 7, 0, 9]], dtype=tf.float32)

    # We have associated a category to each movie
    # categories are mentioned in the features array above

    movie_feats = tf.constant([
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]], dtype=tf.float32)

    # We have associated a language to each movie
    # languages are mentioned in the language array above

    language_feats = tf.constant([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]], dtype=tf.float32)




def favorite_movie_feats():
    global movie_feat_rating
    global language_rating

    # Matrix Multiplication for finding which user likes which features
    movie_feat_rating = tf.matmul(user_ratings, movie_feats)
    language_rating = tf.matmul(user_ratings, language_feats)


    # Scale or Standardize the user_feats matrix
    movie_feat_rating = movie_feat_rating / tf.reduce_sum(movie_feat_rating, axis=1, keepdims=True)
    language_rating = language_rating / tf.reduce_sum(language_rating, axis=1, keepdims=True)



def top_features():
    global feature_names


    # Here we order the features and languages in the descending order based on their scores
    top_movie_features = tf.nn.top_k(movie_feat_rating, num_features)[1]
    top_language_features = tf.nn.top_k(language_rating, num_languages)[1]


    # This is translating each score to the feature it corresponds to
    # thus feature array gets reordered for each user with the most liked feature coming first

    for i in range(num_users):
        feature_names = [features[int(index)] for index in top_movie_features[i]]
        print('{}: {}'.format(users[i], feature_names))

    print('\n')

    for i in range(num_users):
        language_names = [language[int(index)] for index in top_language_features[i]]
        print('{}: {}'.format(users[i], language_names))

    print('\n')

def new_user_ratings():
    global user_ratings_new

    # This step helps us calculate even ratings for movies which were rated 0

    movie_feat_rate = tf.matmul(movie_feat_rating, tf.transpose(movie_feats))
    language_rate = tf.matmul(language_rating, tf.transpose(language_feats))

    #This step combines the scores of movie_feat_rate and language_rate for each movie
    #Thus now each recommendation is made considering both the elements
    user_rate = tf.math.add(movie_feat_rate, language_rate)
    user_rate = user_rate / tf.reduce_sum(user_rate, axis=1, keepdims=True)




    #This step helps us in reducing our matrix to only movies which were rated 0 or not seen by user
    user_ratings_new = tf.where(tf.equal(user_ratings, tf.zeros_like(user_ratings)),
                               user_rate, tf.zeros_like(tf.cast(user_ratings, tf.float32))
                                )



def movie_recommend():
    # This step sets the size of recommendations
    # Where index with highest scores are displayed first
    top_movies = tf.nn.top_k(user_ratings_new, num_recommendations)[1]
    for i in range(num_users):
        movie_names = [movies[index] for index in top_movies[i]]
        print('{}: {}'.format(users[i], movie_names))


if __name__ == '__main__':
    data_initialization()
    dataset_char()
    favorite_movie_feats()
    top_features()
    new_user_ratings()
    movie_recommend()

