import numpy as np
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
    # categories are mentioned features array above

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
    global user_feats
    global usera_feats
    global userb_feats

    new_row = np.array([0, 0, 0, 0])
    # Matrix Multiplication for finding which user likes which features
    usera_feats = tf.matmul(user_ratings, movie_feats)
    userb_feats = tf.matmul(user_ratings, language_feats)
    #userb_feats = np.vstack([userb_feats, new_row])
    #user_feats = tf.matmul(usera_feats, userb_feats)


    # Scale or Standardize the user_feats matrix
    usera_feats = usera_feats / tf.reduce_sum(usera_feats, axis=1, keepdims=True)
    userb_feats = userb_feats / tf.reduce_sum(userb_feats, axis=1, keepdims=True)
    #user_feats = user_feats / tf.reduce_sum(user_feats, axis=1, keepdims=True)

    #print(user_feats)


def top_features():
    global top_user_features
    global feature_names

    new_row = np.array([0, 0, 0, 0])

    # This used neural network to give a score to each movie category for each user
    top_usera_features = tf.nn.top_k(usera_feats, num_features)[1]
    top_userb_features = tf.nn.top_k(userb_feats, num_languages)[1]
    top_userb_features = np.vstack([top_userb_features, new_row])

    #print(top_usera_features)
    #print(top_userb_features)

    # This is translating each sore into the feature it corresponds to
    # thus feature array gets reordered for each user with the most liked feature coming first

    for i in range(num_users):
        feature_names = [features[int(index)] for index in top_usera_features[i]]
        print('{}: {}'.format(users[i], feature_names))
    for i in range(num_users):
        language_names = [language[int(index)] for index in top_userb_features[i]]
        print('{}: {}'.format(users[i], language_names))

def new_user_ratings():
    global user_ratings_new

    # This step helps us calculate even ratings for movies which were rated 0
    print('new_user_rating \n')

    usera_rate = tf.matmul(usera_feats, tf.transpose(movie_feats))
    userb_rate = tf.matmul(userb_feats, tf.transpose(language_feats))

    user_rate = tf.math.add(usera_rate, userb_rate)
    user_rate = user_rate / tf.reduce_sum(user_rate, axis=1, keepdims=True)

    print(usera_rate)




    #This step helps us in reducing our matrix to only movies which weren't rated or not seen by user
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_initialization()
    dataset_char()
    favorite_movie_feats()
    top_features()
    new_user_ratings()
    movie_recommend()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
