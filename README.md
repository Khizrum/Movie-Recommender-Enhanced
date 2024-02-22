# Original File:

- https://www.doczamora.com/content-based-recommender-system-for-movies-with-tensorflow

## Thesis:

- Extend the work done in initial file using Tensorflow library
- In the original file the recommendations were drawn based on 1 parameter, which is the 'Genre' or 'features' of the film
- I added another parameter, which is 'Languages' of films
- I also increased movie dataset from '6' to '20'

### Key Operations:

- ```tf.matmul```: through dot product helps in mapping values to corresponding elements, for e.g we used this function to map movie ratings onto genres and languages.
- ```tf.nn.top_k```: Orders the tensor in a descending order, for e.g we used this function to order movie recommendations, where the movies with highest score were displayed first.
- ```tf.transpose```: Interchanges rows into columns or columns into rows, for e.g we used this function to map genre and language ratings onto each movie for each user.
- ```tf.math.add```: Adds value of each element of a tensor to the corresponding value of other tensor, for e.g we used this function to combine the movie ratings associated with Genre and movie ratings associated with language with each other.
- ```tf.zeros_like```: Turns non zero values in a tensor to zero, for e.g we used this function to remove the movies that were already rated by the users.



