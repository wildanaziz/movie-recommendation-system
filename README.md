# Machine Learning Project Report - Wildan Aziz Hidayat

![movie_banner](images/banner.jpg)  
Figure 1.0: A collection of popular movies

## Project Overview

Movie recommendation system is one of the popular applications of machine learning and has a significant impact on the entertainment industry. This project aims to develop a recommendation system based on content-based filtering and collaborative filtering.

This system will help users find movies that match their preferences, based on user rating data and movie genre information. This approach not only provides a better personalized experience for users but also increases the engagement of streaming platforms.

This method aims to fix the issue of specific suggestions, which happens when data relevant to the user is overlooked. It extracts the user’s personality type, previous viewing experience, and data from many other databases concerning film critics. These are focused on an estimate of composite similarity. The methodology is a hybrid system that combines content-based and collaborative filtering techniques. Hongli LIn et al. suggested a process called content-boosted collaborative filtering to predict the model complexity of a specific instance for each candidate (CBCF). The method is divided into two stages: content-based filtering, which enhances information on current trainee scenario reviews, and collaborative filtering, which provides the most accurate predictions. The CBCF algorithm considers both Content-based and Collaborative approaches, addressing both of their shortcomings at the same time.[[1]](https://ieeexplore.ieee.org/document/9872515/authors#authors)

**Referensi**: [Movie Recommender System Using Content-based and Collaborative Filtering](https://ieeexplore.ieee.org/document/9872515/authors#authors)

## Business Understanding

### Problem Statements
1. How to provide movie recommendations to users based on their preferred genres?
2. How to leverage user rating data to provide more relevant recommendations?

### Goals
1. Get movie recommendations based on genres or themes from favorite movies with a Recall@K rate of more than 80%.
2. Get movie recommendations based on previously rated movies with an error of less than 50%.

### Solution Approach
- Developing a content-based filtering recommendation system that utilizes movie genre information using the TF-IDF and cosine similarity algorithms to create a movie recommendation system based on genres similar to favorite movies.
- Developing a collaborative filtering recommendation system using user rating data to improve the relevance of recommendations using an embedding approach with TensorFlow/Keras to build a collaborative filtering model with a recommendation system outcome similar to previously rated movies.


## Data Understanding

The dataset used is [MovieLens](https://grouplens.org/datasets/movielens/), which consists of two main files:
1. **movies.csv**: Contains information about movie titles and genres.
2. **ratings.csv**: Contains rating data given by users to movies.

Number of data:
- **movies.csv**: 10329 rows and 3 columns.
- **ratings.csv**: 105339 rows and 4 columns.

### Variables in the dataset:
#### movies.csv
- **movieId**: Unique ID for each movie.
- **title**: Movie title.
- **genres**: Movie genre in string format.

#### ratings.csv
- **userId**: Unique ID for each user.
- **movieId**: Unique ID for each movie.
- **rating**: Rating given by the user.
- **timestamp**: The time when the rating was given.

#### Info Data movies.csv

|   # | Column   | Non-Null Count | Dtype   |
| --: | -------- | -------------- | ------- |
|   0 | movieId  | 10329 non-null | int64   |
|   1 | title    | 10329 non-null | object  |
|   2 | genres   | 10329 non-null | object  |


#### Info Data ratings.csv

This dataset contains 105339 entries and 4 columns.

|   # | Column      | Dtype |
| --: | ----------- | ----- |
|   0 | userId      | int64 |
|   1 | movieId     | int64 |
|   2 | rating      | int64 |
|   3 | timestamp   | int64 |

#### Visualisasi Data Ratings
Figure 2.0 plots the rating column of the "rating" dataframe

In the "count" label, the maximum value is less than 30000.

In the "rating" label, there is a bar after the number 0, which means that many users gave a rating of 0.5, indicating the lowest rating, which is 0.5.

![movies_user_plot](images/users-movies.png)  
Figure 2.1 plots the movieId and userId columns of the "rating" dataframe

It can be seen that the number of unique users who gave ratings is at 6.1% and the number of unique movies that were rated is more than 93.9%. This data is **many-to-many** which means that 1 user can rate many movies and 1 movie can be rated by many users

## Data Preparation

### Techniques applied:
1. **Genre to list format**: Converting genre strings to list format for further analysis.
2. **Missing Values ​​Check**: Since there are no missing values ​​in the data obtained, no data is removed.
3. **Encoding userId and movieId**: Converting IDs to numeric representations so that they can be used in collaborative filtering modeling.
4. **Splitting Data**: Dividing data into training (80%) and validation (20%).

### Genre to List Format
- At this stage the genre of each movie in the **movies.csv** dataframe will be changed into an array (list) form. This is done to make it easier to access the genre in the "genres" column. The results are as follows

|   # | movieId  |                          title   |                                             genres |
| --: | -------: | ---------------------------------: | ------------------------------------------------: |
|   0 |    1     |                 Toy Story (1995)   | [Adventure, Animation, Children, Comedy, Fantasy] |
|   1 |    2     |                   Jumanji (1995)   |                    [Adventure, Children, Fantasy] |
|   2 |    3     |          Grumpier Old Men (1995)   |                                 [Comedy, Romance] |
|   3 |    4     |         Waiting to Exhale (1995)   |                          [Comedy, Drama, Romance] |
|   4 |    5     | Father of the Bride Part II (1995) |                                          [Comedy] |

This process is necessary to ensure that the data is ready to be used by machine learning algorithms and improve model performance.

### Checking Missing Values
- Because there are no missing values ​​in the given dataset, no data is deleted.

### Encoding userId dan movieId
- At this stage, the encoding process is carried out on the userId and movieId columns and then inserted into their respective new columns. This is done to represent the user and movie ids in a format that can be processed by the machine learning model.

### Splitting Data

- Splitting the dataframe into train and validation with a ratio of 80:20, but the data is shuffled first before being separated. This is done so that the model can evaluate new data and prevent overfitting.

## Modeling

### Content-Based Filtering
- **Approach**: Using TF-IDF to extract features from movie genres, followed by cosine similarity calculation.
- **Output**: Top-N movie recommendations based on genre similarity with the given movie input.

**Content-Based Filtering** technique is a recommendation system technique to recommend a product that has similarities with a product that is liked. In this case, the system will recommend movies based on genre similarity with movies that the user likes. This technique uses the **Cosine Similarity** formula to get a match between product 1 and another.

The formula for **Cosine Similarity** is:  
$\displaystyle cos~(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$

This technique uses the **TF-IDF Vectorizer** model to obtain information about the genres contained in each movie and convert it into features that can be measured for similarity. An example is as follows

| Title                                                       | horror | war | comedy | western | romance | drama | musical | sci  | thriller |
|-------------------------------------------------------------|--------|-----|--------|---------|---------|-------|---------|------|----------|
| Cloudy with a Chance of Meatballs (2009)                    | 0.0    | 0.0 | 0.0    | 0.0     | 0.000000 | 0.0   | 0.000000 | 0.0  | 0.00000  |
| Red (2008)                                                 | 0.0    | 0.0 | 0.0    | 0.0     | 0.000000 | 0.0   | 0.550387 | 0.0  | 0.83491  |
| Davy Crockett, King of the Wild Frontier (1955)             | 0.0    | 0.0 | 0.0    | 0.0     | 0.832344 | 0.0   | 0.000000 | 0.0  | 0.00000  |
| Ghost of Frankenstein, The (1942)                           | 0.0    | 1.0 | 0.0    | 0.0     | 0.000000 | 0.0   | 0.000000 | 0.0  | 0.00000  |
| Most Hated Family in America, The (2007)                    | 0.0    | 0.0 | 0.0    | 0.0     | 0.000000 | 0.0   | 0.000000 | 0.0  | 0.00000  |

Next, **Cosine Similarity** will be applied to the cleaned movies dataframe to produce the following output:

| Title                                                      | Guilty as Sin (1993) | Girl Who Kicked the Hornet's Nest, The (Luftslottet som sprängdes) (2009) | Leopard Man, The (1943) | American Haunting, An (2005) | Chain Reaction (1996) | Fellini Satyricon (1969) | Hereafter (2010) | Tetsuo, the Ironman (Tetsuo) (1988) | Attack the Block (2011) | Stone (2010) |
|------------------------------------------------------------|----------------------|----------------------------------------------------------------------------|------------------------|-----------------------------|-----------------------|--------------------------|------------------|-------------------------------------|-------------------------|--------------|
| Kingdom, The (2007)                                        | 0.530410             | 0.339414                                                                   | 0.385744               | 0.385744                    | 0.698413              | 0.167196                 | 0.167196         | 0.489422                            | 0.310663                | 0.739518    |
| Blackadder's Christmas Carol (1988)                         | 0.000000             | 0.000000                                                                   | 0.000000               | 0.000000                    | 0.000000              | 0.000000                 | 0.000000         | 0.000000                            | 0.344642                | 0.000000    |
| Pineapple Express (2008)                                    | 0.452900             | 0.656832                                                                   | 0.000000               | 0.000000                    | 0.343118              | 0.000000                 | 0.000000         | 0.240444                            | 0.437749                | 0.000000    |
| I Confess (1953)                                            | 0.598829             | 0.000000                                                                   | 0.435502               | 0.435502                    | 0.516846              | 0.000000                 | 0.000000         | 0.362186                            | 0.000000                | 0.834910    |
| Ink (2009)                                                  | 0.000000             | 0.206915                                                                   | 0.000000               | 0.000000                    | 0.231229              | 0.502118                 | 0.502118         | 0.670387                            | 0.783547                | 0.000000    |

In the table, you can see the compatibility of 1 movie with another. The values ​​in the table represent the percentage of compatibility between the two movies.

#### Getting top-N recommendations
The table is a cosine similarity dataframe that will be used to get top-N movie recommendations. In this case, we will try to get top-10 movie recommendations that are similar to the movie **"Sherlock Holmes (2010)"**. The output is as follows

Data for testing
| # | title                          | genres                           |
|--:|:------------------------------:|:--------------------------------:|
| 0 | Sherlock Holmes (2010)         | [Mystery, Sci-Fi]                |

Recommendation results

| Title                                                     | Genres                      |
|-----------------------------------------------------------|-----------------------------|
| Andromeda Strain, The (1971)                              | [Mystery, Sci-Fi]            |
| Stalker (1979)                                            | [Drama, Mystery, Sci-Fi]     |
| Solaris (Solyaris) (1972)                                  | [Drama, Mystery, Sci-Fi]     |
| Sound of My Voice (2011)                                  | [Drama, Mystery, Sci-Fi]     |
| Fire in the Sky (1993)                                    | [Drama, Mystery, Sci-Fi]     |
| Big Empty, The (2003)                                     | [Comedy, Mystery, Sci-Fi]    |
| Stepford Wives, The (1975)                                | [Mystery, Sci-Fi, Thriller]  |
| District 9 (2009)                                         | [Mystery, Sci-Fi, Thriller]  |
| Seconds (1966)                                            | [Mystery, Sci-Fi, Thriller]  |
| Twelve Monkeys (a.k.a. 12 Monkeys) (1995)                 | [Mystery, Sci-Fi, Thriller]  |

Based on the recommendation results, it can be seen that the recommended movies have genres similar to the input movies, namely the "Mystery" and "Sci-Fi" genres.

#### Advantages of Content-Based Filtering:
- Does not require user data.
- Suitable for systems with limited user data.

#### Disadvantages of Content-Based Filtering:
- Does not take into account variations in preferences between users.

### Collaborative Filtering
- **Approach**: Create embedding for users and movies using TensorFlow/Keras. The model is optimized using the binary crossentropy loss function.
- **Output**: Top-N movie recommendations for a particular user based on rating patterns.

The **Collaborative Filtering** technique is a recommendation system technique for recommending a product based on similarities in preferences between users. In this case, the system will use movies that are highly rated by users to find similarities with other users.

This project uses the **RecommenderNet** model created from the **Model** class owned by **Keras**. And then compiled using **Mean Absolute Error** metric, **Binary Crossentropy** loss function, and **Adam** optimizer. And then the model can be trained.

#### Getting top-N recommendations

We randomly pick a user from the ratings dataframe
```
Showing recommendations for user: 521
========================================
Movies with high ratings from user
----------------------------------------
True Lies (1994) : Action, Adventure, Comedy, Romance, Thriller
For Love or Money (1993) : Comedy, Romance
Much Ado About Nothing (1993) : Comedy, Romance
Sleepless in Seattle (1993) : Comedy, Drama, Romance
Mission: Impossible (1996) : Action, Adventure, Mystery, Thriller
----------------------------------------
```
Then all movies that have not been seen by the user will be taken, and the model will make predictions based on movies with high ratings by the user and their similarity to other users. The results will get the following recommendations

```
Top 10 movies recommendations
----------------------------------------
Grumpier Old Men (1995): Comedy, Romance
Turbo: A Power Rangers Movie (1997): Action, Adventure, Children
Take the Money and Run (1969): Comedy, Crime
Sixteen Candles (1984): Comedy, Romance
Fast Food, Fast Women (2000): Comedy, Romance
```

Based on the results of these recommendations, it shows movies that are relevant to previously rated movies. The recommendations given also vary and are not only limited to certain genres unlike **Content-Based Filtering**.

#### Advantages:
- Utilizes user interaction to generate more personalized recommendations.
- Can capture complex patterns in user data.

#### Disadvantages:
- Requires quite a lot of rating data.
- Cannot recommend new movies that do not yet have a rating.

## Evaluation

### Evaluasi Content-Based Filtering

### Content-Based Filtering Evaluation

The evaluation metric used for **Content Based Filtering** is **Recall@K**.

**Recall@K** is a metric that measures the proportion of relevant items in the top-K of all relevant items in the top-N recommendations.

The formula of Recall@K is:

Recall@K = $\displaystyle \frac{\text{relevant items in top-K}}{\text{relevant items in top-N}}$

Here is the Recall@K analysis for the **Content-Based Filtering** recommendation results.

Data for the trial
| # | title                          | genres                           |
|--:|:------------------------------:|:--------------------------------:|
| 0 | Sherlock Holmes (2010)         | [Mystery, Sci-Fi]                |

Recommendation results

| Title                                                     | Genres                      |
|-----------------------------------------------------------|-----------------------------|
| Andromeda Strain, The (1971)                              | [Mystery, Sci-Fi]            |
| Stalker (1979)                                            | [Drama, Mystery, Sci-Fi]     |
| Solaris (Solyaris) (1972)                                  | [Drama, Mystery, Sci-Fi]     |
| Sound of My Voice (2011)                                  | [Drama, Mystery, Sci-Fi]     |
| Fire in the Sky (1993)                                    | [Drama, Mystery, Sci-Fi]     |
| Big Empty, The (2003)                                     | [Comedy, Mystery, Sci-Fi]    |
| Stepford Wives, The (1975)                                | [Mystery, Sci-Fi, Thriller]  |
| District 9 (2009)                                         | [Mystery, Sci-Fi, Thriller]  |
| Seconds (1966)                                            | [Mystery, Sci-Fi, Thriller]  |
| Twelve Monkeys (a.k.a. 12 Monkeys) (1995)                 | [Mystery, Sci-Fi, Thriller]  |



As in the table, all movies have all three genres in the test data, namely **"Mystery, Sci-Fi"**. This makes the number of relevant items in top-N = 10. then it can also be concluded that the number of items in top-K will always be the same as K.

Then Recall@K for

- K = 5 &rarr; 5/10 \* 100% = 50%
- K = 8 &rarr; 8/10 \* 100% = 80%
- K = 10 &rarr; 10/10 \* 100% = 100%

It can be concluded that the recommendations given have a Recall@K of 100%.

### Collaborative Filtering Evaluation

The evaluation metric used for **Collaborative Filtering** is **Mean Absolute Error (MAE)**

MAE or Mean Absolute Error is applied by measuring the average of the absolute difference between the prediction and the original value (y_original - y_prediction).

The MAE formula is

MAE = $\displaystyle \sum\frac{|y_i - \hat{y}_i|}{n}$

Where:
MAE = Mean Absolute Error value
y = actual value
ŷ = predicted value
i = data sequence
n = number of data
Here is the MAE plot of the model

![model_plot](images/model_metrics_mae.png)
Figure 3.0 MAE plot of the model

It can be seen that this model has a relatively low MAE value of less than 20% and does not experience overfitting because there is no disparity between the train data and the test data of up to 10% so it is suitable for making predictions on new data.

## References

###### [1] J F Mohammad, B Al Faruq1, S Urolagin, "Movie Recommender System Using Content-based and Collaborative Filtering", 2022_ https://ieeexplore.ieee.org/document/9872515/authors#authors
