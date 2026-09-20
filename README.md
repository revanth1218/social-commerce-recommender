Social Commerce Post Feed Recommendation Engine

1. Project Overview

This project implements a personalized recommendation engine for a social commerce platform.

The main objective is to recommend relevant posts to users and rank them based on:

User-post semantic similarity

User interaction signals

Watch time

Post popularity

Content recency

The system uses pre-generated user and post embeddings from MongoDB and combines them with behavioral and content signals to produce a personalized feed.

The recommendation feed is dynamically re-ranked when the user interacts with recommended content.

2. Problem Statement

A social commerce platform may contain a large number of posts from different creators.

Showing posts randomly does not provide a personalized experience.

The recommendation engine should answer:

Which posts should be shown to a user?

In what order should they be shown?

Which content signals should influence the ranking?

How should user interactions affect future recommendations?

How can the system work when direct user interaction history is limited?

This project addresses these questions using an embedding-based hybrid recommendation approach.

3. Solution Approach

The system follows a hybrid recommendation approach.

The main recommendation signal is the similarity between:

User embedding

Post embedding

Additional ranking signals are then combined with the semantic similarity score.

The current ranking formula is:

Final Score =
    Similarity × 0.55
    + Engagement × 0.20
    + Watch Time × 0.10
    + Popularity × 0.10
    + Recency × 0.05

The weights are heuristic and can be tuned using offline evaluation or a trained ranking model in a production system.

4. System Architecture

                    MongoDB
                       |
          +------------+------------+
          |            |            |
      User Data     Post Data    Metrics
          |            |            |
      User          Post          Post
    Embedding     Embedding     Metrics
          |            |            |
          +------------+------------+
                       |
                       v
              Candidate Generation
                       |
                       v
             Cosine Similarity
                       |
                       v
              Ranking Features
          +------------+------------+
          |            |            |
      Engagement    Watch Time   Popularity
          |            |            |
          +------------+------------+
                       |
                    Recency
                       |
                       v
                Final Ranking
                       |
                       v
              Personalized Feed
                       |
                       v
             User Interaction
          +------------+------------+
          |            |            |
        Like         Save        Watch
          |            |            |
          +------------+------------+
                       |
                       v
             Local User Embedding
                       |
                       v
                  Re-ranking
                       |
                       v
              Next Recommendation

5. Technologies Used

Python

MongoDB

PyMongo

NumPy

scikit-learn

Cosine Similarity

MinMaxScaler

JSON

python-dotenv

6. MongoDB Collections Used

userdetails

Used to retrieve available users and basic user information.

Important fields:

userId

name

userName

email

status

userembeddings

Contains the pre-generated user representation.

Important field:

embedding

The current implementation uses the 512-dimensional user embedding.

posts

Contains the candidate posts.

Important fields:

postId

userId

caption

status

createdAt

category

content

Only active posts are considered for recommendation.

postembeddings

Contains the semantic representation of posts.

Important field:

embedding_fused

The current implementation uses the fused post embedding for similarity calculation.

postmetrics

Contains aggregated post-level engagement metrics.

Used signals include:

Likes

Comments

Shares

Bookmarks

Local JSON files

Because the provided MongoDB environment is treated as read-only, user-specific interaction data and updated user embeddings are stored locally.

Files:

user_embedding.json

local_interactions.json

MongoDB is not modified by the recommendation engine.

7. Recommendation Pipeline

Step 1: Select User

The system retrieves active users from the database and allows the user to select a user.

Step 2: Load User Embedding

The system first checks whether a locally updated embedding exists.

If available, the local embedding is used.

Otherwise, the original embedding is retrieved from MongoDB.

This allows the system to preserve personalization across local sessions.

Step 3: Fetch Active Posts

Only posts with:

status = active

are considered.

The selected user's own posts are excluded from the recommendation candidates.

Step 4: Fetch Post Embeddings

The system retrieves the corresponding post embeddings.

Posts without a valid embedding are excluded.

Step 5: Calculate Similarity

Cosine similarity is calculated between the user embedding and each post embedding.

User Embedding
       |
       | cosine similarity
       v
Post Embedding
       |
       v
Similarity Score

A higher similarity means the post is more aligned with the current user representation.

Step 6: Calculate Additional Signals

The system calculates:

Engagement score

Watch-time score

Popularity score

Recency score

These signals are normalized before being combined.

Step 7: Calculate Final Ranking Score

The final score combines all signals:

Final Score =
    0.55 × Similarity
  + 0.20 × Engagement
  + 0.10 × Watch Time
  + 0.10 × Popularity
  + 0.05 × Recency

Posts are sorted by the final score in descending order.

Step 8: Display Recommendation

The highest-ranked post is shown to the user.

The system also displays a short explanation describing why the post was recommended.

Example:

Why recommended:
  -> highly similar to your interests
  -> influenced by your recent interaction
  -> you spent time watching similar content

8. User Interaction Handling

The feed supports the following commands:

Enter       -> Finish current video and go next
l / like    -> Like current video
s / save    -> Save current video
b / both    -> Like + Save current video
n / no      -> Stop feed

Like

A like is treated as a positive preference signal.

Save

A save/bookmark is treated as a strong positive preference signal.

Like + Save

Both signals are combined, producing a stronger influence on the local user embedding.

Watch Time

Watch time provides an implicit preference signal.

Longer watch time indicates more engagement with the content.

Very short watch time without a like or save is treated as a weak-interest signal.

Skip

When a user quickly moves to the next video without liking or saving it, the system does not move the user embedding toward that post.

Therefore, a quick skip does not create a strong positive preference for that content.

9. Online Personalization

One important feature of the system is online personalization.

After a meaningful interaction, the local user embedding is updated.

The update follows:

New User Embedding =
    (1 - weight) × Old User Embedding
    + weight × Post Embedding

The resulting vector is normalized again.

Interaction Weight

The current implementation uses:

Base weight = 0.05

Watch time >= 10 seconds  -> +0.02
Watch time >= 30 seconds  -> +0.05
Like                     -> +0.10
Save                     -> +0.10

Maximum weight = 0.30

This prevents a single interaction from completely changing the user's preference profile.

10. Dynamic Re-ranking

The feed is not static.

Initial User Embedding
        |
        v
Rank Posts
        |
        v
Recommend Video 1
        |
        v
User Likes + Saves
        |
        v
Update Local User Embedding
        |
        v
Recalculate Similarities
        |
        v
Re-rank Remaining Posts
        |
        v
Recommend Video 2

This allows the recommendation feed to adapt during the current session.

11. Why Embeddings Are Used

Embeddings allow the system to compare content based on semantic similarity rather than relying only on exact keywords.

For example, posts related to:

bridal makeup
wedding beauty
bridal glam

may have similar semantic representations even when their captions use different words.

This makes embeddings useful as the primary content-relevance signal.

12. Cold Start Handling

A new user may have little or no direct interaction history.

To handle this case, the system can start with the existing user embedding from MongoDB.

The initial recommendations are therefore primarily driven by:

User Embedding
       +
Post Embedding
       +
Popularity
       +
Recency

As the user starts interacting with the feed, local behavioral signals become available and the user embedding is updated.

13. Read-Only Database Design

The MongoDB database is intentionally used in read-only mode.

The system reads:

Users

User embeddings

Posts

Post embeddings

Post metrics

The system does not modify MongoDB.

Instead, local files store:

user_embedding.json
local_interactions.json

This allows the prototype to demonstrate online personalization without modifying the shared assignment database.

14. Candidate Filtering

Before ranking, the system filters candidates.

A post is excluded if:

The post is not active

The post belongs to the selected user

The post has already been shown in the current session

A valid post embedding is unavailable

This reduces unnecessary ranking work and prevents repeated recommendations.

15. Example Recommendation Flow

Example:

Selected User:
Krishna

Initial Recommendation:

Post:
A touch of bridal glam, a whole lot of elegance.

Similarity:
0.8776

Final Score:
0.8742

The user then performs:

Like + Save

The local user embedding is updated.

The remaining candidates are then re-ranked.

This means the next recommendation can change based on the user's latest interaction.

16. Recommendation Reasoning

The recommendation system can explain recommendations using its ranking signals.

Examples:

Highly similar to your interest profile.

Matches your interests and was influenced by your recent interaction.

You spent more time watching similar content.

This makes the recommendation process easier to understand and debug.

17. Scalability Considerations

The current prototype ranks the available candidate posts directly.

For a production-scale platform with millions of posts, calculating similarity against every post would be expensive.

A scalable architecture could use:

User Embedding
      |
      v
Approximate Nearest Neighbor Search
      |
      v
Top Candidate Set
      |
      v
Feature Generation
      |
      v
Ranking Model
      |
      v
Personalized Feed

Possible production improvements include:

Vector database / vector index

Approximate nearest-neighbor retrieval

Candidate generation service

Feature store

Learning-to-rank model

Recommendation cache

Batch embedding generation

Real-time interaction processing

18. Limitations

The current implementation is a prototype and has some limitations.

Heuristic Ranking

The ranking weights are manually selected.

A production system could learn these weights using historical interaction data.

Local Interaction Storage

User interactions are stored locally because MongoDB is read-only in this prototype.

A production system would persist interaction events in a database or event-streaming system.

Sparse Behavioral Data

The available test data contains limited user interaction history.

This can reduce the reliability of behavior-based personalization.

No Trained Ranking Model

The current ranking layer is heuristic rather than a trained learning-to-rank model.

No Video Duration Metadata

The available post documents do not provide a reliable video duration field, so the current implementation uses raw watch time instead of a completion-rate feature.

19. Future Improvements

Possible improvements include:

Train a learning-to-rank model.

Add explicit negative/skip signals.

Add creator diversity to avoid showing too many posts from the same creator.

Add hashtag/content-tag based candidate retrieval.

Add collaborative filtering.

Add social graph signals such as followed creators.

Add product recommendation as a second recommendation pipeline.

Add offline evaluation using Precision@K, Recall@K and NDCG@K.

Add approximate nearest-neighbor vector search for large-scale retrieval.

Store interaction events in a production database or event stream.

Add recommendation caching for low-latency feed generation.

20. How to Run

1. Install dependencies

pip install pymongo python-dotenv numpy scikit-learn

2. Configure environment variables

Create a .env file:

MONGODB_URI=your_mongodb_connection_string

Do not commit .env to GitHub.

Add it to .gitignore:

.env
user_embedding.json
local_interactions.json
__pycache__/

3. Run the recommendation engine

python recN.py

4. Select a user

The program displays available users.

Enter the required user number.

5. Interact with the feed

Enter -> Next
l     -> Like
s     -> Save
b     -> Like + Save
n     -> Stop

21. Project Structure

social-commerce-recommender/
│
├── recN.py
├── .env
├── .gitignore
├── user_embedding.json
├── local_interactions.json
└── README.md

22. Conclusion

This project demonstrates an embedding-based hybrid recommendation engine for a social commerce post feed.

The system combines semantic similarity with behavioral and business signals to rank posts.

The main personalization mechanism is online local embedding adaptation:

User Interaction
       |
       v
Update Local User Embedding
       |
       v
Recalculate Similarity
       |
       v
Re-rank Remaining Posts
       |
       v
Personalized Feed

The prototype provides a foundation that can be extended with a trained ranking model, scalable vector retrieval, richer behavioral signals, social signals, and product recommendations.