Social Commerce Recommendation Engine

A prototype recommendation engine for a social-commerce post feed.

The system combines pre-generated user/post embeddings with behavioral, social, popularity, and recency signals to rank relevant posts for a selected user.

1. Project Objective

The goal is to answer three questions:

Which posts should be shown to a user?

In what order should those posts be shown?

Which user, content, behavioral, and social signals should influence the ranking?

This project focuses on the Post Feed Recommendation Engine.

2. High-Level Architecture

                 MongoDB
                    |
        +-----------+-----------+
        |                       |
   User Data                 Post Data
        |                       |
 User Embedding          Post Embeddings
        |                       |
        +----------+------------+
                   |
            Candidate Posts
                   |
        +----------+-----------+
        |                      |
 Content Similarity      Ranking Signals
 (Cosine Similarity)     - Engagement
                         - Watch Time
                         - Popularity
                         - Recency
                         - Follow
                         - Creator Affinity
                   |
             Normalization
                   |
             Weighted Score
                   |
             Ranked Feed
                   |
             User Interaction
          /        |        \
       Like      Save      Follow
          \        |        /
           Local State Update
                   |
          User Embedding Update
                   |
            Re-ranking

3. Technology Stack

Python

MongoDB / MongoDB Atlas

PyMongo

NumPy

Scikit-learn

python-dotenv

Cosine Similarity

JSON for local mutable state

4. MongoDB Collections Used

The project reads recommendation-related information from MongoDB.

Main collections

users / user details collection

userembeddings

posts

postembeddings

postengagements

postmetrics

views

follows

user_creator_affinity

Other assignment collections such as products, carts, and orders are available in the database, but the current implementation does not use them for the post-feed ranking.

MongoDB is kept READ-ONLY by this prototype. User interactions and updated embeddings are stored locally.

5. Recommendation Pipeline

The recommendation flow is:

Select User
    ↓
Load User Embedding
    ↓
Fetch Active Posts
    ↓
Fetch Post Embeddings
    ↓
Calculate Cosine Similarity
    ↓
Calculate Ranking Signals
    ↓
Normalize Signals
    ↓
Apply Weighted Ranking Formula
    ↓
Sort Posts
    ↓
Display Recommendation
    ↓
Capture User Interaction
    ↓
Update Local User Embedding
    ↓
Re-rank Remaining Posts

6. Candidate Filtering

Before ranking, the system filters candidates.

A post is considered only when:

The post is active.

The post has a valid embedding.

The post is not created by the selected user.

The post has not already been shown in the current session.

This reduces unnecessary ranking work and prevents repeated posts during the same feed session.

7. Content Similarity

The system uses pre-generated embeddings.

User embedding: 512 dimensions

Post embedding: 512 dimensions

Cosine similarity is used to measure how closely a user's embedding matches a post's embedding.

Conceptually:

User Embedding
      +
Post Embedding
      ↓
Cosine Similarity
      ↓
Content Relevance

A higher similarity means the post is more aligned with the representation of the user's interests.

The embeddings are pre-generated in the provided data. This prototype does not train the embedding model.

8. Ranking Signals

The final recommendation score combines multiple signals.

8.1 Content Similarity

Measures semantic relevance between the user and post embeddings.

Weight: 45%

similarity × 0.45

8.2 Engagement

The local interaction history considers:

Like

Bookmark / Save

These interactions indicate stronger user interest than a simple view.

Weight: 20%

engagement × 0.20

8.3 Watch Time

Watch time represents how long the user spends viewing a post/video.

The current watch-time score is:

watch_score = min(watch_time / 30, 1)

Examples:

5 sec   → 0.167
10 sec  → 0.333
15 sec  → 0.500
30 sec  → 1.000
40 sec  → 1.000

Weight: 10%

watch_score × 0.10

8.4 Popularity

Popularity is calculated using post-level engagement metrics:

popularity =
    likes
    + comments × 2
    + shares × 3
    + bookmarks × 2

The weighted components reflect the prototype's heuristic assumption that different engagement types can have different strengths.

Weight: 5%

popularity × 0.05

8.5 Recency

Newer posts receive a higher recency score.

The current prototype uses:

recency = 1 / (1 + age_in_days)

So, as a post becomes older, its recency contribution decreases.

Weight: 5%

recency × 0.05

9. Social / Creator Signals

The project uses social and creator-level signals.

This is not a full graph-based recommendation system.

9.1 Follow Signal — Implemented

The system checks whether the selected user follows the creator of a post.

Example:

User
 ↓
follows
 ↓
Creator
 ↓
Creator's Post

A post from a followed creator receives additional social relevance.

The system reads follow relationships from MongoDB and also maintains newly created local follows.

9.2 Creator Affinity — Implemented

The system also uses the existing user_creator_affinity collection.

Conceptually:

User → Creator → Affinity Score

This provides an additional creator-level relationship signal.

The affinity value is bounded before being used in the ranking calculation.

9.3 Social Signal Formula

The current social signal combines:

social_signal =
    0.60 × follow_signal
    + 0.40 × affinity_signal

Social signal weight in final ranking: 15%

social_signal × 0.15

10. Final Ranking Formula

The current final ranking formula is:

final_score =
    0.45 × similarity
  + 0.20 × engagement
  + 0.10 × watch_score
  + 0.05 × popularity
  + 0.05 × recency
  + 0.15 × social_signal

The weights are manually selected heuristic weights.

They are not learned from a trained ranking model.

11. Why a Heuristic Ranking Model?

The assignment allows a hand-tuned heuristic ranking approach.

For this prototype, a heuristic model was selected because:

The available interaction data is relatively sparse.

The approach is easy to understand and debug.

Each signal's contribution is interpretable.

It can be implemented quickly for a prototype.

The ranking logic can later be replaced by a learned ranking model.

For example:

Content relevance → 45%
User engagement   → 20%
Watch behavior    → 10%
Popularity        → 5%
Recency           → 5%
Social signals    → 15%

With more historical interaction and impression data, these weights could be learned rather than manually selected.

12. User Interaction

The interactive feed supports:

Enter → Continue
l     → Like
s     → Save / Bookmark
b     → Like + Save
f     → Follow creator
n     → Stop

The system records relevant interaction data locally.

13. View and Watch-Time Tracking

When a video/post starts, the system can count a view.

Watch time is accumulated locally for the user-post interaction.

For example:

User watches:
Post A → 5 sec
Post A → another 8 sec

Total watch time:
Post A → 13 sec

This interaction can influence the user's local representation and subsequent ranking.

14. User Embedding Update

The prototype supports an online-style local user embedding update.

A base weight is assigned to the current user representation.

Additional interaction signals can increase the weight of the interacted post embedding:

Completion ≥ 80% → +0.10
Completion ≥ 50% → +0.05
Like             → +0.10
Save             → +0.10
Follow           → +0.10

The interaction weight is capped at:

0.30

The updated representation is normalized before being saved locally.

This allows the recommendation profile to change based on new interactions without modifying the MongoDB user embedding.

15. Dynamic Re-ranking

The feed is not ranked only once.

After a user interacts with a post:

User Interaction
      ↓
Local Interaction Update
      ↓
User Embedding Update
      ↓
Recalculate Similarity
      ↓
Re-rank Remaining Posts

Therefore, later recommendations can change based on what the user has interacted with during the current session.

16. Local Storage

Because the MongoDB environment is treated as READ-ONLY, mutable recommendation state is stored locally.

Local files

user_embedding.json
local_interactions.json
local_follows.json

Purpose

user_embedding.json

Stores the locally updated user embedding.

local_interactions.json

Stores local interaction information such as:

Watch time

Likes

Saves

Views

local_follows.json

Stores follow relationships created during the local recommendation session.

These files should not be committed to GitHub.

17. Cold-Start Handling

Cold start is handled using the available user embedding and content similarity.

If a user has little or no direct interaction history:

User Embedding
      ↓
Compare with Post Embeddings
      ↓
Cosine Similarity
      ↓
Generate Initial Recommendations

As the user starts interacting with posts, local interaction signals and the updated embedding can make the feed more personalized.

18. Recommendation Explanation

For each recommendation, the system can provide a short reason.

Possible reasons include:

High content similarity

Previous interaction with similar content

Creator is followed

Creator affinity exists

Recent post

Example:

Why this post is recommended:
High content similarity + followed creator + recent post

This improves transparency during the prototype demonstration.

19. Implemented Features

Feature

Status

User selection

Implemented

Active post filtering

Implemented

Pre-generated embeddings

Implemented

Cosine similarity

Implemented

Like signal

Implemented

Save / Bookmark signal

Implemented

Watch-time signal

Implemented

View tracking

Implemented

Popularity signal

Implemented

Recency signal

Implemented

Follow signal

Implemented

Creator affinity

Implemented

Local user embedding update

Implemented

Dynamic re-ranking

Implemented

Recommendation explanation

Implemented

Local interaction storage

Implemented

Read-only MongoDB architecture

Implemented

Full trained ranking model

Not implemented

Collaborative filtering

Not implemented

Product recommendation

Not implemented

Hashtag/content-tag recommendation

Not implemented

Full graph-based recommendation

Not implemented

20. Future Improvements

20.1 Learned Ranking Model

With enough historical data, the manually selected weights can be replaced with a trained ranking model.

Possible training data:

User
Post
Impression
Position
Watch Time
Like
Save
Share
Follow
Final Interaction

A ranking model could learn which signals are most predictive of engagement.

20.2 Creator Diversity

The current system ranks posts independently.

A future improvement is to avoid showing too many posts from the same creator.

Example:

Current:

Creator A
Creator A
Creator A
Creator B

Future:

Creator A
Creator B
Creator A
Creator C

This can make the feed more diverse.

20.3 Advanced Social Graph Features

The current implementation already uses:

Follow relationship

Creator affinity

Future improvements could include:

Friends-of-friends / second-degree connections

More advanced social interaction signals

Graph-based user and creator representations

Graph embeddings

20.4 Better Watch-Time Modeling

The current watch-time score uses a simple 30-second cap.

A future system could consider:

Video duration

Completion rate

Re-watches

Skip behavior

Long-term viewing patterns

The current ranking does not use completion rate as a ranking signal when reliable video duration is unavailable.

20.5 Offline Evaluation

Future evaluation can use historical impression and interaction data.

Possible metrics:

Precision@K
Recall@K
NDCG@K
CTR
Watch-time
Save rate
Like rate

This would allow different ranking strategies to be compared using historical data.

20.6 Scalable Candidate Retrieval

For a large number of posts, comparing every user embedding against every post embedding would become expensive.

A production system could use:

Approximate Nearest Neighbor (ANN) search

Vector databases

Embedding indexes

Candidate generation service

This would reduce retrieval latency.

20.7 Real-Time Event Processing

Instead of local JSON state, a production system could process events using a streaming architecture.

Example:

User Interaction
      ↓
Event Stream
      ↓
Feature Update
      ↓
User Profile Update
      ↓
Recommendation Service

21. Current Limitations

This is a prototype, not a production recommendation platform.

Current limitations include:

Manually selected ranking weights

Pre-generated embeddings

Sparse interaction data

Local JSON state for mutable interactions

No trained ranking model

No collaborative filtering

No full graph-based recommendation

No ANN/vector index

No real-time event streaming

No formal offline evaluation pipeline

Product recommendation is not implemented

Hashtag/content-tag recommendation is not implemented

22. Security

Sensitive configuration such as the MongoDB connection string is stored in .env.

Example:

MONGODB_URI=<your-mongodb-uri>

The .env file should be excluded from Git using .gitignore.

Local user interaction and embedding files should also remain outside version control.

23. Project Structure

social-commerce-recommender/
│
├── recommendation.py
├── README.md
├── .env
├── .gitignore
│
├── user_embedding.json
├── local_interactions.json
└── local_follows.json

Runtime JSON files should be ignored by Git.

24. Installation

Install the required Python packages:

pip install pymongo python-dotenv numpy scikit-learn

25. Environment Configuration

Create a .env file:

MONGODB_URI=<your-mongodb-uri>

Do not commit the actual MongoDB URI to GitHub.

26. Run the Project

python recommendation.py

The application:

Connects to MongoDB.

Loads active users.

Allows the user to select a recommendation target.

Loads the user's embedding.

Retrieves active posts and embeddings.

Calculates ranking scores.

Displays recommended posts.

Accepts user interactions.

Updates local recommendation state.

Re-ranks remaining posts.

27. Example Recommendation Flow

User selects Krishna
        ↓
Load Krishna's 512-D embedding
        ↓
Fetch active posts
        ↓
Compare user embedding with post embeddings
        ↓
Calculate:
  - Similarity
  - Engagement
  - Watch time
  - Popularity
  - Recency
  - Follow
  - Creator affinity
        ↓
Normalize signals
        ↓
Calculate final score
        ↓
Sort by final score
        ↓
Show recommendation
        ↓
User likes / saves / watches / follows
        ↓
Update local state
        ↓
Update user representation
        ↓
Re-rank remaining posts

28. Important Design Decision

The current system intentionally separates:

Candidate Retrieval

Find potentially relevant posts

from:

Ranking

Calculate final score
and order the candidates

This separation makes it easier to replace the current heuristic ranker with a trained ranking model later.

29. Production Architecture — Future Direction

A production version could look like:

                  User Events
                      ↓
              Event Streaming
                      ↓
              Feature Store
                      ↓
       +--------------+--------------+
       |                             |
 User Profile Service        Post/Content Service
       |                             |
       +--------------+--------------+
                      |
              Candidate Retrieval
                Vector / ANN
                      |
                 Ranker Model
                      |
              Business Rules
                      |
                Final Feed
                      |
                    User

The current prototype represents the recommendation and ranking logic that could form part of this larger architecture.

30. Summary

This project implements a hybrid post-feed recommendation prototype using:

Pre-generated user and post embeddings

Cosine similarity

User engagement

Watch time

Popularity

Recency

Follow relationships

Creator affinity

Local user embedding updates

Dynamic re-ranking

Cold-start handling

Recommendation explanations

The current ranking system is heuristic and interpretable, not a trained ML ranking model.

With more data and production infrastructure, the system can be extended with learned ranking, creator diversity, advanced social graph features, offline evaluation, ANN retrieval, and real-time event processing.