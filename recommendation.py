# Load the libraries used by the recommender
import os
import json
import time
import numpy as np

# Use UTC timestamps for local interaction records
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
# CONFIGURATION
# Read configuration from the local .env file
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")
# MONGODB CONNECTION
client = MongoClient(MONGO_URI)

client.admin.command("ping")

print("\nMongoDB connection successful!")
print("MongoDB is used in READ-ONLY mode.\n")

# The assignment database is read-only in this prototype
db = client["test"]
# LOCAL FILES
USER_EMBEDDING_FILE = "user_embedding.json"
LOCAL_INTERACTIONS_FILE = "local_interactions.json"
LOCAL_FOLLOWS_FILE = "local_follows.json"
# HELPER FUNCTIONS
def utc_now():
    return datetime.now(timezone.utc).isoformat()


# Convert numeric values safely
def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        return float(value)

    except (ValueError, TypeError):

        return default


# Keep embeddings normalized before similarity calculations
def normalize_embedding(embedding):

    embedding = np.array(
        embedding,
        dtype=float
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        return embedding

    return embedding / norm
# VIDEO DURATION
def get_video_duration(post):

    """
    Reads video duration.

    Currently checking:
        1. videoDuration
        2. duration

    If your MongoDB stores duration in another field,
    add that field here.
    """

    possible_fields = [
        "videoDuration",
        "duration"
    ]

    for field in possible_fields:

        value = safe_float(
            post.get(field, 0)
        )

        if value > 0:

            return value

    return 0.0
# USER EMBEDDING - LOCAL
def save_user_embedding(
    user_id,
    username,
    embedding
):

    data = {

        "userId": user_id,

        "username": username,

        "embedding": embedding.tolist(),

        "embeddingSize": len(embedding),

        "updatedAt": utc_now()
    }

    with open(
        # Keep mutable user state in local JSON files
        USER_EMBEDDING_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


# Load a previously learned local embedding
def load_local_user_embedding(
    user_id
):

    if not os.path.exists(
        # Keep mutable user state in local JSON files
        USER_EMBEDDING_FILE
    ):

        return None

    try:

        with open(
            # Keep mutable user state in local JSON files
            USER_EMBEDDING_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        if data.get("userId") != user_id:

            return None


        embedding = data.get(
            "embedding"
        )


        if not embedding:

            return None


        return normalize_embedding(
            embedding
        )


    except Exception as e:

        print(
            "Could not load local embedding:",
            e
        )

        return None
# UPDATE USER EMBEDDING
def update_local_user_embedding(

    user_embedding,

    post_embedding,

    watch_time,

    video_duration,

    liked,

    bookmarked,

    followed=False

):

    post_embedding = normalize_embedding(
        post_embedding
    )
    # Completion rate
    if video_duration > 0:

        completion_rate = min(
            watch_time / video_duration,
            1.0
        )

    else:

        completion_rate = 0.0
    # Base weight
    weight = 0.05
    # Watch completion influence
    if completion_rate >= 0.80:

        weight += 0.10

    elif completion_rate >= 0.50:

        weight += 0.05
    # Like influence
    if liked:

        weight += 0.10
    # Save influence
    if bookmarked:

        weight += 0.10
    # Follow influence
    if followed:
        weight += 0.10

    # Maximum influence
    weight = min(
        weight,
        0.30
    )
    # Embedding update
    new_embedding = (

        (1 - weight)
        * user_embedding

        +

        weight
        * post_embedding
    )


    new_embedding = normalize_embedding(
        new_embedding
    )


    return (
        new_embedding,
        weight,
        completion_rate
    )
# LOCAL INTERACTIONS
def load_local_interactions():

    if not os.path.exists(
        LOCAL_INTERACTIONS_FILE
    ):

        return []


    try:

        with open(
            LOCAL_INTERACTIONS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        if isinstance(data, list):

            return data


        return []


    except Exception as e:

        print(
            "Could not load local interactions:",
            e
        )

        return []


# Save local interaction history
def save_local_interactions(
    interactions
):

    with open(
        LOCAL_INTERACTIONS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            interactions,
            f,
            indent=4
        )


# Load creator follows saved during the prototype
def load_local_follows():
    if not os.path.exists(LOCAL_FOLLOWS_FILE):
        return []

    try:
        with open(LOCAL_FOLLOWS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except Exception as e:
        print("Could not load local follows:", e)
        return []


# Save local creator follows
def save_local_follows(follows):
    with open(LOCAL_FOLLOWS_FILE, "w", encoding="utf-8") as f:
        json.dump(follows, f, indent=4)


# Check whether the user follows a creator
def is_creator_followed(local_follows, current_user_id, creator_id):
    return any(
        item.get("userId") == current_user_id
        and item.get("creatorId") == creator_id
        for item in local_follows
    )


# Add a creator to the local follow list
def add_local_follow(local_follows, current_user_id, creator_id):
    if not is_creator_followed(local_follows, current_user_id, creator_id):
        local_follows.append({
            "userId": current_user_id,
            "creatorId": creator_id,
            "followedAt": utc_now()
        })
# FIND EXISTING INTERACTION
def get_interaction(

    interactions,

    user_id,

    post_id

):

    for interaction in interactions:

        if (

            interaction.get("userId")
            == user_id

            and

            interaction.get("postId")
            == post_id

        ):

            return interaction


    return None
# UPDATE INTERACTION
def update_local_interaction(

    interactions,

    user_id,

    post_id,

    watch_time_delta,

    video_duration,

    liked=False,

    bookmarked=False,

    increment_view=False,

    caption=""

):

    """
    Maintains ONE record per user + post.

    watch_time_delta:
        Only the NEW watch-time segment.

    increment_view:
        True ONLY when video starts.

    Therefore:

        l -> no new view
        s -> no new view
        b -> no new view
        Enter -> no new view
        Video start -> +1 view
    """


    existing = get_interaction(

        interactions,

        user_id,

        post_id

    )
    # EXISTING RECORD
    if existing:

        old_watch_time = safe_float(

            existing.get(
                "watchTime",
                0
            )

        )


        total_watch_time = (

            old_watch_time
            +
            watch_time_delta

        )


        existing["watchTime"] = (
            total_watch_time
        )
        # Preserve like
        existing["liked"] = (

            existing.get(
                "liked",
                False
            )

            or

            liked
        )
        # Preserve save
        existing["bookmarked"] = (

            existing.get(
                "bookmarked",
                False
            )

            or

            bookmarked
        )
        # View count
        # Count only one view for each user + post.
        if increment_view and int(existing.get("views", 0)) == 0:

            existing["views"] = 1
        # Duration
        if video_duration > 0:

            existing[
                "videoDuration"
            ] = video_duration


            existing[
                "completionRate"
            ] = min(

                total_watch_time
                /
                video_duration,

                1.0

            )

        else:

            existing[
                "completionRate"
            ] = 0.0


        existing[
            "lastInteractionAt"
        ] = utc_now()


        return existing
    # NEW RECORD
    if video_duration > 0:

        completion_rate = min(

            watch_time_delta
            /
            video_duration,

            1.0

        )

    else:

        completion_rate = 0.0


    new_interaction = {

        "userId":
            user_id,

        "postId":
            post_id,

        "watchTime":
            watch_time_delta,

        "videoDuration":
            video_duration,

        "completionRate":
            completion_rate,

        "liked":
            liked,

        "bookmarked":
            bookmarked,

        "views":
            1 if increment_view else 0,

        "lastInteractionAt":
            utc_now(),

        "caption": caption
    }


    interactions.append(
        new_interaction
    )


    return new_interaction
# FETCH USERS
print("Fetching users...\n")


# Fetch active users for the demo selector
users = list(

    db["userdetails"].find(

        {
            "status": "active"
        },

        {
            "userId": 1,
            "name": 1,
            "userName": 1,
            "email": 1
        }

    )

)


if not users:

    print(
        "No active users found."
    )

    client.close()

    exit()
# DISPLAY USERS
print("=" * 80)
print("AVAILABLE USERS")
print("=" * 80)


for i, user in enumerate(
    users,
    1
):

    print(
        f"{i}. "
        f"{user.get('name', 'N/A')}"
    )

    print(
        f"   Username : "
        f"{user.get('userName', 'N/A')}"
    )

    print(
        f"   User ID  : "
        f"{user.get('userId', 'N/A')}"
    )

    print(
        f"   Email    : "
        f"{user.get('email', 'N/A')}"
    )

    print(
        "-" * 80
    )
# SELECT USER
while True:

    try:

        selected_number = int(
            input(
                "\nSelect user number: "
            )
        )


        if (

            1
            <=
            selected_number
            <=
            len(users)

        ):

            break


        print(
            "Invalid user number."
        )


    except ValueError:

        print(
            "Please enter a number."
        )


# Get the selected user
selected_user = users[
    selected_number - 1
]


user_id = selected_user.get(
    "userId"
)

username = selected_user.get(
    "userName"
)

user_name = selected_user.get(
    "name"
)

email = selected_user.get(
    "email"
)
# SELECTED USER
print("\n")

print("=" * 80)
print("SELECTED USER")
print("=" * 80)

print(
    f"Name      : {user_name}"
)

print(
    f"Username  : {username}"
)

print(
    f"User ID   : {user_id}"
)

print(
    f"Email     : {email}"
)
# USER EMBEDDING
print(
    "\nFetching user embedding..."
)


local_embedding = (
    load_local_user_embedding(
        user_id
    )
)


if local_embedding is not None:

    user_embedding = (
        local_embedding
    )

    print(
        "\nExisting local user embedding loaded."
    )


else:

    embedding_document = (
        db["userembeddings"].find_one(

            {
                "userId":
                    user_id
            },

            {
                "embedding":
                    1
            }

        )
    )


    if not embedding_document:

        print(
            "User embedding not found."
        )

        client.close()

        exit()


    original_embedding = (
        embedding_document.get(
            "embedding"
        )
    )


    if not original_embedding:

        print(
            "User embedding is empty."
        )

        client.close()

        exit()


    user_embedding = (
        normalize_embedding(
            original_embedding
        )
    )


    save_user_embedding(

        user_id,

        username,

        user_embedding

    )


    print(
        "\nUser embedding copied to:"
        f" {USER_EMBEDDING_FILE}"
    )


print(
    f"Local user embedding size: "
    f"{len(user_embedding)}"
)
# LOAD LOCAL INTERACTIONS
local_interactions = (
    load_local_interactions()
)

local_follows = load_local_follows()
# FETCH ACTIVE POSTS
print(
    "\nFetching active posts..."
)


# Fetch active posts used as recommendation candidates
posts = list(

    db["posts"].find(

        {
            "status": "active"
        },

        {
            "postId": 1,
            "userId": 1,
            "caption": 1,
            "createdAt": 1,
            "category": 1,
            "duration": 1,
            "videoDuration": 1
        }

    )

)


print(
    f"Active posts: {len(posts)}"
)


if not posts:

    print(
        "No active posts found."
    )

    client.close()

    exit()
# POST EMBEDDINGS
print(
    "Fetching post embeddings..."
)


# Fetch the pre-generated post embeddings
post_embedding_documents = list(

    db["postembeddings"].find(

        {},

        {
            "postId": 1,
            "embedding_fused": 1
        }

    )

)


post_embeddings = {}


for document in (
    post_embedding_documents
):

    post_id = document.get(
        "postId"
    )

    embedding = document.get(
        "embedding_fused"
    )


    if (
        post_id
        and
        embedding
    ):

        post_embeddings[
            post_id
        ] = normalize_embedding(
            embedding
        )
# POST METRICS
post_metric_documents = list(

    db["postmetrics"].find(

        {},

        {
            "postId": 1,
            "totalLikes": 1,
            "totalComments": 1,
            "totalShares": 1,
            "totalBookmarks": 1
        }

    )

)


post_metrics = {}


for document in (
    post_metric_documents
):

    post_id = document.get(
        "postId"
    )


    if not post_id:

        continue


    post_metrics[
        post_id
    ] = {

        "likes":
            safe_float(
                document.get(
                    "totalLikes",
                    0
                )
            ),

        "comments":
            safe_float(
                document.get(
                    "totalComments",
                    0
                )
            ),

        "shares":
            safe_float(
                document.get(
                    "totalShares",
                    0
                )
            ),

        "bookmarks":
            safe_float(
                document.get(
                    "totalBookmarks",
                    0
                )
            )
    }
# RECENCY
def get_age_days(
    created_at
):

    if created_at is None:

        return 9999.0


    try:

        if isinstance(
            created_at,
            datetime
        ):

            if created_at.tzinfo is None:

                created_at = (
                    created_at.replace(
                        tzinfo=timezone.utc
                    )
                )


            now = datetime.now(
                timezone.utc
            )


            return max(

                0.0,

                (

                    now
                    -
                    created_at

                ).total_seconds()
                /
                86400

            )


        if isinstance(
            created_at,
            str
        ):

            created_at = (
                created_at.replace(
                    "Z",
                    "+00:00"
                )
            )


            created_at = (
                datetime.fromisoformat(
                    created_at
                )
            )


            if created_at.tzinfo is None:

                created_at = (
                    created_at.replace(
                        tzinfo=timezone.utc
                    )
                )


            now = datetime.now(
                timezone.utc
            )


            return max(

                0.0,

                (

                    now
                    -
                    created_at

                ).total_seconds()
                /
                86400

            )


    except Exception:

        pass


    return 9999.0
# NORMALIZATION
def normalize_values(
    values
):

    values = np.array(
        values,
        dtype=float
    ).reshape(
        -1,
        1
    )


    if len(values) <= 1:

        return np.zeros(
            len(values)
        )


    scaler = MinMaxScaler()


    return scaler.fit_transform(
        values
    ).flatten()
# SOCIAL SIGNALS: FOLLOWS + CREATOR AFFINITY
db_followed_creators = set()

for follow in db["follows"].find(
    {"followerId": user_id},
    {"followeeId": 1}
):
    followee_id = follow.get("followeeId")
    if followee_id:
        # Load creators already followed in MongoDB
        db_followed_creators.add(followee_id)

# Load existing user-to-creator affinity scores
creator_affinity = {}

for affinity in db["user_creator_affinity"].find(
    {"parentUserId": user_id},
    {"user_id": 1, "affinity_score": 1}
):
    creator_id = affinity.get("user_id")
    score = safe_float(affinity.get("affinity_score", 0))

    if creator_id:
        creator_affinity[creator_id] = score

print(f"Followed creators: {len(db_followed_creators)}")
print(f"Creator affinity records: {len(creator_affinity)}")
# RANK POSTS
def calculate_ranked_posts(

    current_user_embedding,

    excluded_post_ids

):

    candidates = []


    raw_popularity = []

    raw_recency = []

    raw_engagement = []

    raw_watch = []

    # Completion rate is not used because video duration is
    # unavailable/reliable in the provided posts collection.


    for post in posts:

        post_id = post.get(
            "postId"
        )

        creator_id = post.get(
            "userId"
        )
        # Do not recommend own post
        if creator_id == user_id:

            continue
        # Do not show already seen post
        if post_id in excluded_post_ids:

            continue
        # Need embedding
        if post_id not in post_embeddings:

            continue


        post_embedding = (
            post_embeddings[
                post_id
            ]
        )
        # Similarity
        similarity = float(

            cosine_similarity(

                current_user_embedding.reshape(
                    1,
                    -1
                ),

                post_embedding.reshape(
                    1,
                    -1
                )

            )[0][0]

        )
        # Popularity
        metrics = (
            post_metrics.get(
                post_id,
                {}
            )
        )


        likes = safe_float(
            metrics.get(
                "likes",
                0
            )
        )

        comments = safe_float(
            metrics.get(
                "comments",
                0
            )
        )

        shares = safe_float(
            metrics.get(
                "shares",
                0
            )
        )

        bookmarks = safe_float(
            metrics.get(
                "bookmarks",
                0
            )
        )


        popularity = (

            likes

            +

            comments * 2

            +

            shares * 3

            +

            bookmarks * 2

        )
        # Recency
        age_days = get_age_days(

            post.get(
                "createdAt"
            )

        )


        recency = (
            1
            /
            (
                1
                +
                age_days
            )
        )
        # Local interaction
        interaction = (
            get_interaction(

                local_interactions,

                user_id,

                post_id

            )
        )


        liked = 0

        bookmarked = 0

        watch_time = 0

        # Duration is unavailable in the provided post metadata.
        completion_rate = 0


        if interaction:

            liked = int(

                interaction.get(
                    "liked",
                    False
                )

            )


            bookmarked = int(

                interaction.get(
                    "bookmarked",
                    False
                )

            )


            watch_time = safe_float(

                interaction.get(
                    "watchTime",
                    0
                )

            )


            completion_rate = safe_float(

                interaction.get(
                    "completionRate",
                    0
                )

            )
        # Engagement
        engagement_score = (

            liked * 3

            +

            bookmarked * 4

            +

            min(
                watch_time / 30,
                1
            ) * 3

        )


        candidates.append({

            "post":
                post,

            "similarity":
                similarity,

            "popularity":
                popularity,

            "recency":
                recency,

            "engagement":
                engagement_score,

            "watchTime":
                watch_time,

            "completionRate":
                completion_rate,

            "creatorFollowed":
                (
                    creator_id in db_followed_creators
                    or
                    is_creator_followed(
                        local_follows,
                        user_id,
                        creator_id
                    )
                ),

            "creatorAffinity":
                creator_affinity.get(
                    creator_id,
                    0.0
                )
        })


        raw_popularity.append(
            popularity
        )

        raw_recency.append(
            recency
        )

        raw_engagement.append(
            engagement_score
        )

        raw_watch.append(
            watch_time
        )

        # Completion rate intentionally omitted from ranking.


    if not candidates:

        return []
    # Normalize
    popularity_norm = normalize_values(
        raw_popularity
    )

    recency_norm = normalize_values(
        raw_recency
    )

    engagement_norm = normalize_values(
        raw_engagement
    )

    watch_norm = normalize_values(
        raw_watch
    )

    # No completion normalization because duration is unavailable.
    # Final score
    for i, item in enumerate(
        candidates
    ):

        similarity = item[
            "similarity"
        ]


        creator_followed = item["creatorFollowed"]

        affinity_raw = item["creatorAffinity"]

        # Convert affinity to a bounded 0..1 signal.
        # Existing affinity values are not assumed to have a fixed maximum.
        affinity_norm = min(max(affinity_raw / 2.0, 0.0), 1.0)

        social_signal = (
            0.60 * float(creator_followed)
            +
            0.40 * affinity_norm
        )

        final_score = (

            similarity * 0.45

            +

            engagement_norm[i]
            * 0.20

            +

            watch_norm[i]
            * 0.10

            +

            popularity_norm[i]
            * 0.05

            +

            recency_norm[i]
            * 0.05

            +

            social_signal
            * 0.15

        )


        item[
            "finalScore"
        ] = final_score
    # Sort
    candidates.sort(

        key=lambda x:
            x["finalScore"],

        reverse=True

    )


    return candidates
# RECOMMENDATION REASON
def get_recommendation_reason(
    item
):

    reasons = []


    similarity = item[
        "similarity"
    ]

    recency = item[
        "recency"
    ]

    engagement = item[
        "engagement"
    ]

    creator_followed = item.get(
        "creatorFollowed",
        False
    )

    # Load existing user-to-creator affinity scores
    creator_affinity = safe_float(
        item.get(
            "creatorAffinity",
            0
        )
    )


    if similarity >= 0.70:

        reasons.append(
            "highly similar to your interests"
        )

    elif similarity >= 0.50:

        reasons.append(
            "matches your interests"
        )


    if engagement > 0:

        reasons.append(
            "your previous interactions indicate interest"
        )


    if creator_followed:
        reasons.append(
            "you follow this creator"
        )

    if creator_affinity > 0:
        reasons.append(
            "you have creator affinity with this account"
        )


    if recency > 0.50:

        reasons.append(
            "it is recent"
        )


    if not reasons:

        reasons.append(
            "recommended based on your profile"
        )


    return ", ".join(
        reasons
    )
# PERSONALIZED VIDEO FEED
print("\n")

print("=" * 80)

print(
    "PERSONALIZED VIDEO FEED"
)

print("=" * 80)

print(
    """
Commands:

  Enter       -> Finish current video and go next
  l / like    -> Like current video and STAY
  s / save    -> Save current video and STAY
  b / both    -> Like + Save current video and STAY
  f / follow  -> Follow current creator and STAY
  n / no      -> Stop

Important:
  views increase only once when a video starts.
  l/s/b do NOT increase views.
"""
)

print("=" * 80)
# SESSION STATE
shown_post_ids = set()

video_number = 0
# FEED LOOP
while True:
    # Re-rank using latest local embedding
    ranked_posts = calculate_ranked_posts(

        user_embedding,

        # Track posts already shown in this session
        shown_post_ids

    )


    if not ranked_posts:

        print(
            "\nNo more recommended videos available."
        )

        break
    # Select best remaining post
    current_item = ranked_posts[0]

    current_post = (
        current_item["post"]
    )


    current_post_id = (
        current_post.get(
            "postId"
        )
    )


    # Track posts already shown in this session
    shown_post_ids.add(
        current_post_id
    )


    video_number += 1
    # Values
    similarity = (
        current_item[
            "similarity"
        ]
    )

    final_score = (
        current_item[
            "finalScore"
        ]
    )

    creator_id = (
        current_post.get(
            "userId"
        )
    )

    caption = (
        current_post.get(
            "caption",
            "No caption"
        )
    )


    # Check whether duration is available
    video_duration = (
        get_video_duration(
            current_post
        )
    )
    # DISPLAY VIDEO
    print("\n")

    print("=" * 80)

    print(
        f"VIDEO {video_number}"
    )

    print("=" * 80)

    print(
        f"Post ID        : "
        f"{current_post_id}"
    )

    print(
        f"Creator ID     : "
        f"{creator_id}"
    )

    print(
        f"Caption        : "
        f"{caption}"
    )

    print(
        f"Similarity     : "
        f"{similarity:.4f}"
    )

    print(
        f"Final Score    : "
        f"{final_score:.4f}"
    )

    creator_followed = (
        creator_id in db_followed_creators
        or
        is_creator_followed(
            local_follows,
            user_id,
            creator_id
        )
    )

    creator_affinity_value = creator_affinity.get(
        creator_id,
        0.0
    )

    print(
        f"Creator Follow : "
        f"{'Yes' if creator_followed else 'No'}"
    )

    print(
        f"Creator Affinity: "
        f"{creator_affinity_value:.4f}"
    )

    print(
        f"Video Duration : "
        f"{video_duration:.2f} seconds"
    )
    # Duration warning
    if video_duration <= 0:

        print(
            "\nWARNING:"
        )

        print(
            "Video duration is 0 or missing."
        )

        print(
            "Completion rate cannot be calculated."
        )

        print(
            "Watch time will still be recorded."
        )


    print(
        "\nWhy recommended:"
    )

    print(
        "  -> "
        +
        get_recommendation_reason(
            current_item
        )
        +
        "."
    )


    print("-" * 80)
    # CURRENT VIDEO STATE
    current_watch_time = 0.0

    liked = False

    bookmarked = False
    # Existing interaction
    existing_interaction = (
        get_interaction(

            local_interactions,

            user_id,

            current_post_id

        )
    )


    if existing_interaction:

        liked = bool(

            existing_interaction.get(
                "liked",
                False
            )

        )


        bookmarked = bool(

            existing_interaction.get(
                "bookmarked",
                False
            )

        )
    # COUNT ONE VIEW
    update_local_interaction(

        local_interactions,

        user_id,

        current_post_id,

        watch_time_delta=0,

        video_duration=video_duration,

        liked=False,

        bookmarked=False,

        increment_view=True, caption=caption

    )


    save_local_interactions(
        local_interactions
    )
    # START TIMER
    print(
        "\nVideo started..."
    )

    print(
        "Timer started."
    )


    # Start timing the current watch segment
    segment_start = time.time()
    # COMMAND LOOP
    while True:

        # Wait for the user's interaction command
        command = input(
            "\nYour command: "
        ).strip().lower()
        # Calculate ONLY the latest segment
        now = time.time()


        # Measure only the latest watch segment
        segment_watch_time = (

            now
            -
            segment_start

        )
        # Don't exceed actual duration
        if video_duration > 0:

            remaining = (

                video_duration
                -
                current_watch_time

            )


            # Measure only the latest watch segment
            segment_watch_time = min(

                segment_watch_time,

                max(
                    0,
                    remaining
                )

            )


        current_watch_time += (
            segment_watch_time
        )
        # LIKE
        if command in [
            "l",
            "like"
        ]:

            liked = True


            # Save ONLY this segment
            update_local_interaction(

                local_interactions,

                user_id,

                current_post_id,

                watch_time_delta=(
                    segment_watch_time
                ),

                video_duration=(
                    video_duration
                ),

                liked=True,

                bookmarked=False,

                increment_view=False, caption=caption

            )


            save_local_interactions(
                local_interactions
            )
            # Update embedding
            post_embedding = (
                post_embeddings[
                    current_post_id
                ]
            )


            (
                user_embedding,
                weight,
                completion_rate
            ) = update_local_user_embedding(

                user_embedding,

                post_embedding,

                current_watch_time,

                video_duration,

                liked=True,

                bookmarked=bookmarked

            )


            save_user_embedding(

                user_id,

                username,

                user_embedding

            )


            print(
                "\nLIKE recorded."
            )

            print(
                "You remain on the same video."
            )

            print(
                f"Current watch time : "
                f"{current_watch_time:.2f} seconds"
            )

            print(
                f"Liked              : "
                f"{liked}"
            )

            print(
                f"Saved              : "
                f"{bookmarked}"
            )

            print(
                f"Embedding weight   : "
                f"{weight:.2f}"
            )

            print(
                "Views were NOT incremented."
            )


            # Important:
            # Start a NEW timing segment
            segment_start = time.time()


            continue
        # SAVE
        elif command in [
            "s",
            "save"
        ]:

            bookmarked = True


            update_local_interaction(

                local_interactions,

                user_id,

                current_post_id,

                watch_time_delta=(
                    segment_watch_time
                ),

                video_duration=(
                    video_duration
                ),

                liked=False,

                bookmarked=True,

                increment_view=False, caption=caption

            )


            save_local_interactions(
                local_interactions
            )


            post_embedding = (
                post_embeddings[
                    current_post_id
                ]
            )


            (
                user_embedding,
                weight,
                completion_rate
            ) = update_local_user_embedding(

                user_embedding,

                post_embedding,

                current_watch_time,

                video_duration,

                liked=liked,

                bookmarked=True

            )


            save_user_embedding(

                user_id,

                username,

                user_embedding

            )


            print(
                "\nSAVE recorded."
            )

            print(
                "You remain on the same video."
            )

            print(
                f"Current watch time : "
                f"{current_watch_time:.2f} seconds"
            )

            print(
                f"Liked              : "
                f"{liked}"
            )

            print(
                f"Saved              : "
                f"{bookmarked}"
            )

            print(
                f"Embedding weight   : "
                f"{weight:.2f}"
            )

            print(
                "Views were NOT incremented."
            )


            # Start timing the current watch segment
            segment_start = time.time()


            continue
        # LIKE + SAVE
        elif command in [
            "b",
            "both"
        ]:

            liked = True

            bookmarked = True


            update_local_interaction(

                local_interactions,

                user_id,

                current_post_id,

                watch_time_delta=(
                    segment_watch_time
                ),

                video_duration=(
                    video_duration
                ),

                liked=True,

                bookmarked=True,

                increment_view=False, caption=caption

            )


            save_local_interactions(
                local_interactions
            )


            post_embedding = (
                post_embeddings[
                    current_post_id
                ]
            )


            (
                user_embedding,
                weight,
                completion_rate
            ) = update_local_user_embedding(

                user_embedding,

                post_embedding,

                current_watch_time,

                video_duration,

                liked=True,

                bookmarked=True

            )


            save_user_embedding(

                user_id,

                username,

                user_embedding

            )


            print(
                "\nLIKE + SAVE recorded."
            )

            print(
                "You remain on the same video."
            )

            print(
                f"Current watch time : "
                f"{current_watch_time:.2f} seconds"
            )

            print(
                f"Liked              : "
                f"{liked}"
            )

            print(
                f"Saved              : "
                f"{bookmarked}"
            )

            print(
                f"Embedding weight   : "
                f"{weight:.2f}"
            )

            print(
                "Views were NOT incremented."
            )


            # Start timing the current watch segment
            segment_start = time.time()


            continue
        # FOLLOW CREATOR
        elif command in [
            "f",
            "follow"
        ]:

            add_local_follow(
                local_follows,
                user_id,
                creator_id
            )

            save_local_follows(
                local_follows
            )

            followed = True

            # Follow is a creator-level signal.
            # We use the current creator's post embedding as a
            # lightweight prototype update to the user embedding.
            post_embedding = (
                post_embeddings[
                    current_post_id
                ]
            )

            (
                user_embedding,
                weight,
                completion_rate
            ) = update_local_user_embedding(
                user_embedding,
                post_embedding,
                current_watch_time,
                video_duration,
                liked=liked,
                bookmarked=bookmarked,
                followed=True
            )

            save_user_embedding(
                user_id,
                username,
                user_embedding
            )

            print(
                "\nFOLLOW recorded."
            )

            print(
                f"Creator ID       : {creator_id}"
            )

            print(
                "Follow saved locally."
            )

            print(
                f"Embedding weight : {weight:.2f}"
            )

            print(
                "Views were NOT incremented."
            )

            # Start timing the current watch segment
            segment_start = time.time()

            continue
        # ENTER = NEXT VIDEO
        elif command == "":

            print(
                "\nFinishing current video..."
            )
            # Save ONLY the final segment
            update_local_interaction(

                local_interactions,

                user_id,

                current_post_id,

                watch_time_delta=(
                    segment_watch_time
                ),

                video_duration=(
                    video_duration
                ),

                liked=liked,

                bookmarked=bookmarked,

                increment_view=False, caption=caption

            )


            save_local_interactions(
                local_interactions
            )
            # Update embedding from this viewing session
            post_embedding = (
                post_embeddings[
                    current_post_id
                ]
            )


            (
                user_embedding,
                weight,
                completion_rate
            ) = update_local_user_embedding(

                user_embedding,

                post_embedding,

                current_watch_time,

                video_duration,

                liked,

                bookmarked

            )


            save_user_embedding(

                user_id,

                username,

                user_embedding

            )
            # Read final interaction
            final_interaction = (
                get_interaction(

                    local_interactions,

                    user_id,

                    current_post_id

                )
            )


            print(
                "\nLocal interaction saved."
            )

            print(
                f"Watch time      : "
                f"{final_interaction.get('watchTime', 0):.2f} seconds"
            )

            if video_duration > 0:
                print(
                    f"Completion rate : "
                    f"{final_interaction.get('completionRate', 0) * 100:.2f}%"
                )
            else:
                print(
                    "Completion rate : unavailable (video duration missing)"
                )

            print(
                f"Liked           : "
                f"{final_interaction.get('liked', False)}"
            )

            print(
                f"Saved           : "
                f"{final_interaction.get('bookmarked', False)}"
            )

            print(
                f"Views           : "
                f"{final_interaction.get('views', 0)}"
            )

            print(
                f"Embedding weight: "
                f"{weight:.2f}"
            )

            print(
                "MongoDB user embedding was NOT modified."
            )

            print(
                "\nRe-ranking remaining videos "
                "using updated local embedding..."
            )


            break
        # STOP
        elif command in [
            "n",
            "no"
        ]:

            print(
                "\nStopping feed..."
            )
            # Save current segment
            update_local_interaction(

                local_interactions,

                user_id,

                current_post_id,

                watch_time_delta=(
                    segment_watch_time
                ),

                video_duration=(
                    video_duration
                ),

                liked=liked,

                bookmarked=bookmarked,

                increment_view=False, caption=caption

            )


            save_local_interactions(
                local_interactions
            )


            final_interaction = (
                get_interaction(

                    local_interactions,

                    user_id,

                    current_post_id

                )
            )


            print(
                "\nCurrent interaction saved."
            )

            print(
                f"Watch time : "
                f"{final_interaction.get('watchTime', 0):.2f} seconds"
            )

            print(
                f"Views      : "
                f"{final_interaction.get('views', 0)}"
            )

            print(
                "MongoDB was not modified."
            )


            client.close()

            exit()
        # INVALID COMMAND
        else:

            print(
                "\nInvalid command."
            )

            print(
                "Use:"
            )

            print(
                "  Enter = next"
            )

            print(
                "  l = like"
            )

            print(
                "  s = save"
            )

            print(
                "  b = like + save"
            )

            print(
                "  f = follow creator"
            )

            print(
                "  n = stop"
            )
# END
print("\n")

print("=" * 80)

print(
    "PERSONALIZED FEED COMPLETED"
)

print("=" * 80)

print(
    f"Videos shown        : "
    f"{video_number}"
)

print(
    f"Local interactions  : "
    f"{len(local_interactions)}"
)

print(
    f"Embedding file      : "
    f"{USER_EMBEDDING_FILE}"
)

print(
    f"Interactions file   : "
    f"{LOCAL_INTERACTIONS_FILE}"
)

print(
    f"Local follows       : "
    f"{LOCAL_FOLLOWS_FILE}"
)

print(
    "\nMongoDB remained READ-ONLY."
)


client.close()
