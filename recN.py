import os
import json
import time
import numpy as np

from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in .env file")


client = MongoClient(MONGODB_URI)

client.admin.command("ping")

print("\nMongoDB connection successful!")
print("MongoDB is used in READ-ONLY mode.\n")

db = client["test"]


USER_EMBEDDING_FILE = "user_embedding.json"
LOCAL_INTERACTIONS_FILE = "local_interactions.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


def normalize_embedding(embedding):
    embedding = np.array(
        embedding,
        dtype=float
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


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
        USER_EMBEDDING_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


def load_local_user_embedding(user_id):

    if not os.path.exists(
        USER_EMBEDDING_FILE
    ):
        return None

    try:

        with open(
            USER_EMBEDDING_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if data.get("userId") != user_id:
            return None

        embedding = data.get("embedding")

        if not embedding:
            return None

        return normalize_embedding(embedding)

    except Exception as e:

        print(
            "Could not load local embedding:",
            e
        )

        return None


def update_local_user_embedding(
    user_embedding,
    post_embedding,
    watch_time,
    liked,
    bookmarked
):

    post_embedding = normalize_embedding(
        post_embedding
    )

    weight = 0.05

    if watch_time >= 30:
        weight += 0.05
    elif watch_time >= 10:
        weight += 0.02

    if liked:
        weight += 0.10

    if bookmarked:
        weight += 0.10

    weight = min(
        weight,
        0.30
    )

    new_embedding = (
        (1 - weight) * user_embedding
        +
        weight * post_embedding
    )

    new_embedding = normalize_embedding(
        new_embedding
    )

    return (
        new_embedding,
        weight
    )


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


def get_interaction(
    interactions,
    user_id,
    post_id
):

    for interaction in interactions:

        if (
            interaction.get("userId") == user_id
            and
            interaction.get("postId") == post_id
        ):

            return interaction

    return None


def update_local_interaction(
    interactions,
    user_id,
    post_id,
    watch_time_delta,
    liked=False,
    bookmarked=False,
    increment_view=False,
    caption=""
):

    existing = get_interaction(
        interactions,
        user_id,
        post_id
    )

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

        existing["liked"] = (
            existing.get(
                "liked",
                False
            )
            or
            liked
        )

        existing["bookmarked"] = (
            existing.get(
                "bookmarked",
                False
            )
            or
            bookmarked
        )

        if increment_view:

            existing["views"] = (
                int(
                    existing.get(
                        "views",
                        0
                    )
                )
                +
                1
            )

        if caption:

            existing["caption"] = caption

        existing[
            "lastInteractionAt"
        ] = utc_now()

        return existing

    new_interaction = {
        "userId": user_id,
        "postId": post_id,
        "caption": caption,
        "watchTime": watch_time_delta,
        "liked": liked,
        "bookmarked": bookmarked,
        "views": 1 if increment_view else 0,
        "lastInteractionAt": utc_now()
    }

    interactions.append(
        new_interaction
    )

    return new_interaction


def get_age_days(created_at):

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

        elif isinstance(
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

        else:

            return 9999.0

        now = datetime.now(
            timezone.utc
        )

        return max(
            0.0,
            (
                now - created_at
            ).total_seconds()
            / 86400
        )

    except Exception:

        return 9999.0


def normalize_values(values):

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


def calculate_ranked_posts(
    current_user_embedding,
    excluded_post_ids,
    user_id
):

    candidates = []

    raw_popularity = []
    raw_recency = []
    raw_engagement = []
    raw_watch = []

    for post in posts:

        post_id = post.get(
            "postId"
        )

        creator_id = post.get(
            "userId"
        )

        if not post_id:
            continue

        if creator_id == user_id:
            continue

        if post_id in excluded_post_ids:
            continue

        if post_id not in post_embeddings:
            continue

        post_embedding = (
            post_embeddings[
                post_id
            ]
        )

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

        interaction = get_interaction(
            local_interactions,
            user_id,
            post_id
        )

        liked = 0
        bookmarked = 0
        watch_time = 0

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

        engagement_score = (
            liked * 3
            +
            bookmarked * 4
        )

        watch_score = min(
            watch_time / 30,
            1
        )

        candidates.append(
            {
                "post": post,
                "similarity": similarity,
                "popularity": popularity,
                "recency": recency,
                "engagement": engagement_score,
                "watchTime": watch_time
            }
        )

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
            watch_score
        )

    if not candidates:
        return []

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

    for i, item in enumerate(
        candidates
    ):

        similarity = item[
            "similarity"
        ]

        final_score = (
            similarity * 0.55
            +
            engagement_norm[i] * 0.20
            +
            watch_norm[i] * 0.10
            +
            popularity_norm[i] * 0.10
            +
            recency_norm[i] * 0.05
        )

        item[
            "finalScore"
        ] = final_score

    candidates.sort(
        key=lambda x:
            x["finalScore"],
        reverse=True
    )

    return candidates


def get_recommendation_reason(
    item
):

    reasons = []

    similarity = item[
        "similarity"
    ]

    engagement = item[
        "engagement"
    ]

    watch_time = item[
        "watchTime"
    ]

    recency = item[
        "recency"
    ]

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
            "your previous interaction indicates interest"
        )

    if watch_time >= 10:

        reasons.append(
            "you spent time watching similar content"
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


print("Fetching users...\n")

users = list(
    db["userdetails"].find(
        {},
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
        "No users found."
    )

    client.close()
    exit()


print("=" * 80)
print("AVAILABLE USERS")
print("=" * 80)

for i, user in enumerate(
    users,
    1
):

    print(
        f"{i}. {user.get('name', 'N/A')}"
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
                "userId": user_id
            },
            {
                "embedding": 1
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


local_interactions = (
    load_local_interactions()
)


print(
    "\nFetching active posts..."
)

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
            "category": 1
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


print(
    "Fetching post embeddings..."
)

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

for document in post_embedding_documents:

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


print(
    "Fetching post metrics..."
)

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

for document in post_metric_documents:

    post_id = document.get(
        "postId"
    )

    if not post_id:
        continue

    post_metrics[
        post_id
    ] = {
        "likes": safe_float(
            document.get(
                "totalLikes",
                0
            )
        ),
        "comments": safe_float(
            document.get(
                "totalComments",
                0
            )
        ),
        "shares": safe_float(
            document.get(
                "totalShares",
                0
            )
        ),
        "bookmarks": safe_float(
            document.get(
                "totalBookmarks",
                0
            )
        )
    }


print("\n")
print("=" * 80)
print("PERSONALIZED VIDEO FEED")
print("=" * 80)

print(
    """
Commands:

  Enter       -> Finish current video and go next
  l / like    -> Like current video and STAY
  s / save    -> Save current video and STAY
  b / both    -> Like + Save current video and STAY
  n / no      -> Stop

Important:
  views increase only once when a video starts.
  l/s/b do NOT increase views.
  watch time is recorded locally.
  MongoDB remains READ-ONLY.
"""
)

print("=" * 80)


shown_post_ids = set()
video_number = 0


while True:

    ranked_posts = calculate_ranked_posts(
        user_embedding,
        shown_post_ids,
        user_id
    )

    if not ranked_posts:

        print(
            "\nNo more recommended videos available."
        )

        break

    current_item = ranked_posts[0]

    current_post = (
        current_item["post"]
    )

    current_post_id = (
        current_post.get(
            "postId"
        )
    )

    shown_post_ids.add(
        current_post_id
    )

    video_number += 1

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


    current_watch_time = 0.0

    liked = False
    bookmarked = False


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


    update_local_interaction(
        local_interactions,
        user_id,
        current_post_id,
        watch_time_delta=0,
        liked=False,
        bookmarked=False,
        increment_view=True,
        caption=caption
    )

    save_local_interactions(
        local_interactions
    )


    print(
        "\nVideo started..."
    )

    print(
        "Timer started."
    )

    segment_start = time.time()


    while True:

        command = input(
            "\nYour command: "
        ).strip().lower()

        now = time.time()

        segment_watch_time = (
            now
            -
            segment_start
        )

        current_watch_time += (
            segment_watch_time
        )


        if command in [
            "l",
            "like"
        ]:

            liked = True

            update_local_interaction(
                local_interactions,
                user_id,
                current_post_id,
                watch_time_delta=segment_watch_time,
                liked=True,
                bookmarked=False,
                increment_view=False,
                caption=caption
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
                weight
            ) = update_local_user_embedding(
                user_embedding,
                post_embedding,
                current_watch_time,
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

            segment_start = time.time()

            continue


        elif command in [
            "s",
            "save"
        ]:

            bookmarked = True

            update_local_interaction(
                local_interactions,
                user_id,
                current_post_id,
                watch_time_delta=segment_watch_time,
                liked=False,
                bookmarked=True,
                increment_view=False,
                caption=caption
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
                weight
            ) = update_local_user_embedding(
                user_embedding,
                post_embedding,
                current_watch_time,
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

            segment_start = time.time()

            continue


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
                watch_time_delta=segment_watch_time,
                liked=True,
                bookmarked=True,
                increment_view=False,
                caption=caption
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
                weight
            ) = update_local_user_embedding(
                user_embedding,
                post_embedding,
                current_watch_time,
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

            segment_start = time.time()

            continue


        elif command == "":

            print(
                "\nFinishing current video..."
            )

            update_local_interaction(
                local_interactions,
                user_id,
                current_post_id,
                watch_time_delta=segment_watch_time,
                liked=liked,
                bookmarked=bookmarked,
                increment_view=False,
                caption=caption
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
                weight
            ) = update_local_user_embedding(
                user_embedding,
                post_embedding,
                current_watch_time,
                liked,
                bookmarked
            )

            save_user_embedding(
                user_id,
                username,
                user_embedding
            )

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


        elif command in [
            "n",
            "no"
        ]:

            print(
                "\nStopping feed..."
            )

            update_local_interaction(
                local_interactions,
                user_id,
                current_post_id,
                watch_time_delta=segment_watch_time,
                liked=liked,
                bookmarked=bookmarked,
                increment_view=False,
                caption=caption
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
                "  n = stop"
            )


print("\n")
print("=" * 80)
print("PERSONALIZED FEED COMPLETED")
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
    "\nMongoDB remained READ-ONLY."
)

client.close()