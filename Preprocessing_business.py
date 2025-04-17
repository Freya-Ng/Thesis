# Version 3.0
#!/usr/bin/env python
# user ID - bin_useful - bin_avg_star - bin_review_count
# item ID - bin_avg_star - bin_review_count - category_feature_hash
#!/usr/bin/env python#!/usr/bin/env python
#!/usr/bin/env python
"""
Updated Version 3.1: Compact re-indexing for users and businesses

User feature vector: [ user_id, bin_useful, bin_avg_star, bin_review_count ]
Business feature vector: [ business_id, bin_avg_star, bin_review_count, category_feature_hash ]

After filtering (non-elite users and open businesses with popular categories),
this version reindexes users and businesses so that feature IDs are assigned sequentially
only to those with actual interactions.
"""

import json
import datetime
import csv
import argparse
import random
import hashlib
from collections import defaultdict, Counter

# === Binning Functions ===

def bin_avg_star(star):
    if star < 2.5:
        return 0
    elif star < 3.5:
        return 1
    elif star < 4.0:
        return 2
    elif star < 4.5:
        return 3
    else:
        return 4

def bin_review_count_user(count):
    if count < 10:
        return 1010
    elif count < 50:
        return 1011
    elif count < 200:
        return 1012
    else:
        return 1013

def bin_review_count_business(rc):
    if rc < 10:
        return 3010
    elif rc < 50:
        return 3011
    elif rc < 200:
        return 3012
    else:
        return 3013

def bin_generic(val, base):
    if val == 0:
        return base
    elif val < 5:
        return base + 1
    elif val < 20:
        return base + 2
    else:
        return base + 3

# === Data Loading Functions ===

def load_users(user_file):
    """
    Loads Yelp user JSON and builds a reduced user feature vector.
    Final user feature vector:
      [ user_id, bin_useful, bin_avg_star, bin_review_count ]
    Only non-elite users are considered.
    """
    users = {}
    user_mapping = {}
    idx = 0
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            uid = data['user_id']
            # Skip elite users.
            if data.get('elite'):
                continue
            if uid not in user_mapping:
                user_mapping[uid] = idx
                idx += 1
            # Original user feature is built with the original mapping.
            user_id_feat = user_mapping[uid] + 1
            avg_star = float(data.get('average_stars', 0.0))
            review_count_val = int(data.get('review_count', 0))
            useful_val = int(data.get('useful', 0))
            user_avg_star_bin = bin_avg_star(avg_star) + 1000
            user_review_count_bin = bin_review_count_user(review_count_val)
            user_useful_bin = bin_generic(useful_val, 1100)
            users[uid] = [user_id_feat, user_useful_bin, user_avg_star_bin, user_review_count_bin]
    return users, user_mapping

def load_businesses(business_file, use_hashing=False, num_category_bins=1000, min_category_frequency=500):
    """
    Loads Yelp business JSON and builds a reduced business feature vector.
    Final business feature vector:
      [ business_id, bin_avg_star, bin_review_count, category_feature_hash ]
    Only open businesses with at least one popular category are kept.
    """
    businesses = {}
    business_mapping = {}
    category_counter = Counter()
    category_mapping = {}
    category_idx = 4000
    idx = 0

    # First pass: count category frequencies.
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_cats = data.get('categories')
            if raw_cats:
                cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
                for cat in cats:
                    category_counter[cat] += 1

    # Second pass: build business features.
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            bid = data['business_id']
            raw_cats = data.get('categories')
            if raw_cats:
                cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
            else:
                cats = []
            # Filter out businesses without any popular category.
            if not any(category_counter[cat] >= min_category_frequency for cat in cats):
                continue
            # Only keep open businesses.
            if data.get("is_open", 0) != 1:
                continue
            if bid not in business_mapping:
                business_mapping[bid] = idx
                idx += 1
            # Original business feature built with the original mapping.
            business_id_feat = business_mapping[bid] + 1 + 2000  # Offset for business IDs.
            stars_val = float(data.get('stars', 0.0))
            stars_bin = bin_avg_star(stars_val) + 3000
            rc = int(data.get('review_count', 0))
            rc_bin = bin_review_count_business(rc)
            # Use only the first category.
            if cats:
                if use_hashing:
                    h = int(hashlib.md5(cats[0].encode('utf-8')).hexdigest(), 16)
                    cat_feat = 4000 + (h % num_category_bins)
                else:
                    if cats[0] not in category_mapping:
                        category_mapping[cats[0]] = category_idx
                        category_idx += 1
                    cat_feat = category_mapping[cats[0]]
            else:
                cat_feat = 0
            businesses[bid] = [business_id_feat, stars_bin, rc_bin, cat_feat]
    return businesses, business_mapping

def load_reviews(review_file, users, businesses):
    """
    Loads reviews, keeping only those where both user and business are present.
    """
    reviews_by_user = defaultdict(list)
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            uid = data['user_id']
            bid = data['business_id']
            date_str = data.get('date', None)
            if not date_str:
                continue
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if uid not in users or bid not in businesses:
                continue
            reviews_by_user[uid].append({'business_id': bid, 'date': date_obj})
    return reviews_by_user

def frequency_filter_reviews(reviews_by_user, min_user_reviews):
    """
    Keep only users with at least min_user_reviews reviews.
    """
    return {uid: reviews for uid, reviews in reviews_by_user.items() if len(reviews) >= min_user_reviews}

def stratified_subsample_reviews(reviews_by_user, subsample_ratio):
    """
    Performs stratified subsampling of users based on how many reviews they have.
    """
    bins = defaultdict(list)
    for uid, reviews in reviews_by_user.items():
        bins[len(reviews)].append(uid)
    sampled_reviews = {}
    for count, uids in bins.items():
        sample_size = max(1, int(len(uids) * subsample_ratio))
        sampled_uids = random.sample(uids, sample_size)
        for uid in sampled_uids:
            sampled_reviews[uid] = reviews_by_user[uid]
    return sampled_reviews

def split_train_test(reviews_by_user):
    """
    For each user, uses the most recent review as test and the rest as train.
    """
    train_reviews = []
    test_reviews = []
    for uid, review_list in reviews_by_user.items():
        review_list.sort(key=lambda x: x['date'])
        test_review = review_list[-1]
        for r in review_list[:-1]:
            train_reviews.append({'user_id': uid, 'business_id': r['business_id']})
        test_reviews.append({'user_id': uid, 'business_id': test_review['business_id']})
    return train_reviews, test_reviews

def features_to_string(feature_list):
    """
    Converts a feature list to a dash-separated string.
    """
    return "-".join(str(x) for x in feature_list)

def write_csv(filename, interactions, users, businesses):
    """
    Writes CSV rows where each row contains a user feature string and a business feature string.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for inter in interactions:
            uid = inter['user_id']
            bid = inter['business_id']
            user_feat_str = features_to_string(users[uid])
            business_feat_str = features_to_string(businesses[bid])
            writer.writerow([user_feat_str, business_feat_str])

# === Main Function ===

def main():
    parser = argparse.ArgumentParser(
        description="Convert Yelp JSON data to train.csv and test.csv with minimal features for recommendation."
    )
    parser.add_argument("--user_file", type=str, default="yelp_user.json")
    parser.add_argument("--business_file", type=str, default="yelp_business.json")
    parser.add_argument("--review_file", type=str, default="yelp_review.json")
    parser.add_argument("--train_output", type=str, default="train.csv")
    parser.add_argument("--test_output", type=str, default="test.csv")
    parser.add_argument("--min_user_reviews", type=int, default=5, help="Minimum reviews per user to keep.")
    parser.add_argument("--min_category_frequency", type=int, default=10000, help="Minimum frequency for a category to be considered popular.")
    parser.add_argument("--subsample_ratio", type=float, default=0.5, help="Subsample ratio for users (0-1).")
    parser.add_argument("--hash_categories", action="store_true", help="Use feature hashing for categories.")
    parser.add_argument("--num_category_bins", type=int, default=1000, help="Number of bins for category hashing.")
    args = parser.parse_args()

    print("Loading user data (non-elite only)...")
    users, _ = load_users(args.user_file)
    print(f"Loaded {len(users)} users.")

    print("Loading business data (open businesses with popular categories)...")
    businesses, _ = load_businesses(
        args.business_file,
        use_hashing=args.hash_categories,
        num_category_bins=args.num_category_bins,
        min_category_frequency=args.min_category_frequency
    )
    print(f"Loaded {len(businesses)} businesses after filtering.")

    print("Loading review data...")
    reviews_by_user = load_reviews(args.review_file, users, businesses)
    reviews_by_user = frequency_filter_reviews(reviews_by_user, args.min_user_reviews)
    print(f"After filtering, {len(reviews_by_user)} users remain (min {args.min_user_reviews} reviews).")

    if args.subsample_ratio < 1.0:
        reviews_by_user = stratified_subsample_reviews(reviews_by_user, args.subsample_ratio)
        print(f"After subsampling, {len(reviews_by_user)} users remain (subsample ratio {args.subsample_ratio}).")

    # ===== Reindex Users =====
    # Only keep users with interactions and assign them new compact IDs.
    new_user_mapping = {}
    new_users = {}
    new_user_index = 0
    for uid in reviews_by_user.keys():
        new_user_mapping[uid] = new_user_index
        old_feats = users[uid]
        # Update user ID feature with new index (starting from 1).
        new_feats = [new_user_mapping[uid] + 1] + old_feats[1:]
        new_users[uid] = new_feats
        new_user_index += 1
    users = new_users
    print(f"Users reindexed: {len(users)} users with interactions.")

    # ===== Reindex Businesses =====
    # Determine which businesses are used in reviews.
    used_business_ids = set()
    for review_list in reviews_by_user.values():
        for r in review_list:
            used_business_ids.add(r['business_id'])

    new_business_mapping = {}
    new_businesses = {}
    new_business_index = 0
    for bid in used_business_ids:
        new_business_mapping[bid] = new_business_index
        old_feats = businesses[bid]
        # Update business ID feature with new index (starting from 1, preserving offset +2000).
        new_feats = [new_business_mapping[bid] + 1 + 2000] + old_feats[1:]
        new_businesses[bid] = new_feats
        new_business_index += 1
    businesses = new_businesses
    print(f"Businesses reindexed: {len(businesses)} businesses used in reviews.")

    print("Splitting data into train and test sets...")
    train_reviews, test_reviews = split_train_test(reviews_by_user)
    print(f"Train interactions: {len(train_reviews)}, Test interactions: {len(test_reviews)}")

    print("Writing train.csv...")
    write_csv(args.train_output, train_reviews, users, businesses)
    print("Writing test.csv...")
    write_csv(args.test_output, test_reviews, users, businesses)
    print("Done.")

if __name__ == "__main__":
    main()
