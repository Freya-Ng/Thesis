import os
import json
import datetime
import csv
import argparse
import random
from collections import defaultdict

# === Data Loading Functions (Without Binning) ===

def load_users(user_file):
    """
    Loads Yelp user JSON and builds a reduced user feature vector.
    Final user feature vector:
      [ user_id, unique_useful, unique_avg_star, unique_review_count ]
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
            user_id_feat = user_mapping[uid] + 1
            avg_star = float(data.get('average_stars', 0.0))
            review_count_val = int(data.get('review_count', 0))
            useful_val = int(data.get('useful', 0))
            # Unique feature transformation:
            # Multiply to preserve one decimal of precision and add an offset.
            unique_avg_star = int(round(avg_star * 10)) + 40000  # e.g., 4.3 -> 43 + 40000 = 40043
            unique_review_count = review_count_val + 50000
            unique_useful = useful_val + 60000
            users[uid] = [user_id_feat, unique_useful, unique_avg_star, unique_review_count]
    return users, user_mapping

def load_businesses(business_file, allowed_categories=None):
    """
    Loads Yelp business JSON and builds a reduced business feature vector.
    Final business feature vector:
      [ business_id, unique_avg_star, unique_review_count, city ]
    Only open businesses that belong to one of the allowed categories are kept.
    
    Parameters:
      business_file (str): Path to the business JSON file.
      allowed_categories (set or None): A set of category names to filter by.
                                        If None, defaults to {"Restaurants"}.
                                        
    Returns:
      businesses (dict): Mapping from business ID to its feature vector.
      business_mapping (dict): Mapping from original business ID to new index.
    """
    if allowed_categories is None:
        allowed_categories = {"Food", "Restaurants", "Shopping"}

    businesses = {}
    business_mapping = {}
    city_mapping = {}   # new dynamic mapping for city names to a compact unique ID
    idx = 0

    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            bid = data['business_id']
            raw_cats = data.get('categories')
            cats = []
            if raw_cats:
                cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
            # Filter out businesses that do not belong to at least one allowed category.
            if not set(cats).intersection(allowed_categories):
                continue
            # Only keep open businesses.
            if data.get("is_open", 0) != 1:
                continue
            if bid not in business_mapping:
                business_mapping[bid] = idx
                idx += 1
            # Reindex business with an offset (+4000) so the business feature ID is unique.
            business_id_feat = business_mapping[bid] + 1 + 4000
            stars_val = float(data.get('stars', 0.0))
            # Multiply stars_val by 10 to capture one decimal of precision.
            unique_avg_star = int(round(stars_val * 10)) + 5000
            review_count_val = int(data.get('review_count', 0))
            unique_review_count = review_count_val + 6000
            # Instead of using a raw frequency count for city,
            # assign a compact unique ID for each encountered city.
            city_raw = data.get('city', "").strip()
            if city_raw not in city_mapping:
                city_mapping[city_raw] = len(city_mapping) + 1  # start numbering from 1
            city_val = city_mapping[city_raw]
            businesses[bid] = [business_id_feat, unique_avg_star, unique_review_count, city_val]
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
            date_str = data.get('date')
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
    Filters out users with fewer than the specified minimum reviews.
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
    Implements leave-one-out: For each user, the most recent review is set aside for testing
    and the remainder are used for training.
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
    Converts a list of features to a dash-separated string.
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
        description="Create a synthetic train/test set for combined categories with unique feature values."
    )
    parser.add_argument("--user_file", type=str, default="yelp_user.json", help="Path to the Yelp user JSON file")
    parser.add_argument("--business_file", type=str, default="yelp_business.json", help="Path to the Yelp business JSON file")
    parser.add_argument("--review_file", type=str, default="yelp_review.json", help="Path to the Yelp review JSON file")
    parser.add_argument("--min_user_reviews", type=int, default=5, help="Minimum reviews per user to keep.")
    parser.add_argument("--subsample_ratio", type=float, default=1.0, help="Subsample ratio for users (0-1)")
    parser.add_argument("--output_dir", type=str, default=r"/media/leo/Huy/Project/CARS/Yelp JSON/yelp_dataset/F_yelp", help="Output folder for synthetic train/test CSV files")
    args = parser.parse_args()

    # Define allowed categories as the union of Food, Home Services, and Shopping.
    allowed_categories = {"Food", "Restaurants", "Shopping"}
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading user data (non-elite only)...")
    users, _ = load_users(args.user_file)
    print(f"Loaded {len(users)} users.")

    print("Loading business data for allowed categories...")
    businesses, _ = load_businesses(args.business_file, allowed_categories=allowed_categories)
    print(f"Loaded {len(businesses)} businesses from allowed categories: {allowed_categories}.")

    print("Loading review data...")
    reviews_by_user = load_reviews(args.review_file, users, businesses)
    reviews_by_user = frequency_filter_reviews(reviews_by_user, args.min_user_reviews)
    print(f"After filtering, {len(reviews_by_user)} users remain (min {args.min_user_reviews} reviews).")

    if args.subsample_ratio < 1.0:
        reviews_by_user = stratified_subsample_reviews(reviews_by_user, args.subsample_ratio)
        print(f"After subsampling, {len(reviews_by_user)} users remain.")

    # ===== Global Reindexing of Users =====
    new_users = {}
    new_user_index = 0
    for uid in reviews_by_user.keys():
        old_feats = users[uid]
        # Replace the original user feature ID with the new global index (starting from 1).
        new_feats = [new_user_index + 1] + old_feats[1:]
        new_users[uid] = new_feats
        new_user_index += 1
    users_global = new_users
    print(f"Users reindexed: {len(users_global)} users with interactions.")

    # ===== Global Reindexing of Businesses =====
    used_business_ids = set()
    for review_list in reviews_by_user.values():
        for r in review_list:
            used_business_ids.add(r['business_id'])

    new_businesses = {}
    new_business_index = 0
    for bid in used_business_ids:
        old_feats = businesses[bid]
        # Update the business ID with the new global index (starting from 1, preserving offset +2000).
        new_feats = [new_business_index + 1 + 2000] + old_feats[1:]
        new_businesses[bid] = new_feats
        new_business_index += 1
    businesses_global = new_businesses
    print(f"Businesses reindexed: {len(businesses_global)} businesses used in reviews.")

    print("Splitting data using the leave-one-out protocol...")
    train_reviews, test_reviews = split_train_test(reviews_by_user)
    print(f"Total interactions: {len(train_reviews)} train, {len(test_reviews)} test.")

    # Write the synthetic train/test CSV files.
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    print(f"Writing train.csv to {train_path}...")
    write_csv(train_path, train_reviews, users_global, businesses_global)
    print(f"Writing test.csv to {test_path}...")
    write_csv(test_path, test_reviews, users_global, businesses_global)
    print("Synthetic train/test set creation complete.")

if __name__ == "__main__":
    main()
