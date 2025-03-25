#!/usr/bin/env python
import json
import datetime
import csv
import argparse
import random
from collections import defaultdict, Counter

# Function to bin review count into categories
def bin_review_count(rc):
    if rc < 10:
        return 0
    elif rc < 50:
        return 1
    elif rc < 200:
        return 2
    else:
        return 3

# Function to bin average star ratings into categories
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

# Load user data and generate features
def load_users(user_file):
    users = {}
    user_mapping = {}
    idx = 0
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            uid = data['user_id']
            if uid not in user_mapping:
                user_mapping[uid] = idx
                idx += 1
            # User feature: starting index from 1, bin for avg_star and review_count, elite flag
            user_id_feat = user_mapping[uid] + 1  # Start from 1
            avg_star = bin_avg_star(float(data.get('average_stars', 0.0))) + 1000
            review_count = bin_review_count(int(data.get('review_count', 0))) + 1010
            elite = 1021 if data.get('elite') else 1020
            users[uid] = [user_id_feat, avg_star, review_count, elite]
    return users, user_mapping

# Load business data and generate features for "Shopping" category
def load_businesses(business_file, top_k_cities=100):
    businesses = {}
    business_mapping = {}
    city_counter = Counter()
    category_mapping = {}
    city_mapping = {}
    state_mapping = {}
    category_idx = 4000
    city_idx = 5000
    state_idx = 5100
    idx = 0

    # First pass: count cities for businesses in the "Shopping" category
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_cats = data.get('categories')
            if not raw_cats or "Shopping" not in raw_cats:
                continue
            city = data.get("city", "").strip()
            if city:
                city_counter[city] += 1

    # Select top K cities based on frequency
    top_cities = set([city for city, _ in city_counter.most_common(top_k_cities)])

    # Second pass: process business data and compute features
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_cats = data.get('categories')
            if not raw_cats or "Shopping" not in raw_cats:
                continue

            bid = data['business_id']
            if bid not in business_mapping:
                business_mapping[bid] = idx
                idx += 1
            business_id_feat = business_mapping[bid] + 2000

            stars_val = float(data.get('stars', 0.0))
            stars_bin = bin_avg_star(stars_val)
            stars_feat = 3000 + stars_bin

            rc = int(data.get('review_count', 0))
            rc_bin = bin_review_count(rc)
            rc_feat = 3010 + rc_bin

            is_open_feat = 3021 if data.get("is_open", 0) == 1 else 3020

            # Process categories: only keep up to 3 features
            categories = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
            cat_feats = []
            for cat in categories:
                if cat not in category_mapping:
                    category_mapping[cat] = category_idx
                    category_idx += 1
                cat_feats.append(category_mapping[cat])
            # Ensure three category features by padding with 0 if needed
            cat_feats = cat_feats[:3]
            while len(cat_feats) < 3:
                cat_feats.append(0)

            # Process city: assign feature if city is in top cities
            city = data.get("city", "").strip()
            if city in top_cities:
                if city not in city_mapping:
                    city_mapping[city] = city_idx
                    city_idx += 1
                city_feat = city_mapping[city]
            else:
                city_feat = 0

            # Process state
            state = data.get("state", "").strip()
            if state:
                if state not in state_mapping:
                    state_mapping[state] = state_idx
                    state_idx += 1
                state_feat = state_mapping[state]
            else:
                state_feat = 0

            businesses[bid] = [
                business_id_feat,
                stars_feat,
                rc_feat,
                is_open_feat,
                *cat_feats,
                city_feat,
                state_feat
            ]
    return businesses, business_mapping

# Load review data and group reviews by user
def load_reviews(review_file, users, businesses):
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
                # Parse date string
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            # Only include review if user and business exist in our datasets
            if uid not in users or bid not in businesses:
                continue
            reviews_by_user[uid].append({'business_id': bid, 'date': date_obj})
    return reviews_by_user

# Split reviews into training and testing sets for each user
def split_train_test(reviews_by_user, min_reviews=2):
    train_reviews = []
    test_reviews = []
    for uid, review_list in reviews_by_user.items():
        if len(review_list) < min_reviews:
            continue
        # Sort reviews by date to pick the latest one for testing
        review_list.sort(key=lambda x: x['date'])
        test_review = review_list[-1]
        for r in review_list[:-1]:
            train_reviews.append({'user_id': uid, 'business_id': r['business_id']})
        test_reviews.append({'user_id': uid, 'business_id': test_review['business_id']})
    return train_reviews, test_reviews

# Convert a list of features into a string for CSV output
def features_to_string(feature_list):
    return "-".join(str(x) for x in feature_list)

# Write the interactions to a CSV file
def write_csv(filename, interactions, users, businesses):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for inter in interactions:
            uid = inter['user_id']
            bid = inter['business_id']
            user_feat_str = features_to_string(users[uid])
            business_feat_str = features_to_string(businesses[bid])
            writer.writerow([user_feat_str, business_feat_str])

# Stratified sampling function for users based on binned avg_star and review_count
def stratified_sample_users(users, sample_fraction=0.1):
    """
    Perform stratified sampling on users.
    users: dict {user_id: [user_id_feat, avg_star, review_count, elite]}
    sample_fraction: fraction of users to sample from each stratum.
    """
    strata = defaultdict(list)
    for uid, feats in users.items():
        # Remove offsets to get original bins
        star_bin = feats[1] - 1000
        rc_bin = feats[2] - 1010
        strata[(star_bin, rc_bin)].append(uid)
    
    sampled_users = set()
    for key, uid_list in strata.items():
        # Ensure at least one sample per stratum
        n_sample = max(1, int(len(uid_list) * sample_fraction))
        sampled_users.update(random.sample(uid_list, n_sample))
    return sampled_users

# Stratified sampling function for businesses based on binned stars and review_count
def stratified_sample_businesses(businesses, sample_fraction=0.1):
    """
    Perform stratified sampling on businesses.
    businesses: dict {business_id: [business_id_feat, stars_feat, rc_feat, is_open_feat, ...]}
    sample_fraction: fraction of businesses to sample from each stratum.
    """
    strata = defaultdict(list)
    for bid, feats in businesses.items():
        # Remove offsets to get original bins
        star_bin = feats[1] - 3000
        rc_bin = feats[2] - 3010
        strata[(star_bin, rc_bin)].append(bid)
    
    sampled_businesses = set()
    for key, bid_list in strata.items():
        # Ensure at least one sample per stratum
        n_sample = max(1, int(len(bid_list) * sample_fraction))
        sampled_businesses.update(random.sample(bid_list, n_sample))
    return sampled_businesses

# Filter reviews to include only those reviews where both the user and business are in the sampled sets
def filter_reviews(reviews_by_user, sampled_users, sampled_businesses):
    """
    reviews_by_user: dict {user_id: list of reviews}
    Returns a filtered dictionary of reviews.
    """
    filtered_reviews = {}
    for uid, reviews in reviews_by_user.items():
        if uid not in sampled_users:
            continue
        filtered = [r for r in reviews if r['business_id'] in sampled_businesses]
        if filtered:
            filtered_reviews[uid] = filtered
    return filtered_reviews

# Main pipeline to load data, perform stratified sampling, split into train/test and write CSV files
def main_sample_pipeline():
    parser = argparse.ArgumentParser(description="Process Yelp JSON data and perform stratified sampling for train/test split.")
    parser.add_argument("--user_file", type=str, default="yelp_user.json", help="Path to the Yelp user JSON file")
    parser.add_argument("--business_file", type=str, default="yelp_business.json", help="Path to the Yelp business JSON file")
    parser.add_argument("--review_file", type=str, default="yelp_review.json", help="Path to the Yelp review JSON file")
    parser.add_argument("--train_output", type=str, default="Shopping/train_sampled.csv", help="Output file for training interactions")
    parser.add_argument("--test_output", type=str, default="Shopping/test_sampled.csv", help="Output file for testing interactions")
    parser.add_argument("--sample_fraction", type=float, default=0.2, help="Fraction to sample for stratified sampling")
    args = parser.parse_args()

    # Load user data
    print("Loading user data...")
    users, user_mapping = load_users(args.user_file)
    print(f"Loaded {len(users)} users.")

    # Load business data (filtering for "Shopping")
    print("Loading business data (only 'Shopping')...")
    businesses, business_mapping = load_businesses(args.business_file)
    print(f"Loaded {len(businesses)} businesses.")

    # Load review data
    print("Loading review data...")
    reviews_by_user = load_reviews(args.review_file, users, businesses)
    print(f"Loaded reviews for {len(reviews_by_user)} users.")

    # Perform stratified sampling on users and businesses
    print("Performing stratified sampling on users and businesses...")
    sampled_users = stratified_sample_users(users, sample_fraction=args.sample_fraction)
    sampled_businesses = stratified_sample_businesses(businesses, sample_fraction=args.sample_fraction)
    print(f"Sampled {len(sampled_users)} users and {len(sampled_businesses)} businesses.")

    # Filter reviews based on sampled users and businesses
    sampled_reviews = filter_reviews(reviews_by_user, sampled_users, sampled_businesses)
    print(f"Filtered reviews for {len(sampled_reviews)} users after sampling.")

    # Split the filtered reviews into training and testing sets
    print("Splitting data into train and test sets...")
    train_reviews, test_reviews = split_train_test(sampled_reviews, min_reviews=2)
    print(f"Train interactions: {len(train_reviews)}, Test interactions: {len(test_reviews)}")

    # Write the train and test CSV files
    print("Writing train CSV file...")
    write_csv(args.train_output, train_reviews, users, businesses)
    print("Writing test CSV file...")
    write_csv(args.test_output, test_reviews, users, businesses)
    print("Sampling and splitting process completed.")

if __name__ == "__main__":
    main_sample_pipeline()
