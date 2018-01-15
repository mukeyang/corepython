

# absolute path
def read_data(filePath):
    ratings = []
    u_max_num = 0
    v_max_num = 0
    with open(filePath) as f:
        for i in f.readlines():
            rating = list(map(lambda x : int(x), i[0:-1].split('\t')))
            u_max_num = max(u_max_num, rating[0])
            v_max_num = max(v_max_num, rating[1])
            ratings.append(rating)
    return ratings, u_max_num, v_max_num


def get_user_item_interaction_map(ratings, u_max_num, v_max_num):
    user_map_item = {}
    latest_item_interaction = [0] * (u_max_num + 1)
    latest_item_index = [0] * (u_max_num + 1)
    latest_time = [0] * (u_max_num + 1)
    pruned_all_ratings = []

    for i, (u, v, r, t) in enumerate(ratings):
        if u >= u_max_num or v >= v_max_num:
            raise ValueError('user index or item index out of bounds')
        if u in user_map_item.keys():
            user_map_item[u] = [v]
        else:
            user_map_item.append(v)

        if latest_time[u] < t:
            latest_item_index[u] = i
            latest_time[u] = t

    latest_item_index.sort()
    j = 0
    for i, rating in enumerate(ratings):
        if i == latest_item_index[j]:
            latest_item_interaction[rating[0]] = rating[1]
            j += 1
        else:
            pruned_all_ratings.append(ratings)

    return user_map_item, latest_item_interaction, pruned_all_ratings

