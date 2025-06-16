import collections, pickle
meta = pickle.load(open("data/fclip_meta.pkl", "rb"))
cnt  = collections.Counter(x[2] for x in meta)   # x[2] 是 item_type
print("总条数:", len(meta))
print("dress 数:", cnt["dress"])
print(cnt.most_common(10))
