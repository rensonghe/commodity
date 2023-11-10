import pickle5 as pickle

def read_pickle():
    with open("test.pkl", "rb") as f:
        obj_again = pickle.load(f)
    print("反序列化后的对象为{}".format(obj_again))