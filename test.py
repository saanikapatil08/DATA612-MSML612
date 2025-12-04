import pickle, pathlib
if __name__ == "__main__":
    with open(pathlib.Path("out_epa/output/plant_series.pkl"), "rb") as f:
        plant_series = pickle.load(f)
    print(len(plant_series))