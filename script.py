from utils import data
from config import DATA_PATH

PATH_X = DATA_PATH / "data_x"
PATH_Y = DATA_PATH / "data_y"

def main() -> None:
    dataset = data.Dataset(PATH_X, PATH_Y)
    print(dataset[3])

if __name__ == "__main__":
    main()