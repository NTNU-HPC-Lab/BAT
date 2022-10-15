import os
from src.result.zenodo import Zenodo

if __name__ == "__main__":
    datasets = os.listdir("./results")
    z = Zenodo(datasets)
    z.upload()
    # WARNING!: DO NOT UNCOMMENT THIS LINE UNLESS YOU FULLY UNDERSTAND WHAT IT DOES:  z.delete_all()

