from main_train import main as train
from main_infer import main_infer as infer

if __name__ == "__main__":
    train(pretest=False)
    infer()
