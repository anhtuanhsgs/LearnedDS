import argparse
from models.models import test
parser = argparse.ArgumentParser(description='test_model')

if __name__ == "__main__":
    args = parser.parse_args()
    print ("testing")
    test (args)