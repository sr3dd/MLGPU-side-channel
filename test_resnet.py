from resnet.inferencer import Inferencer

if __name__ == '__main__':
    inferencer = Inferencer(custom_batch_size=1)
    inferencer.load_test_data()
    inferencer.perform_inference()
    