import feature_preparation
import model_training
import model_inference


if __name__ == '__main__':
    feature_preparation.prepare_features()
    model_training.train_model()
    model_inference.infer_on_model()
