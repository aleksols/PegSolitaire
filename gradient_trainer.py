from splitgd import SplitGD
import tensorflow as tf

class GradientTrainer(SplitGD):
    
    def __init__(self, keras_model, eligibilities):
        super().__init__(keras_model)
        self.eligibilities = eligibilities

    def modify_gradients(self, gradients):
        print("Gradients")
        modified = []
        for i, gradient in enumerate(gradients):
            print(gradient)
            print(i, gradient*0 + 1)
            print(gradient)
            modified.append(gradient*0+1)
        return modified