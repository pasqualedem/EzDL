

class PolyLR:
    def __init__(self, power):
        self.poly_exp = power

    def perform_scheduling(self, initial_lr, epoch, max_epoch, **kwargs):
        """
        Perform polyLR learning rate scheduling
        """
        return initial_lr * pow((1.0 - epoch / max_epoch), self.poly_exp)
