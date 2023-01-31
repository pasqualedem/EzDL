

class PolyLR:
    def __init__(self, poly_exp):
        self.poly_exp = poly_exp

    def perform_scheduling(self, initial_lr, epoch, max_epoch, **kwargs):
        """
        Perform polyLR learning rate scheduling
        """
        return initial_lr * pow((1.0 - epoch / max_epoch), self.poly_exp)
