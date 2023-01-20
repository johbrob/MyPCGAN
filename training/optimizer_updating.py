import utils

class OptimizerUpdater:

    def __init__(self, disc_ga, gen_ga, disc_optimizers, gen_optimizers):
        self.disc_ga = disc_ga
        self.gen_ga = gen_ga
        self.disc_optimizers = disc_optimizers
        self.gen_optimizers = gen_optimizers

    def step(self, total_steps, losses):
        if total_steps % self.disc_ga == 0:
            print('update disc')
            utils.backward(losses)
            utils.step(self.disc_optimizers)
            utils.zero_grad(self.disc_optimizers)

        if total_steps % self.gen_ga == 0:
            print('update gen')
            utils.backward(losses)
            utils.step(self.gen_optimizers)
            utils.zero_grad(self.gen_optimizers)