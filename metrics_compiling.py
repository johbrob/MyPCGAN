import torch


class MetricCompiler():
    def __init__(self):
        self.correct_FD = 0
        self.correct_fake_GD = 0
        self.correct_real_GD = 0
        self.correct_gender_fake_GD = 0
        self.correct_digit = 0
        self.fixed_correct_gender = 0

    def compute_metrics(self, pred_secret, gender, fake_pred_secret, real_pred_secret,
                        fake_secret, gen_secret, pred_digit,
                        digit, fixed_pred_secret):
        # FD accuracy on original gender
        predicted_gender_FD = torch.argmax(pred_secret, 1)
        self.correct_FD += (predicted_gender_FD == gender.long()).sum()

        # GD accuracy on original gender in real and generated (fake) data,
        # and sampled gender in generated (fake) data
        predicted_fake_GD = torch.argmax(fake_pred_secret, 1)
        predicted_real_GD = torch.argmax(real_pred_secret, 1)

        self.correct_fake_GD += (predicted_fake_GD == fake_secret).sum()
        self.correct_real_GD += (predicted_real_GD == gender).sum()
        self.correct_gender_fake_GD += (predicted_fake_GD == gen_secret).sum()

        # Calculate number of correct classifications for the fixed classifiers on the training set
        predicted_digit = torch.argmax(pred_digit.data, 1)
        self.correct_digit += (predicted_digit == digit).sum()

        fixed_predicted = torch.argmax(fixed_pred_secret.data, 1)
        self.fixed_correct_gender += (fixed_predicted == gender.long()).sum()

    def reset(self):
        self.correct_FD = 0
        self.correct_fake_GD = 0
        self.correct_real_GD = 0
        self.correct_gender_fake_GD = 0
        self.correct_digit = 0
        self.fixed_correct_gender = 0
