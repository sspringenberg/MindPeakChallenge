import os
import torch
from torch import nn
from torch.autograd import Variable

class CNNTrainer:
    """
    construct CNNTrainer that conducts the training process
    """

    def __init__(self, args, model, train_loader, test_loader, optimizer):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.args = args
        self.model = model
        self.criterion = nn.NLLLoss()

    def train_epochs(self):
        """
        train for as many epochs as passed as --num-epochs argument
        at the end of each epoch print losses, accuracy and save model
        """

        step = 0
        # last_test_losses = [np.Inf]

        for epoch in range(self.args.num_epochs):
            epoch_train_loss, step = self.train(step)

            if epoch % self.args.checkpoint_interval == 0:
                print('====> Epoch: {} Average train loss'.format(
                    (epoch+1)))
                print('-------------------------')
                print(epoch_train_loss)
                print('-------------------------')
                self.args.writer.add_scalar('training loss',epoch_train_loss,(epoch+1))


                epoch_test_loss, test_accuracy = self.test()
                print('====> Epoch: {} Average test loss'.format(
                    (epoch+1)))
                print('-------------------------')
                print(epoch_test_loss)
                print('-------------------------')
                print('====> Epoch: {} test ACC'.format(
                    (epoch+1)))
                print('-------------------------')
                print(test_accuracy)
                print('-------------------------')

                self.args.writer.add_scalar('test loss',epoch_test_loss,(epoch+1))
                self.args.writer.add_scalar('test accuracy',test_accuracy,(epoch+1))


                if not os.path.exists('experiments'):
                    os.makedirs('experiments')

                torch.save(self.model.state_dict(), 'experiments/'+self.args.model_name)

    def train(self, step):
        """
        iterate over training set and perform SGD after each batch
        """

        self.model.train()

        epoch_loss = 0

        batch_amount = 0
        for batch_idx, (data, label, _) in enumerate(self.train_loader):
            batch_amount += 1
            self.optimizer.zero_grad()

            data = Variable(data)

            if self.args.use_cuda:
                data = data.cuda()

            output = self.model(data)

            loss = self.criterion(output, label)
            loss.backward()

            self.optimizer.step()
            step += 1
            epoch_loss += loss

        return epoch_loss/batch_amount, step

    def test(self):
        """
        iterate over test set and calculate avg loss, accuracy
        """

        self.model.eval()

        epoch_loss = 0

        batch_amount = 0

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, label, _) in enumerate(self.test_loader):

                batch_amount += 1
                data = Variable(data)
                if self.args.use_cuda:
                    data = data.cuda()

                output = self.model(data)

                loss = self.criterion(output, label)
                epoch_loss += loss

                # calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total

        return epoch_loss/batch_amount, accuracy
