import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
# import wandb
import time
import argparse
from utilities import *
from TaylorNet import *

def train_epoch(epoch, train_loader, model, criterion, optimizer, device, log_interval=10):
    model.train()
    
    total_time = 0
    total_loss = 0.0
    total_label = torch.tensor([], dtype=torch.float).to(device)
    total_pred = torch.tensor([], dtype=torch.float).to(device)

    for batch_idx, (x_batch, y_label) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_label = y_label.to(device)
        y_label = y_label.long()

        optimizer.zero_grad()

        time_start = time.time()
        y_out, z_out = model(x_batch)
        loss = criterion(y_out, y_label) + model.output_loss(z_out)

        loss.backward()
        optimizer.step()

        time_end = time.time()

        total_time += time_end - time_start

        y_pred = F.softmax(y_out, 1)

        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_label = torch.cat((total_label, y_label), dim=0)
        total_loss += loss.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            predicted_results = total_pred.clone().detach().cpu().numpy()
            ground_truths = total_label.clone().detach().cpu().numpy()
            step = epoch * len(train_loader) + batch_idx

            acc, _, _, f1 = macro_statistics(predicted_results, ground_truths)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1: {:.6f}\tAcc: {:.6f}\tTime: {:.6f}'.format(
                epoch, batch_idx * len(x_batch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / batch_idx, f1, acc, total_time / 60))
            # wandb.log({'train_loss': total_loss / batch_idx, 'train_f1': f1, 'train_acc': acc}, step=step)


def test_epoch(epoch, test_loader, train_loader, model, criterion, device, log=False):
    ground_truths = torch.tensor([], dtype=torch.float).to(device)
    predicted_results = torch.tensor([], dtype=torch.float).to(device)
    concept_results = torch.tensor([], dtype=torch.float).to(device)
    test_loss = 0.0
    model.eval()
    
    with torch.no_grad():
        for batch_id, (x_batch, y_label) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_label = y_label.to(device)
            if model.num_outputs != 1:
                y_label = y_label.long()

            y_out, z_out = model(x_batch)
            loss = criterion(y_out, y_label) + model.output_loss(z_out)
            y_pred = F.softmax(y_out, 1)
            test_loss += loss.item()

            predicted_results = torch.cat((predicted_results, y_pred), dim=0)
            ground_truths = torch.cat((ground_truths, y_label), dim=0)
            concept_results = torch.cat((concept_results, z_out), dim=0)

        predicted_results = predicted_results.detach().cpu().numpy()
        ground_truths = ground_truths.detach().cpu().numpy()
        concept_results = concept_results.detach().cpu().numpy()
        step = (epoch + 1) * len(train_loader)
        test_loss /= len(test_loader)

        test_acc, _, _, test_f1 = macro_statistics(predicted_results, ground_truths)
        print('\nTest set: Average loss: {:.4f}\tF1: {:.4f}\tAcc: {:.4f}\n'.format(test_loss, test_f1, test_acc))
        # if log:
        #     wandb.log({'test_loss': test_loss, 'test_f1': test_f1, 'test_acc': test_acc}, step=step)

        return test_loss, ground_truths, predicted_results, concept_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAT')

    # basic config
    parser.add_argument('--training', type=int, required=True, default=1, help='status')
    parser.add_argument('--seed', type=int, required=True, default=0, help='random seed')

    # data
    parser.add_argument('--data_name', type=str, required=True, default='airbnb', help='dataset name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./results/', help='location of results')

    # model
    parser.add_argument('--input_layer', type=str, default='linear', help='input layer type')
    parser.add_argument('--hidden_dims', type=str, default='64,64,32', help='hidden dimensions of concept encoders')
    parser.add_argument('--concept_dropout', type=float, default=0.1, help='dropout of concept encoders')
    parser.add_argument('--order', type=int, default=2, help='order of Taylor polynomial')
    parser.add_argument('--rank', type=int, default=8, help='rank of Tucker decomposition')
    parser.add_argument('--initial', type=str, default='Taylor', help='initialization method')
    parser.add_argument('--batchnorm', type=bool, default=True, help='use batch normalization')
    parser.add_argument('--output_penalty', type=float, default=0.0, help='output penalty')
    parser.add_argument('--encode_concepts', type=bool, default=True, help='encode concepts')

    # optimization
    parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='logging interval')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.use_gpu:
        device = torch.device("cuda", args.gpu)
    else:
        device = torch.device("cpu")

    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    train, val, test, target, features = get_datasets(args.data_name)
    num_out = train.y.sum().item()
    num_in = len(train) - num_out
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TaylorNet(len(features), 
                      DATASETS[args.data_name]['num_classes'],
                      concept_groups=None,
                      input_layer=args.input_layer,
                      hidden_dims=[int(x) for x in args.hidden_dims.split(',')],
                      order=args.order,
                      rank=args.rank,
                      initial=args.initial,
                      concept_dropout=0.0,
                      batchnorm=args.batchnorm,
                      output_penalty=0.0,
                      encode_concepts=False)
    
    # criterion = DATASETS[args.data_name]['criterion']
    if args.data_name == 'CelebA':
        criterion = WeightedLoss('sigmoid', [num_in, num_out], 0.9999, 2.0)
    else:
        criterion = DATASETS[args.data_name]['criterion']
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    if args.training:
        early_stopping = EarlyStopping(patience=args.patience, verbose=False)

        # perform the training

        # with wandb.init(project='CAT', name='Taylor_order{}_{}_seed{}'.format(args.order, args.data_name, args.seed)):
        print('Start training...')
        start_time = time.time()

        for epoch in range(1, args.num_epochs + 1):
            adjust_learning_rate(optimizer, args.learning_rate, epoch, decay=0.1)
            train_epoch(epoch, train_loader, model, criterion, optimizer, device, args.log_interval)
            test_loss, ground_truths, predicted_results, _ = test_epoch(epoch, val_loader, train_loader, model, criterion, device, log=True)

            test_acc, _, _, test_f1 = macro_statistics(predicted_results, ground_truths)
            test_score = - (test_acc + test_f1)
            early_stopping(test_score, model, path=args.checkpoints + args.data_name + '_taylor{}_model_seed{}.pt'.format(args.order, args.seed))
            if early_stopping.early_stop:
                print("Early stopping.")
                break

        end_time = time.time()
        throughput = len(train) * args.num_epochs / (end_time - start_time)
        print('Throughput: {:.6f} samples/s'.format(throughput))
        print('Training finished.')

    else:
        model.load_state_dict(torch.load(args.checkpoints + args.data_name + '_taylor{}_model_seed{}.pt'.format(args.order, args.seed)))
        _, ground_truths, predicted_results, _ = test_epoch(0, test_loader, train_loader, model, criterion, device)

        with open(args.results + args.data_name + '_taylor{}_seed{}.txt'.format(args.order, args.seed), 'w') as f:
            test_acc, test_prec, test_recall, test_f1 = macro_statistics(predicted_results, ground_truths)
            f.write('Accuracy: {:.4f}\n'.format(test_acc))
            f.write('Precision: {:.4f}\n'.format(test_prec))
            f.write('Recall: {:.4f}\n'.format(test_recall))
            f.write('F1: {:.4f}\n'.format(test_f1))