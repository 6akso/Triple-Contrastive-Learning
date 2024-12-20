import random
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Subset, random_split, DataLoader
from tqdm import tqdm
from model import Transformer
from config import get_config
from Base_loss import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel
from earlystop import EarlyStopping

class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert1')

        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('bert2', add_prefix_space=True)

        else:
            raise ValueError('unknown model')

        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")




    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            #------------------------------------------------------------------------------
            outputs1 = self.model(inputs)
            loss1 = criterion(outputs1, targets)
            loss_init=loss1
            loss=loss_init
            #-------------------------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)

            n_correct += ((torch.argmax(outputs1['predicts'], -1) == targets) ).sum().item()
            n_train += targets.size(0)
        return train_loss / n_train, n_correct / n_train,loss

    def _test(self, dataloader, criterion):
        num_classes = args.num_classes
        test_loss, n_correct, n_test = 0, 0, 0
        n_tp = [0] * num_classes
        n_fp = [0] * num_classes
        n_fn = [0] * num_classes
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                predicted_labels = torch.argmax(outputs['predicts'], -1)
                n_correct += (predicted_labels == targets).sum().item()
                n_test += targets.size(0)
                for c in range(num_classes):
                    n_tp[c] += ((predicted_labels == c) & (targets == c)).sum().item()
                    n_fp[c] += ((predicted_labels == c) & (targets != c)).sum().item()
                    n_fn[c] += ((predicted_labels != c) & (targets == c)).sum().item()

        accuracy = n_correct / n_test
        class_precision = [n_tp[c] / (n_tp[c] + n_fp[c]) if (n_tp[c] + n_fp[c]) > 0 else 0 for c in range(num_classes)]
        class_recall = [n_tp[c] / (n_tp[c] + n_fn[c]) if (n_tp[c] + n_fn[c]) > 0 else 0 for c in range(num_classes)]
        class_f1_score = [2 * class_precision[c] * class_recall[c] / (class_precision[c] + class_recall[c]) if (class_precision[c] +class_recall[c]) > 0 else 0 for c in range(num_classes)]
        avg_precision = sum(class_precision) / num_classes
        avg_recall = sum(class_recall) / num_classes
        avg_f1_score = sum(class_f1_score) / num_classes
        return test_loss / n_test, accuracy, avg_precision, avg_recall, avg_f1_score




    def run(self):
        seed=123
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        train_fraction = 0.75
        kf = KFold(n_splits=5, shuffle=False)
        dataset, workers, collate_fn = load_data(dataset=self.args.dataset,
                                                 data_dir=self.args.data_dir,
                                                 tokenizer=self.tokenizer,
                                                 model_name=self.args.model_name,
                                                 method=self.args.method,
                                                 train_batch_size=self.args.train_batch_size,
                                                 test_batch_size=self.args.test_batch_size,
                                                 workers=0)

        sumbestaccuracy = 0
        sumbestprecision = 0
        sumbestrecall = 0
        sumbestf1 = 0
        a = []
        b = []
        c = []
        d = []

        sumbestaccuracy2 = 0
        sumbestprecision2 = 0
        sumbestrecall2 = 0
        sumbestf12 = 0

        aa = []
        bb = []
        cc = []
        dd = []


        for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            best_loss, best_acc, best_precision, best_recall, best_f1 = 0, 0, 0, 0, 0

            if args.model_name == 'bert':
                base_model = AutoModel.from_pretrained('bert1')
            elif args.model_name == 'roberta':
                base_model = AutoModel.from_pretrained('bert2')
            else:
                raise ValueError('unknown model')
            self.model = Transformer(base_model, args.num_classes,args.method)
            self.model.to(args.device)  # 将模型部署到当前计算机
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            if self.args.method == 'ce':
                criterion = CELoss()
            elif self.args.method == 'scl':
                criterion = SupConLoss(self.args.alpha, self.args.temp)
            elif self.args.method == 'dualcl':
                criterion = DualLoss(self.args.alpha, self.args.temp)
            else:
                raise ValueError('unknown method')
            optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)


            train_subset = Subset(dataset, train_indices)
            train_subset_size = int(len(train_subset) * train_fraction)
            validation_subset_size = len(train_subset) - train_subset_size
            train_dataset, validation_dataset = random_split(train_subset, [train_subset_size, validation_subset_size])
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                          num_workers=workers, collate_fn=collate_fn, pin_memory=True, shuffle=False)
            validation_dataloader = DataLoader(validation_dataset, batch_size=args.train_batch_size,
                                               num_workers=workers, collate_fn=collate_fn, pin_memory=True,
                                               shuffle=False)
            test_dataloader = DataLoader(Subset(dataset, val_indices), batch_size=args.test_batch_size,
                                         num_workers=workers, collate_fn=collate_fn, pin_memory=True, shuffle=True)

            best_model_state = None
            with open("metrics.txt", "w") as f:
                trainlosses = []
                testlosses = []
                trainaaccuracy = []
                testaccuracy = []
                earstop = EarlyStopping()
                for epoch in range(self.args.num_epoch):
                    epoch2=epoch+1

                    train_loss, train_acc, loss_init, = self._train(train_dataloader, criterion,optimizer)
                    test_loss, test_acc, precision, recall, f1_score = self._test(validation_dataloader,criterion)


                    trainlosses.append(train_loss)
                    testlosses.append(test_loss)
                    trainaaccuracy.append(train_acc)
                    testaccuracy.append(test_acc)
                    if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                        best_acc, best_loss = test_acc, test_loss
                        best_model_state = self.model.state_dict()
                    if precision > best_precision or (precision == best_precision and test_loss < best_loss):
                        best_precision = precision
                    if recall > best_recall or (recall == best_recall and test_loss < best_loss):
                        best_recall = recall
                    if f1_score > best_f1 or (recall == best_recall and test_loss < best_loss):
                        best_f1 = f1_score

                    earstop(test_acc)
                    if earstop.early_stop:
                        print("Early stopping")
                        break



                    self.logger.info('{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch,100 * (epoch + 1) / self.args.num_epoch))
                    self.logger.info('[train] loss: {:.4f}, acc: {:.2f},loss_init:{:.4f}'.format(train_loss,train_acc * 100,loss_init))
                    self.logger.info('[test] loss: {:.4f}, acc: {:.2f},precision: {:2f},recall: {:2f},f1 : {:2f}'.format(test_loss,test_acc * 100,precision * 100,recall * 100,f1_score * 100))
                self.logger.info('fold：{},best loss: {:.4f}, best acc: {:.2f},best precision: {:.2f},best recall: {:.2f},best f1: {:.2f}'.format(fold, best_loss, best_acc * 100, best_precision * 100, best_recall * 100, best_f1 * 100, ))

                plt.plot(range(epoch2), trainlosses, label="trainLoss")
                plt.plot(range(epoch2), testlosses, label="testLoss")
                plt.plot(range(epoch2), trainaaccuracy, label="trainaccuracy")
                plt.plot(range(epoch2), testaccuracy, label="testaccuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Irregular Loss Curve")
                plt.legend()
                plt.show()
                sumbestaccuracy = sumbestaccuracy + (best_acc * 100)
                a.append(best_acc * 100)

                sumbestprecision = sumbestprecision + (best_precision * 100)
                b.append(best_precision * 100)

                sumbestrecall = sumbestrecall + (best_recall * 100)
                c.append(best_recall * 100)

                sumbestf1 = sumbestf1 + (best_f1 * 100)
                d.append(best_f1 * 100)

                if best_model_state:
                    # 加载最佳模型状态
                    self.model.load_state_dict(best_model_state)
                    # 使用最佳模型状态在测试集上运行
                    val_loss, val_acc, val_precision, val_recall, val_f1_score = self._test(test_dataloader, criterion)
                    self.logger.info(f"Test Loss: {val_loss}, Test Accuracy: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1_score}")

                    sumbestaccuracy2 = sumbestaccuracy2 + (val_acc * 100)
                    aa.append(val_acc * 100)

                    sumbestprecision2 = sumbestprecision2 + (val_precision * 100)
                    bb.append(val_precision * 100)

                    sumbestrecall2 = sumbestrecall2 + (val_recall * 100)
                    cc.append(val_recall * 100)

                    sumbestf12 = sumbestf12 + (val_f1_score * 100)
                    dd.append(val_f1_score * 100)


                self.logger.info('log saved: {}'.format(self.args.log_name))
        self.logger.info('averageacc: {:}, a:{:}.averageprecision: {:},b:{:}. averagercall: {:},c:{:}.averagf1: {:},c:{:},'.format(sumbestaccuracy / 5, a, sumbestprecision / 5, b, sumbestrecall / 5, c, sumbestf1 / 5, d))
        self.logger.info('Test:averageacc: {:}, a:{:}.averageprecision: {:},b:{:}. averagercall: {:},c:{:}.averagf1: {:},c:{:},'.format(sumbestaccuracy2 / 5, aa, sumbestprecision2 / 5, bb, sumbestrecall2 / 5, cc, sumbestf12 / 5, dd))

if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()