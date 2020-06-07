
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from sklearn.metrics import roc_auc_score
    
#function to train the pytorch model
def train_model(model, train_loader, validation_loader, epochs = 30, print_epochs = False, balance_weights = False):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if balance_weights:
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([80]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
        
        
    optimizer = optim.Adam(model.parameters(), lr= 0.003)
       

    model.train()

    train_roc_lst = []
    validation_roc_lst = []
    train_loss_lst = []
    validation_loss_lst = []

    for e in range(epochs):

        epoch_loss = 0
        epoch_roc = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(inputs)

            loss = criterion(output, labels.unsqueeze(1))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            output = torch.round(torch.sigmoid(output))
            output = output.cpu().detach().numpy()
            labels = labels.cpu().unsqueeze(1)

            try:
                epoch_roc += roc_auc_score(labels, output)
            except ValueError:
                pass


        #model evaluation between each epoch
        model.eval()
        with torch.no_grad():

            validation_loss = 0
            validation_roc = 0

            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                output = model.forward(inputs)

                v_loss = criterion(output, labels.unsqueeze(1))
                validation_loss += v_loss.item()

                output = torch.round(torch.sigmoid(output))
                output = output.cpu().detach().numpy()
                labels = labels.cpu().unsqueeze(1)

                try:
                    validation_roc += roc_auc_score(labels, output)
                except ValueError:
                    pass


        epoch_loss = epoch_loss/len(train_loader)
        validation_loss = validation_loss/len(validation_loader)

        epoch_roc = epoch_roc/len(train_loader)
        validation_roc = validation_roc/len(validation_loader)

        if(print_epochs):
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(epoch_loss),
                  "validation Loss: {:.3f}.. ".format(validation_loss),
                  #"recall: : {:.3f}".format(epoch_recall/len(train_loader)),
                  #"prec : {:.3f}".format(epoch_precision/len(train_loader)),
                  "train roc score : {:.3f}..  ".format(epoch_roc),
                  "validation roc score : {:.3f}".format(validation_roc)
                  )

        train_roc_lst.append(epoch_roc)
        validation_roc_lst.append(validation_roc)
        train_loss_lst.append(epoch_loss)
        validation_loss_lst.append(validation_loss)

    return [train_roc_lst, validation_roc_lst, train_loss_lst, validation_loss_lst]


#function to visualize the training and validation loss and evaluation
def visualize_results(results):

    train_roc_lst = results[0]
    validation_roc_lst = results[1]
    train_loss_lst = results[2]
    validation_loss_lst = results[3]

    epoch_lst = list(range(1, len(train_roc_lst)+1))

    plt.figure(figsize=(20,8))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_lst, train_loss_lst, label = 'Train Loss')
    plt.plot(epoch_lst, validation_loss_lst, label = 'validation Loss')
    plt.xticks(list(range(0, len(train_roc_lst)+5, 5)))

    plt.xlabel('epochs')
    plt.ylabel('Loss')

    plt.title('Loss per epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_lst, train_roc_lst, label = 'Train ROC AUC score')
    plt.plot(epoch_lst, validation_roc_lst, label = 'validation ROC AUC score')
    plt.xticks(list(range(0, len(train_roc_lst)+5, 5)))

    plt.xlabel('epochs')
    plt.ylabel('ROC AUC score')

    plt.title('ROC AUC score per epoch')
    plt.legend()

    plt.show()