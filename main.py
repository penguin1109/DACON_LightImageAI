import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available)
model = UFRC()

if torch.cuda.is_available():
    model.to('cuda:0')


def train_unet(model, optimizer, scheduler, start_epoch, num_epochs, train_loader):
    start_epoch = 50
    num_epoch = 50
    net, optim, start_epoch = load(ckpt_dir = CKPT_DIR, net = net, optim = optim)
    net = net.to(device)

    for epoch in range(start_epoch + 1, num_epoch + 1+start_epoch):
        net.train()
        loss_arr = []

    for batch, data in enumerate(train_loader, 1):
        # forward
        input = data['input'].to(device)
        transformed = data['transformed'].to(device)
        label = data['label'].to(device)
        for iter in range(2):
            if iter == 0:
                output_1 = net(input)
                optim.zero_grad()
                loss = fn_loss(output_1, label)
                loss.backward()
                optim.step()
                loss_arr += [loss.item()]

            else:
                output_2 = net(transformed)
                optim.zero_grad()
                loss = fn_loss(output_2, label)
                loss.backward()
                optim.step()
                loss_arr += [loss.item()]
  

        if int(batch) == 1:
            print('train : epoch %f/ %f | Batch %f/ %f | Loss %f'%(epoch, num_epoch+start_epoch, batch, num_train_for_epoch, loss))

            
    train_writer.add_scalar('loss', np.mean(loss_arr), epoch)
  
  # validation test
  # validation dataset을 이용해서 학습의 정확도를 판단하고자 하는 상황이기 때문에 backprop진행을 안함
    with torch.no_grad() :
        net.eval()
        loss_arr = []

        for batch,data in enumerate(val_loader, 1):
      # forward
        input = data['input'].to(device)
        label = data['label'].to(device)

        output = net(input)
        loss = fn_loss(output, label)
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        if int(batch) == 1:
            print('valid : epoch %f/ %f | Batch %f / %f | Loss %f'%(epoch, num_epoch+start_epoch, batch, num_val_for_epoch, loss))

    val_writer.add_scalar('loss', np.mean(loss_arr), epoch)

    save(ckpt_dir = CKPT_DIR, net = net, optim = optim, epoch = epoch)


def train_fullnet(model, optimizer, scheduler, start_epoch, num_epochs, train_loader):
    for epoch in range(start_epoch, num_epochs + start_epoch):
        model.train()
        loss_ar = []

        for batch, data in enumerate(train_loader, 1):
            input = data['input'].to(device)
            transformed = data['transformed'].to(device)
            label = data['label'].to(device)
            label_orig = data['original_label'].to(device)

            output = model(input)
            optim.zero_grad()
            loss = fn_loss(output, label_orig)
            loss.backward()
            optim.step()
            loss_arr += [loss.item()]
        

            