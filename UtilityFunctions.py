from tensorboardX import SummaryWriter
writer = SummaryWriter('new_test')


def lr_finder(model, optim, criterion, dataloader, skip=10, max_iter=200, lr_min=1e-4):

    count = 0
    for param in optim.param_groups:
        param['lr'] = lr_min

    while True:
        for batch in dataloader:
            count += 1

            optim.zero_grad()

            *X, Y = batch

            output = model(X[0], X[1])
            loss = criterion(output.view(-1), Y.float())
            loss.backward()

            optim.step()

            if count <= skip:
                continue

            for param in optim.param_groups:
                param['lr'] = lr_min * (count - skip)
                lr = param['lr']

            writer.add_scalar('lr', loss/1000, lr)

            if count - skip == max_iter:
                return