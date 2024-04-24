I am trying to solve a seq-to-seq problem with a LSTM in Pytorch. Concretely, I am taking sequences of 5 elements, to predict the next 5 ones. My concern has to do with the data transformations. I have tensors of size [bs, seq_length, features], where `seq_length = 5`, and `features = 1`. Each feature is an int number between 0 and 3 (both included).

I believed they had to be transformed to float range [0, 1] with a MinMaxScaler, in order to make the LSTM learning process easier. After that, I apply a Linear layer, which transforms the hidden states into the corresponding output, with size `features`. The code I use to do the training loop is the following:

```python
def train_loop(t, checkpoint_epoch, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, X in enumerate(dataloader):
        X = X[0].type(torch.float).to(device)

        # X = torch.Size([batch_size, 10, input_dim])
        # Split sequences into input and target
        inputs = transform(X[:, :5, :]) # inputs = [batch_size, 5, input_dim]
        targets = transform(X[:, 5:, :]) # targets = [batch_size, 5, input_dim]

        # predictions (forward pass)
        with autocast():
            pred = model(inputs)  # pred = [batch_size, 5, input_dim]
            loss = loss_fn(pred, targets)

        # backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            #print(f"Current loss: {loss:>7f}, [{current:>5d}/{size:>5d}]")

        # Delete variables and empty cache
        del X, inputs, targets, pred
        torch.cuda.empty_cache()

    return loss
```

I also show my definition of the LSTM network in Pytorch:

```python
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(LSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout_prob)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    ...

    def forward(self, X):
        out, (hidden, cell) = self.lstm_layer(X)
        out = self.output_layer(out)
        return out
```

Trying this, the model was overfitting so much, so I was thinking that maybe directly calculating the loss between `targets` (float values in range [0, 1]) and `pred` (float values in range [-1, 1]), with different scales might be wrong. Then, I tried aplying a sigmoid activation function right after the Linear layer in the forward pass, but overfitting was also appearing. I tried executions with many hyperparameter combinations, but any resulted in a "normal" training curve (I'll also attach a screenshot for 5000 epochs to ilustrate the training process).

My questions are, what seems to be wrong in my training process? Is there anything I said that is thought in a wrong way?

Thanks in advance.
