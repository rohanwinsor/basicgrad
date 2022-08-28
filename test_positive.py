from basicgrad import Value, Neuron

def test_c2f():
    c2f = lambda x : float(1.8*x + 32) 
    model = Neuron(1, nonlin=False)
    model.parameters()
    X = [[Value(float(i))] for i in range(-5,5)]
    y = [Value(c2f(i)) for i in range(-5,5)]
    lr = 0.01
    while True:
        losss = []
        for x, yout in zip(X,y):
            pred = model(x)
            loss = (yout - pred)**2
            losss.append(loss)
        loss = sum(losss)
        if loss.data < 0.0001:
            break
        for pr in model.parameters():
            pr.grad = 0.0
        loss.backward()
        for pr in model.parameters():
            pr.data += (-lr * pr.grad)
    # print(model.parameters())
    assert sum([abs(model([Value(i)]).data - c2f(i)) for i in range(5,10)])/5 < 0.01
