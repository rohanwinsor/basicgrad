import torch
from basicgrad import Value, Neuron

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()


def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

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
