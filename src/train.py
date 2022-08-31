import equinox as eqx

def step(model, x, y, rng, optim, optim_state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y, key=rng)
    updates, optim_state = optim.update(grad, optim_state)
    model = eqx.apply_updates(model, updates)
    return model, optim_state, loss