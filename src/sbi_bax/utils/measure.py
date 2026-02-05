def initial_measurement(x_grid, n_init, data, true_thetas, nuisance=None):
    # Make an initial measurement taking first elements of x_grid
    initial_x = x_grid[:n_init]
    if initial_x.ndim == 1:
        initial_x = initial_x.unsqueeze(0)
    # Get the dimension of x
    x_dim = initial_x.shape[1]
    # Repeat for all true thetas
    initial_x = (
        initial_x.unsqueeze(0).repeat_interleave(len(true_thetas), 0).view(-1, x_dim)
    )
    # Repeat thetas for n_init
    true_thetas = true_thetas.repeat_interleave(n_init, 0)
    # Repeat nuisance if given
    if nuisance is not None:
        nuisance = nuisance.repeat_interleave(n_init, 0)
    # Get the intermediate y (or y itself) from the simulator
    y_obs = data.simulator(initial_x, true_thetas.to("cpu"), nuisance=nuisance)
    if y_obs.ndim == 1:
        y_obs = y_obs.unsqueeze(-1)
    # Get the dimension of the observed data
    y_dim = y_obs.shape[1]
    # Define the observed data
    return y_obs.view(-1, n_init, y_dim), initial_x.view(-1, n_init, x_dim)
