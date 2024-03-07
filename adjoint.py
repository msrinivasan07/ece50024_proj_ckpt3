class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        # Forward code is not shown here but it's where the ODE is solved forward in time.
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        # Use torch.no_grad() for the entire operation to avoid tracking history
        with torch.no_grad():
            # Pre-allocate tensor for 'z' with the correct device and dtype
            z = torch.empty(time_len, bs, *z_shape, device=z0.device, dtype=z0.dtype)
            z[0] = z0
            for i_t in range(1, time_len):  # Start loop from 1 since z[0] is already set
                z0 = ode_solve(z0, t[i_t - 1], t[i_t], func)
                z[i_t] = z0  # This is already an in-place operation

        # Store necessary objects in ctx
        ctx.func = func
        ctx.save_for_backward(t, z, flat_parameters)  # No need to clone z before saving
        return z


    @staticmethod
    def backward(ctx, dLdz):
        # Retrieve function and saved tensors from context.
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        
        # Extract batch size and dimensions of the state variables.
        bs, *z_shape = dLdz.shape[1:]
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.numel()

        # Define the dynamics for the augmented system used in the backward pass.
        def augmented_dynamics(aug_z_i, t_i):
            # Split the augmented state into the original state and adjoint state.
            z_i = aug_z_i[:, :n_dim].view(bs, *z_shape)
            a = aug_z_i[:, n_dim:2*n_dim].view(bs, *z_shape)
            
            # Enable gradient computation for the backward pass of the backward pass.
            with torch.enable_grad():
                z_i.requires_grad_(True)
                t_i.requires_grad_(True)
                
                # Compute the dynamics and its gradients with respect to state, parameters, and time.
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)
                
                # Handle possible None gradients by providing zero tensors as placeholders.
                if adfdz is not None:
                    adfdz = adfdz.reshape(bs, n_dim)
                else:
                    adfdz = torch.zeros_like(z_i).view(bs, n_dim)

                if adfdp is not None:
                    adfdp = adfdp
                else: 
                    adfdp = torch.zeros(bs, n_params, device=z_i.device)

                if adfdt is not None:
                    adfdt = adfdt
                else: 
                    adfdt = torch.zeros(bs, 1, device=z_i.device)

            # Return the concatenated gradients as the output of the augmented system's dynamics.
            return torch.cat([func_eval.view(bs, n_dim), -adfdz, -adfdp, -adfdt], dim=1)

        # Flatten the gradient of the loss with respect to the output over time.
        dLdz = dLdz.reshape(t.size(0), bs, -1)
        
        # Initialize tensors for storing adjoint states and gradients.
        adj_z = torch.zeros_like(dLdz[0])
        adj_p = torch.zeros(bs, n_params, device=dLdz.device)
        adj_t = torch.zeros_like(t)

        # Iterate backwards through time to compute adjoint states and parameter gradients.
        for i_t in range(t.size(0) - 1, 0, -1):
            z_i = z[i_t].view(bs, n_dim)
            
            # Update the adjoint state with the gradient of the loss.
            adj_z += dLdz[i_t]
            
            # Compute direct gradients with respect to time.
            f_i = func(z_i.view(bs, *z_shape), t[i_t]).view(bs, n_dim)
            adj_t[i_t] -= torch.sum(f_i * adj_z, dim=1, keepdim=True)

            # Concatenate the current state, adjoint state, and other parameters for augmented dynamics.
            aug_z = torch.cat([z_i, adj_z, torch.zeros_like(adj_p), adj_t[i_t:i_t+1]], dim=1)
            
            # Solve the augmented system backwards in time to update adjoints and gradients.
            aug_ans = ode_solve(aug_z, t[i_t], t[i_t - 1], augmented_dynamics)
            adj_z, adj_p_partial, adj_t[i_t - 1:i_t] = aug_ans[:, n_dim:2*n_dim], aug_ans[:, 2*n_dim:2*n_dim + n_params], aug_ans[:, -1:]
            
            # Update the total parameter gradient.
            adj_p += adj_p_partial

        # Update the adjoint state at the initial time with the direct gradient.
        adj_z += dLdz[0]
        
        # Compute the direct gradient with respect to the initial time.
        f_i = func(z[0].view(bs, *z_shape), t[0]).view(bs, n_dim)
        adj_t[0] -= torch.sum(f_i * adj_z, dim=1, keepdim=True)

        # Return gradients with respect to initial state, initial time, parameters, and None for the function (as it's not a leaf node).
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None
