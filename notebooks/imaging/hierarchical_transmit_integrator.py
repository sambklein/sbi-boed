import torch
from gradoptics.integrator import HierarchicalSamplingIntegrator

class HierarchicalIntegratorWithTransmittance(HierarchicalSamplingIntegrator):
    def __init__(self, nb_mc_steps, nb_importance_samples):
        super().__init__(nb_mc_steps, nb_importance_samples)
        
    def compute_integral(self, incident_rays, pdf, t_min, t_max):
        t_vals = torch.linspace(0., 1., steps=self.nb_mc_steps + 1, device=incident_rays.origins.device)
        z_vals = t_min[:, None] * (1.-t_vals[None, :]) + t_max[:, None] * (t_vals[None, :])

        z_vals = z_vals.expand([incident_rays.origins.shape[0], self.nb_mc_steps+1])

        if self.stratify > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=incident_rays.origins.device)

            z_vals = lower + (upper - lower) * t_rand

        pts = incident_rays.origins[...,None,:] + incident_rays.directions[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        x_vals_mid = incident_rays.origins.expand(self.nb_mc_steps, -1, -1).transpose(0, 1) + z_vals_mid.unsqueeze(
                -1) * incident_rays.directions.expand(self.nb_mc_steps, -1, -1).transpose(0, 1)
        
        if self.nb_importance_samples > 0:
            deltas = z_vals[:, 1:] - z_vals[:, :-1]
            weights = pdf(x_vals_mid.reshape(-1, 3)).reshape((x_vals_mid.shape[:2]))*deltas
        
            z_imp = self.sample_pdf(z_vals_mid, 
                                    weights[...,1:], self.nb_importance_samples, det=(self.stratify==0.))
            z_imp = z_imp.detach()

            z_vals = torch.cat((z_imp, z_vals), dim=-1)
            z_vals, index = torch.sort(z_vals, dim=-1)
            
        deltas = z_vals[:, 1:] - z_vals[:, :-1]
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # 3d positions at the different times t
        x = incident_rays.origins.expand(z_vals_mid.shape[-1], -1, -1).transpose(0, 1) + z_vals_mid.unsqueeze(
            -1) * incident_rays.directions.expand(z_vals_mid.shape[-1], -1, -1).transpose(0, 1)

        densities = pdf(x.reshape(-1, 3)).reshape((x.shape[:2]))
        
        alpha = 1.0 - torch.exp(-(densities * deltas))
        weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]).to(alpha), 1. - alpha + 1e-7], -1), -1)[:, :-1] 
       
        return weights.sum(dim=1)