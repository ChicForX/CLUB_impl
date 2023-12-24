import torch
import torch.nn as nn
from nets import PriorGenerator, ZDiscriminator, Encoder, UtilityDecoder, UncertaintyDecoder, UtilityDiscriminator

class CLUB(nn.Module):
    def __init__(self, dim_z, dim_u, dim_noise, dim_img):
        super(CLUB, self).__init__()
        self.prior_generator = PriorGenerator(DIM_Z=dim_z, noise_dim=dim_noise)
        self.z_discriminator = ZDiscriminator(DIM_Z=dim_z)
        self.encoder = Encoder(dim_z=dim_z, img_dim=dim_img)
        self.utility_decoder = UtilityDecoder(DIM_Z=dim_z, DIM_U=dim_u)
        self.uncertainty_decoder = UncertaintyDecoder(DIM_Z=dim_z, DIM_S=dim_u)
        self.utility_discriminator = UtilityDiscriminator(DIM_U=dim_u)

    def forward(self, x, s):
        # since the training procedure is complex, CLUB is manually forwarded in main()
        pass

