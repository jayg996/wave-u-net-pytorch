import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import HParams
import time

use_cuda = torch.cuda.is_available()

class Waveunet(nn.Module):
    def __init__(self,config):
        super(Waveunet, self).__init__()
        self.enc_num_layers = config['enc_num_layers']
        self.dec_num_layers = config['dec_num_layers']
        self.enc_filter_size = config['enc_filter_size']
        self.dec_filter_size = config['dec_filter_size']
        self.input_channel = config['input_channel']
        self.nfilters = config['nfilters']

        enc_channel_in = [self.input_channel] + [min(self.dec_num_layers, (i + 1)) * self.nfilters for i in range(self.enc_num_layers - 1)]
        enc_channel_out = [min(self.dec_num_layers, (i + 1)) * self.nfilters for i in range(self.enc_num_layers)]
        dec_channel_out = enc_channel_out[:self.dec_num_layers][::-1]
        dec_channel_in = [enc_channel_out[-1]*2 + self.nfilters] + [enc_channel_out[-i-1] + dec_channel_out[i-1] for i in range(1, self.dec_num_layers)]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(self.enc_num_layers):
            self.encoder.append(nn.Conv1d(enc_channel_in[i], enc_channel_out[i], self.enc_filter_size))

        for i in range(self.dec_num_layers):
            self.decoder.append(nn.Conv1d(dec_channel_in[i], dec_channel_out[i], self.dec_filter_size))

        self.middle_layer = nn.Sequential(
            nn.Conv1d(enc_channel_out[-1], enc_channel_out[-1] + self.nfilters, self.enc_filter_size),
            nn.LeakyReLU(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Conv1d(self.nfilters + self.input_channel, self.input_channel, kernel_size=1),
            nn.Tanh()
        )

    def forward(self,x):
        encoder = list()
        input = x

        # Downsampling
        for i in range(self.enc_num_layers):
            x = self.encoder[i](x)
            x = F.leaky_relu(x,0.2)
            encoder.append(x)
            x = x[:,:,::2]

        x = self.middle_layer(x)

        # Upsampling
        for i in range(self.dec_num_layers):
            x = F.interpolate(x, size=x.shape[-1]*2-1, mode='linear', align_corners=True)
            x = self.crop_and_concat(x, encoder[self.enc_num_layers - i - 1])
            x = self.decoder[i](x)
            x = F.leaky_relu(x,0.2)

        # Concat with original input
        x = self.crop_and_concat(x, input)

        # Output prediction
        output_vocal = self.output_layer(x)
        output_accompaniment = self.crop(input, output_vocal.shape[-1]) - output_vocal
        return output_vocal, output_accompaniment

    def crop_and_concat(self, x1, x2):
        crop_x2 = self.crop(x2, x1.shape[-1])
        x = torch.cat([x1,crop_x2],dim=1)
        return x

    def crop(self, tensor, target_shape):
        # Center crop
        shape = tensor.shape[-1]
        diff = shape - target_shape
        crop_start = diff // 2
        crop_end = diff - crop_start
        return tensor[:,:,crop_start:-crop_end]


if __name__ == "__main__":
    config = HParams.load("hparams.yaml")
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 2
    input_channel = 2
    input_sample = 147443

    input_mix = torch.randn(batch_size, input_channel, input_sample, requires_grad=True).to(device)
    input_mix = input_mix.tanh()

    model = Waveunet(config=config.waveunet).to(device)

    st = time.time()
    output_vocal, output_accompaniment = model(input_mix)
    print(output_vocal.shape)
    print(output_accompaniment.shape)
    print(time.time() - st)
