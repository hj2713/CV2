import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        
    def forward(self, x):
        pe = [x]
        for i in range(self.L):
            freq = 2.0 ** i
            pe.append(torch.sin(freq * x))
            pe.append(torch.cos(freq * x))
        return torch.cat(pe, dim=-1)

class NeuralRadianceField(nn.Module):
    def __init__(self, L_xyz=10, L_dir=4, d_filter=256):
        super().__init__()
        self.pe_xyz = PositionalEncoding(L_xyz)
        self.pe_dir = PositionalEncoding(L_dir)
        d_input = 3 + 3 * 2 * L_xyz
        d_direction = 3 + 3 * 2 * L_dir
        
        self.layer1 = nn.Linear(d_input, d_filter)
        self.layer2 = nn.Linear(d_filter, d_filter)
        self.layer3 = nn.Linear(d_filter, d_filter)
        self.layer4 = nn.Linear(d_filter, d_filter)
        
        self.layer5 = nn.Linear(d_filter + d_input, d_filter)
        self.layer6 = nn.Linear(d_filter, d_filter)
        self.layer7 = nn.Linear(d_filter, d_filter)
        self.layer8 = nn.Linear(d_filter, d_filter) 
        
        self.sigma_out = nn.Linear(d_filter, 1)
        self.rgb_layer1 = nn.Linear(d_filter + d_direction, 128)
        self.rgb_out = nn.Linear(128, 3)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, xyzs, r_ds):
        num_rays, num_samples, _ = xyzs.shape
        x_enc = self.pe_xyz(xyzs) 
        d_enc = self.pe_dir(r_ds) 
        d_enc = d_enc.unsqueeze(1).expand(-1, num_samples, -1) 
        
        h = self.relu(self.layer1(x_enc))
        h = self.relu(self.layer2(h))
        h = self.relu(self.layer3(h))
        h = self.relu(self.layer4(h))
        h = torch.cat([h, x_enc], dim=-1)
        h = self.relu(self.layer5(h))
        h = self.relu(self.layer6(h))
        h = self.relu(self.layer7(h))
        h = self.layer8(h) 
        
        sigma = self.relu(self.sigma_out(h)) 
        h_color = torch.cat([h, d_enc], dim=-1)
        rgb = self.relu(self.rgb_layer1(h_color))
        rgb = self.sigmoid(self.rgb_out(rgb))
        return rgb, sigma
