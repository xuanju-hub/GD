import pandas as pd
import numpy as np
import torch
import torch_geometric as pyg
from tqdm.auto import *

from smartgd.model import Generator, Discriminator
from smartgd.data import GraphDrawingData
from smartgd.datasets import  RomeDataset
from smartgd.metrics import Stress, Crossings
from smartgd.transformations import Compose, Center, NormalizeRotation, RescaleByStress
from smartgd.criteria import RGANCriterion

model_name = "ckpts-xin"
batch_size = 16
start_epoch = 1492
max_epoch = 2000
max_lr = 0.01
min_lr = 0.0001
wr_period = 200

device = "cpu"
for backend, device_name in {
    # torch.backends.mps: "mps",
    torch.cuda: "cuda",
}.items():
    if backend.is_available():
        device = device_name

GraphDrawingData.set_optional_fields([
    "edge_pair_metaindex",
    # "face",
    # "rng"
])
dataset = RomeDataset(
    index=pd.read_csv("assets/rome_index.txt", header=None)[0],
)
init_layouts = np.load("assets/layouts/pmds.npy", allow_pickle=True)
target_layouts = np.load("assets/layouts/pmds.npy", allow_pickle=True)


generator = Generator(
    params=Generator.Params(
        num_blocks=11,
        block_depth=3,
        block_width=8,
        block_output_dim=8,
        edge_net_depth=2,
        edge_net_width=16,
        edge_attr_dim=2,
        node_attr_dim=2,
    ),
).to(device)
discriminator = Discriminator(
    params=Discriminator.Params(
        num_layers=9,
        hidden_width=16,
        edge_net_shared_depth=8,
        edge_net_embedded_depth=8,
        edge_net_width=64,
        edge_attr_dim=2
    )
).to(device)
canonicalizer = Compose(
    Center(),
    NormalizeRotation(),
    RescaleByStress(),
)
metrics = {
    # Stress(): 1,
    Crossings(): 1,
    # dgd.EdgeVar(): 0,
    # dgd.Occlusion(): 0,
    # dgd.IncidentAngle(): 0,
    # dgd.TSNEScore(): 0,
}
tie_break_metrics = {
    Stress(): 1
}
criterion = RGANCriterion()
gen_optim = torch.optim.AdamW(generator.parameters(), lr=max_lr)
dis_optim = torch.optim.AdamW(discriminator.parameters(), lr=max_lr)


if start_epoch:
    generator.load_state_dict(torch.load(f"./{model_name}/generator_{start_epoch-1}.pt"))
    discriminator.load_state_dict(torch.load(f"./{model_name}/discriminator_{start_epoch-1}.pt"))
    gen_optim.load_state_dict(torch.load(f"./{model_name}/gen_optim_{start_epoch-1}.pt"))
    dis_optim.load_state_dict(torch.load(f"./{model_name}/dis_optim_{start_epoch-1}.pt"))
    target_layouts = torch.load(f"./{model_name}/layouts_{start_epoch-1}.pt")
gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(gen_optim, T_0=wr_period, eta_min=min_lr, last_epoch=start_epoch-1)
dis_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(dis_optim, T_0=wr_period, eta_min=min_lr, last_epoch=start_epoch-1)
def create_dataloaders():
    datalist = list(dataset)
    for i, data in enumerate(datalist):
        data.pos = torch.tensor(init_layouts[i]).float()
        data.target_pos = torch.tensor(target_layouts[i]).float()
        data.fake_pos = torch.zeros_like(data.target_pos)
        data.index = i
    train_loader = pyg.loader.DataLoader(datalist[:10000], batch_size=batch_size, shuffle=True)
    val_loader = pyg.loader.DataLoader(datalist[11000:], batch_size=batch_size, shuffle=False)
    test_loader = pyg.loader.DataLoader(datalist[10000:11000], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
def generate_init_pos(batch):
    # pos = torch.rand_like(batch.pos)
    pos = canonicalizer(
        pos=batch.pos,
        apsp=batch.apsp_attr,
        edge_index=batch.perm_index,
        batch_index=batch.batch,
    )
    return pos

def get_edge_features(all_pair_shortest_path):
    return torch.cat([
        all_pair_shortest_path[:, None],
        1 / all_pair_shortest_path[:, None].square()
    ], dim=-1)

def evaluate(pos, batch):
    score = 0
    for c, w in metrics.items():
        score += w * c(pos, batch.perm_index, batch.apsp_attr, batch.batch, batch.edge_pair_index)
    return score

def evaluate_tie(pos, batch):
    score = 0
    for c, w in tie_break_metrics.items():
        score += w * c(pos, batch.perm_index, batch.apsp_attr, batch.batch, batch.edge_pair_index)
    return score

def forward(batch, train=False):
    edge_attr = get_edge_features(batch.apsp_attr)
    pred = generator(
        init_pos=generate_init_pos(batch),
        edge_index=batch.perm_index,
        edge_attr=edge_attr,
        batch_index=batch.batch,
    )
    fake_pos = canonicalizer(pred, batch.apsp_attr, batch.perm_index, batch.batch)
    fake_score = evaluate(fake_pos, batch)
    output = {
        'fake_pos': fake_pos,
        'fake_score': fake_score,
    }
    if train:
        fake_tie_break_score = evaluate_tie(fake_pos, batch)
        fake_logits = discriminator(
            pos=fake_pos,
            edge_index=batch.perm_index,
            edge_attr=edge_attr,
            batch_index=batch.batch,
        )
        real_pos = canonicalizer(batch.target_pos, batch.apsp_attr, batch.perm_index, batch.batch)
        real_score = evaluate(real_pos, batch)
        real_tie_break_score = evaluate_tie(real_pos, batch)
        real_logits = discriminator(
            pos=real_pos,
            edge_index=batch.perm_index,
            edge_attr=edge_attr,
            batch_index=batch.batch,
        )
        fake_better = (fake_score < real_score) | ((fake_score == real_score) & (fake_tie_break_score < real_tie_break_score))
        good_logits = torch.cat([
            fake_logits[fake_better],
            real_logits[~fake_better],
        ])
        bad_logits = torch.cat([
            real_logits[fake_better],
            fake_logits[~fake_better],
        ])
        output |= {
            'fake_logits': fake_logits,
            'real_pos': real_pos,
            'real_score': real_score,
            'real_logits': real_logits,
            'good_logits': good_logits,
            'bad_logits': bad_logits,
            'fake_better': fake_better,
        }
    return output


for epoch in range(start_epoch, max_epoch):
    train_loader, val_loader, test_loader = create_dataloaders()

    generator.train()
    discriminator.train()
    gen_losses = []
    dis_losses = []
    fake_scores = []
    real_scores = []
    replacements = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)

        generator.zero_grad()
        discriminator.zero_grad()
        output = forward(batch, train=True)
        dis_loss = criterion(encourage=output['good_logits'], discourage=output['bad_logits'])
        dis_loss.backward()
        dis_optim.step()

        generator.zero_grad()
        discriminator.zero_grad()
        output = forward(batch, train=True)
        gen_loss = criterion(encourage=output['bad_logits'], discourage=output['good_logits'])
        gen_loss.backward()
        gen_optim.step()

        gen_losses.append(gen_loss.item())
        dis_losses.append(dis_loss.item())
        fake_scores += output['fake_score'].tolist()
        real_scores += output['real_score'].tolist()

        batch.fake_pos = output['fake_pos']
        for fake_better, data in zip(output['fake_better'], batch.to_data_list()):
            if fake_better:
                target_layouts[data['index']] = data['fake_pos'].detach().cpu().numpy()
                replacements += 1

    gen_scheduler.step()
    dis_scheduler.step()
    print(f'[Epoch {epoch}] Learning Rates:\tgen={gen_scheduler.get_last_lr()[0]}\tdis={dis_scheduler.get_last_lr()[0]}')
    print(f'[Epoch {epoch}] Train Loss:\tgen={np.mean(gen_losses)}\tdis={np.mean(dis_losses)}')
    print(f'[Epoch {epoch}] Train Score:\t{np.mean(fake_scores)}/{np.mean(real_scores)}')
    print(f'[Epoch {epoch}] Replacements:\t{replacements}')

    with torch.no_grad():
        generator.eval()
        discriminator.eval()
        scores = []
        for batch in tqdm(test_loader, disable=True):
            batch = batch.to(device)
            output = forward(batch)
            scores += output['fake_score'].tolist()

        print(f'[Epoch {epoch}] Test Score:\t{np.mean(scores)}')
    torch.save(generator.state_dict(), f"./{model_name}/generator_{epoch}.pt")
    torch.save(discriminator.state_dict(), f"./{model_name}/discriminator_{epoch}.pt")
    torch.save(gen_optim.state_dict(), f"./{model_name}/gen_optim_{epoch}.pt")
    torch.save(dis_optim.state_dict(), f"./{model_name}/dis_optim_{epoch}.pt")
    torch.save(target_layouts, f"./{model_name}/layouts_{epoch}.pt")

## test
# _, _, test_loader = create_dataloaders()
# for epoch in range(0, 2000):
#
#     generator.load_state_dict(torch.load(f"/content/drive/MyDrive/smartgd/{model_name}/generator_{epoch}.pt"))
#     discriminator.load_state_dict(torch.load(f"/content/drive/MyDrive/smartgd/{model_name}/discriminator_{epoch}.pt"))
#
#     with torch.no_grad():
#         generator.eval()
#         discriminator.eval()
#         scores = []
#         for batch in tqdm(test_loader, disable=True):
#             batch = batch.to(device)
#             output = forward(batch)
#             scores += output['fake_score'].tolist()
#
#         print(f'[Epoch {epoch}] Test Score:\t{np.mean(scores)}')