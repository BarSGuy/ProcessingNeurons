
from torch_geometric.nn import  GINEConv
import torch_scatter
import torch
import torch.nn as nn
import torch_geometric.data as data
import torch_geometric as pyg
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm
import torch_geometric.nn as gnnn
import wandb
import numpy as np
from dataset_creator import *
from model import *
##############################################################
###################### dataset ###############################
##############################################################

def get_zinc_dataloader(batch_size=128):
    zinc_dataloader = {
                name: data.DataLoader(
                    pyg.datasets.ZINC(
                        split=name,
                        subset=True,
                        root='zinc',
                        ),
                    batch_size=batch_size,
                    num_workers=4,
                    shuffle=(name == "train"),
                )
                for name in ["train", "val", "test"]
            }
    num_elements_in_target = 1
    return zinc_dataloader, num_elements_in_target





##############################################################
######################## model ###############################
##############################################################


class CustomGINE(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, layernorm, track_running_stats, num_edge_emb=4):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(
                emb_dim, track_running_stats=track_running_stats)
            if not layernorm
            else torch.nn.LayerNorm(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = GINEConv(nn=mlp, train_eps=True)
        self.edge_embedding = torch.nn.Embedding(
            num_embeddings=num_edge_emb, embedding_dim=in_dim
        )

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_embedding(edge_attr))

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.layer.reset_parameters()


class GNNnetwork(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        feature_encoder,
        GNNConv,
        layernorm=False,
        add_residual=False,
        track_running_stats=True,
        num_tasks: int = None,
    ):

        super().__init__()

        self.emb_dim = emb_dim

        self.feature_encoder = feature_encoder

        self.gnn_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GNNConv(
                    emb_dim if i != 0 else in_dim,
                    emb_dim,
                    layernorm,
                    track_running_stats=track_running_stats,
                )
            )
            self.bn_layers.append(
                torch.nn.BatchNorm1d(
                    emb_dim, track_running_stats=track_running_stats)
                if not layernorm
                else torch.nn.LayerNorm(emb_dim)
            )
        self.add_residual = add_residual
        self.final_layers = None
        if num_tasks is not None:
            emb_dim = emb_dim
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 * emb_dim,
                                out_features=num_tasks),
            )

    def forward(self, batched_data):
        x, edge_index, edge_attr = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
        )
        x = self.feature_encoder(x)  # (g, node, _)

        for gnn, bn in zip(self.gnn_layers, self.bn_layers):
            h = torch.relu(bn(gnn(x, edge_index, edge_attr)))

            if self.add_residual:
                x = h + x
            else:
                x = h

        x_pool = gnnn.global_mean_pool(x, batched_data.batch)
        # x = torch_scatter.segment_csr(
        #     src=x,
        #     indptr=batched_data.batch,
        #     reduce="mean",
        # )  # : (g, node, _) -> (g, _)
        out = self.final_layers(x_pool)

        return out

def get_model():
    model = GNNnetwork(num_layers=6,
                       in_dim=128,
                       emb_dim=128,
                       feature_encoder=AtomEncoder(128),
                       GNNConv=CustomGINE,
                       layernorm=False,
                       add_residual=True,
                       track_running_stats=True,
                       num_tasks=1)
    return model




##############################################################
########################### loss #############################
##############################################################

def train_loop(model, loader, critn, optim, epoch, device, task='regression'):
    model.train()
    loss_list = []
    pbar = tqdm(loader, total=len(loader))
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        optim.zero_grad()
        if task == 'classification':
            is_labeled = batch.y == batch.y
            pred = model(batch)  # pred is 128 x 12

            labeled_y = batch.y.to(torch.float32)[is_labeled]
            labeled_pred = pred.to(torch.float32)[is_labeled]
            # TODO: make sure this 2 lines are ok
            labeled_y = labeled_y.reshape(-1)
            labeled_pred = labeled_pred.reshape(-1)

            assert labeled_y.shape == labeled_pred.shape
            loss = critn(labeled_pred, labeled_y)
        elif task == 'regression':
            pred = model(batch).view(batch.y.shape)
            loss = critn(pred, batch.y)
        else:
            raise ValueError(
                f"Invalid task type: {task}. Expected 'regression' or 'classification'.")

        loss.backward()
        optim.step()

        loss_list.append(loss.item())
        pbar.set_description(
            f"Epoch {epoch} Train Step {i}: Loss = {loss.item()}")
        # wandb.log({"Epoch": epoch, "Train Step": i, "Train Loss": loss.item()})

    return loss_list


def average_dicts(*input_dicts):
    # Check if there is at least one dictionary
    if len(input_dicts) == 0:
        raise ValueError("No dictionaries provided.")

    # If only one dictionary is provided, return it as is
    if len(input_dicts) == 1:
        return input_dicts[0]

    # Check if all dictionaries have the same keys
    keys = set(input_dicts[0].keys())
    if not all(keys == set(d.keys()) for d in input_dicts):
        raise ValueError("All dictionaries must have the same keys.")

    averaged_dict = {}

    num_dicts = len(input_dicts)
    for key in keys:
        # Sum the tensors from all dictionaries for the same key
        total_tensor = sum(d[key] for d in input_dicts)

        # Calculate the average
        averaged_tensor = total_tensor / num_dicts
        averaged_dict[key] = averaged_tensor

    return averaged_dict


def eval_loop(model, loader, eval, device, average_over=1, task='regression'):
    model.eval()
    # TODO: for average_over > 0 this works only if test/val dataloader doesn't shuffle!!!
    input_dict_for_votes = []
    for vote in range(average_over):
        pbar = tqdm(loader, total=len(loader),
                    desc=f"Vote {vote + 1} out of {average_over} votes")
        pred, true = [], []
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            with torch.no_grad():
                if task == 'classification':
                    model_pred = model(batch)
                    true.append(batch.y.view(model_pred.shape).detach().cpu())
                    pred.append(model_pred.detach().cpu())
                elif task == 'regression':
                    true.append(batch.y)
                    pred.append(model(batch).view(batch.y.shape))
                else:
                    raise ValueError(
                        f"Invalid task type: {task}. Expected 'regression' or 'classification'.")

        input_dict = {
            "y_true": torch.cat(true, dim=0),
            "y_pred": torch.cat(pred, dim=0)
        }

        input_dict_for_votes.append(input_dict)
    average_votes_dict = average_dicts(*input_dict_for_votes)
    input_dict = average_votes_dict
    metric = eval.eval(input_dict)
    # TODO: assuming 'metric' is a dictionary with 1 single key!
    metric = list(metric.values())[0]
    return metric


class ZincLEvaluator(nn.L1Loss):
    def forward(self, input_dict):
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        return super().forward(y_pred, y_true)

    def eval(self, input_dict):
        L1_val = self.forward(input_dict)
        L1_val_dict = {
            'L1loss': L1_val.item()
        }
        return L1_val_dict


def update_best_metrics(best_metrics, val_metric, test_metric, epoch, goal='minimize'):
    assert goal in [
        "minimize", "maximize"], "Invalid goal: must be either 'minimize' or 'maximize'"
    if (goal == "minimize" and val_metric < best_metrics["val_loss"]) or \
       (goal == "maximize" and val_metric > best_metrics["val_loss"]):
        best_metrics.update({
            "val_loss": val_metric,
            "test_loss": test_metric,
            "epoch": epoch
        })
    return best_metrics


def initialize_best_metrics(goal='minimize'):
    assert goal in [
        "minimize", "maximize"], "Invalid goal: must be either 'minimize' or 'maximize'"
    return {
        "val_loss": float('inf') if goal == "minimize" else float('-inf'),
        "test_loss": float('inf') if goal == "minimize" else float('-inf'),
        "epoch": -1
    }
    
def log_wandb(epoch, optim, loss_list, val_metric, test_metric, best_metrics):
    lr = optim.param_groups[0]['lr']
    wandb.log({
        "Epoch": epoch,
        "Train Loss": np.mean(loss_list),
        "Val Loss": val_metric,
        "Test Loss": test_metric,
        "Learning Rate": lr,  # Log the learning rate
        # unpack best metrics into the lognv
        **{f"best_{key}": value for key, value in best_metrics.items()}
    })
    
def Train_GIN_zinc12k():
    wandb.init(settings=wandb.Settings(
        start_method='thread'), project="neurons")

    goal = "minimize"
    device = torch.device(f"cuda:0")
    dataloader, num_target = get_zinc_dataloader()
    task = "regression"
    model = get_model()
    model = model.to(device)
    eval = ZincLEvaluator()
    optim = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0)
    sched = torch.optim.lr_scheduler.StepLR(optim,
                                            # mode='min',
                                            step_size=300,
                                            gamma=0.5,
                                            # patience=300,
                                            # verbose=True
                                            )
    critn = nn.L1Loss()
    wandb.watch(model)

    best_metrics = initialize_best_metrics(goal=goal)
    pbar = tqdm(range(1000))
    for epoch in pbar:
        loss_list = train_loop(
            model=model, loader=dataloader["train"], critn=critn, optim=optim, epoch=epoch, device=device, task=task)
        val_metric = eval_loop(
            model=model, loader=dataloader["val"], eval=eval, device=device, average_over=1, task=task)
        test_metric = eval_loop(
            model=model, loader=dataloader["test"], eval=eval, device=device, average_over=1, task=task)
        best_metrics = update_best_metrics(
            best_metrics=best_metrics, val_metric=val_metric, test_metric=test_metric, epoch=epoch, goal=goal)
        log_wandb(epoch=epoch, optim=optim, loss_list=loss_list, val_metric=val_metric,
                  test_metric=test_metric, best_metrics=best_metrics)
        sched.step()
    # Assuming 'model' is your trained model
    torch.save(model.state_dict(), 'gin_0_001.pth')
    

##############################################################
#################### Activation model ########################
##############################################################

def get_activation_dataloader(type="val"):
    device = torch.device(f"cuda:0")
    model = get_model()
    model = model.to(device)
    model.load_state_dict(torch.load('gin_0_001.pth'))
    dataloader, num_target = get_zinc_dataloader(batch_size=1)
    zinc12k_dataset = dataloader[type]
    model.device = device
    # Assuming `model` is your pre-trained model and `zinc12k_dataset` is your original dataset
    activation_dataset = ActivationDataset(
        './zinc_' + type, model, zinc12k_dataset)
    activation_dataloader = create_custom_dataloader(
        activation_dataset, batch_size=7)
    return activation_dataloader

def get_activation_model():
    model = NeuronArchitecture(d=128, L=6, d_hidden=4, d_final=1, 
                               num_layers=2)
    return model

def train_activation_model():
    wandb.init(settings=wandb.Settings(
        start_method='thread'), project="neurons")

    goal = "minimize"
    device = torch.device(f"cuda:0")
    dataloader_train = get_activation_dataloader(type="val")
    dataloader_test = get_activation_dataloader(type="test")
    task = "regression"
    model = get_activation_model()
    model = model.to(device)
    eval = ZincLEvaluator()
    optim = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0)
    sched = torch.optim.lr_scheduler.StepLR(optim,
                                            # mode='min',
                                            step_size=50,
                                            gamma=0.5,
                                            # patience=300,
                                            # verbose=True
                                            )
    critn = nn.L1Loss()
    wandb.watch(model)

    best_metrics = initialize_best_metrics(goal=goal)
    pbar = tqdm(range(1000))
    for epoch in pbar:
        loss_list = train_loop(
            model=model, loader=dataloader_train, critn=critn, optim=optim, epoch=epoch, device=device, task=task)
        test_metric = eval_loop(
            model=model, loader=dataloader_test, eval=eval, device=device, average_over=1, task=task)
        best_metrics = update_best_metrics(
            best_metrics=best_metrics, val_metric=test_metric, test_metric=test_metric, epoch=epoch, goal=goal)
        log_wandb(epoch=epoch, optim=optim, loss_list=loss_list, val_metric=test_metric,
                  test_metric=test_metric, best_metrics=best_metrics)
        sched.step()

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    train_activation_model()
    # Train_GIN_zinc12k()
    
    
    exit()
    device = torch.device(f"cuda:0")
    model = get_model()
    model = model.to(device)
    # Load the state dictionary
    model.load_state_dict(torch.load('gin_0_001.pth'))
    dataloader, num_target = get_zinc_dataloader(batch_size=1)
    zinc12k_dataset = dataloader["val"]
    model.device = device
    # Assuming `model` is your pre-trained model and `zinc12k_dataset` is your original dataset
    activation_dataset = ActivationDataset('.', model, zinc12k_dataset)
    activation_dataloader = create_custom_dataloader(activation_dataset)
    
    
    
    model = NeuronArchitecture(d=128, L=6, d_hidden=32, d_final=1, num_layers=4 )
    model = model.to(device)
    
    for batch in activation_dataloader:
        batch = batch.to(device)
        model(batch)
        # batch.activation_idx = batch.activation_idx + \
        #     (batch.num_layers[0] * batch.batch).reshape(-1,1)
        print("guy")
    exit()
    # Register hooks
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks on GNN layers and batch normalization layers
    for i, (gnn_layer, bn_layer) in enumerate(zip(model.gnn_layers, model.bn_layers)):
        gnn_layer.register_forward_hook(get_activation(f'gnn_layer_{i}'))
        bn_layer.register_forward_hook(get_activation(f'bn_layer_{i}'))

    # Register hook on the final layers if needed
    for i, layer in enumerate(model.final_layers):
        layer.register_forward_hook(get_activation(f'final_layer_{i}'))

    dataloader, num_target = get_zinc_dataloader(batch_size=1)
    model.eval()
    for batch in dataloader["val"]:
        batch = batch.to(device)
        with torch.no_grad():
            x = model(batch)
            # Print the activations
            for layer_name, activation in activations.items():
                print(f'{layer_name} activation shape: {activation.shape}')
            exit()
