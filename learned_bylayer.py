import argparse
import os
import time
import logging


import torch
from timm import create_model
from timm.data import create_dataset, create_loader, resolve_data_config

from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('/a/home/cc/students/cs/tomerporian/timm_averaging/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('/a/home/cc/students/cs/tomerporian/timm_averaging/results_imgnet_avg/23-08-03-ImageNet-rsn50-wd-and-coeffs-compute/002_23-08-03-ImageNet-rsn50-wd-and-coeffs-compute+lr=0.24+wei_dec=3.33e-05'),
        help="Where the models are.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()

# Utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class AlphaWrapper(torch.nn.Module):
    def __init__(self, paramslist, model, names):
        super(AlphaWrapper, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        ralpha = torch.ones(len(paramslist))
        ralpha = torch.nn.functional.softmax(ralpha)
        self.alpha_raw = torch.nn.Parameter(ralpha)
        self.beta = torch.nn.Parameter(torch.tensor(1.))

    def alpha(self):
        return torch.nn.functional.softmax(self.alpha_raw)

    def forward(self, inp):
        alph = self.alpha()
        params = tuple(sum(alph[i] * p for i, p in enumerate(paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        out = self.model(inp)
        return self.beta * out

def get_imagenet_acc(loader_test):
    with torch.no_grad():
        correct = 0.
        n = 0
        end = time.time()
        for i, batch in enumerate(loader_test):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            data_time = time.time() - end
            end = time.time()
            logits = alpha_model(inputs)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            y = labels
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            batch_time = time.time() - end
            percent_complete = 100.0 * i / len(loader_test)
            if ( i % 10 ) == 0:
                print(
                    f"Train Epoch: {0} [{percent_complete:.0f}% {i}/{len(loader_test)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            end = time.time()
        acc = correct / float(n)
        print('Top-1', acc)
    return acc


if __name__ == '__main__':
    print("parse arguments...")
    args = parse_arguments()
    NUM_MODELS = 250

    model_paths = [os.path.join(args.model_location, f'checkpoint-{i}.pt') for i in range(NUM_MODELS)]
    print("Creating base model...")
    base_model = create_model('resnet50', pretrained=False)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda')
    print("setting up data loaders...")

    # data_config = resolve_data_config(vars(args), model=base_model, verbose=True)
    dataset_train = create_dataset(
        'imagenet',
        root=args.data_location,
        split='train',
        is_training=True,
        class_map='',
        download=False,
        batch_size=args.batch_size,
        seed=42,
    )

    dataset_eval = create_dataset(
        'imagenet',
        root=args.data_location,
        split='validation',
        is_training=False,
        class_map='',
        download=False,
        batch_size=args.batch_size,
    )
    loader_train = create_loader(
        dataset_train,
        batch_size=args.batch_size,
        input_size=(3, 224, 224),
        is_training=True,
        num_workers=args.workers,
        device=device
    )
    loader_test = create_loader(
        dataset_eval,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=False,
        num_workers=args.workers,
        device=device
    )
    
    print("loading models...")
    sds = [torch.load(cp, map_location='cpu') for cp in model_paths]
    feature_dim = sds[0]['fc.weight'].shape[1]
    num_classes = sds[0]['fc.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    model = model.to(device)

    _, names = make_functional(model)
    first = False

    paramslist = [tuple(v.detach().requires_grad_().cpu() for _, v in sd.items() if not isinstance(v, (torch.LongTensor, torch.cuda.LongTensor))) for i, sd in enumerate(sds)]
    torch.cuda.empty_cache()
    alpha_model = AlphaWrapper(paramslist, model, names)


    print(alpha_model.alpha())
    print(len(list(alpha_model.parameters())))

    lr = 0.05
    epochs = 5

    optimizer = torch.optim.AdamW(alpha_model.parameters(), lr=lr, weight_decay=0.)
    num_batches = len(loader_train)

    for epoch in range(epochs):
        end = time.time()
        for i, batch in enumerate(loader_train):
            step = i + epoch * num_batches
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()

            data_time = time.time() - end
            end = time.time()

            optimizer.zero_grad()

            out = alpha_model(inputs)

            loss = criterion(out, labels)
            
            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            percent_complete = 100.0 * i / len(loader_test)
            if ( i % 10 ) == 0:
                print(alpha_model.beta)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
                print(alpha_model.alpha())
            end = time.time()

    acc = get_imagenet_acc(loader_test)
    print('Accuracy is', 100 * acc)
    print(alpha_model.alpha())
    torch.save(
        {'alpha' : alpha_model.alpha(), 'beta' : alpha_model.beta}, 
        f'alphas_{lr}_{epochs}.pt'
    )
