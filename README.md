# CIKM-2025-FedSTEP

This repository contains the implementation of FedSTEP, a modular federated learning framework with decoupled components for training orchestration, client simulation, model transmission, and communication efficiency monitoring.

# Prerequisites

- Python 3.5+
- PyTorch
- CUDA environment

# Directory Structure

1. `./FedBridge.py`: HTTP interface (Flask) for uploading/downloading models/configs, logging metrics.
2. `./Server.py`: Main federated server logic with async/sync training scheduler and aggregation.
3. `./Client.py`: Simulates multiple clients asynchronously connecting to the server.
4. `./Utils.py`: Common utilities: compression, decompression, serialization, config loading.
5. `./Model.py`: Model zoo: VGG11, ResNet18, AlexNet, LSTM, DNN with unified `get_model()` function.
6. `./Dataset.py`: Data loading and Dirichlet splitting for CIFAR, MNIST, Shakespeare, HARBox.
7. `./Config.yaml`: Global configuration file (server ports, model type, dataset, hyperparams).

# Run Federated Learning

Open three terminals to launch components in the following order:

1. Launch HTTP Server (FedBridge)
```
python ./FedBridge.py
```

2. Launch Federated Server
```
python ./Server.py
```

3. Launch Clients (Simulated Locally)
```
python ./Client.py
```

# Citation

If you use the simulator or some results in our paper for a published project, please cite our work by using the following bibtex entry:

```
@inproceedings{yan2025fedstep,
  title     = {FedSTEP: Asynchronous and Staleness-Aware Personalization for Efficient Federated Learning},
  author    = {Gang Yan, Jian Li and Wan Du},
  booktitle = {Proc. of ACM CIKM},
  year      = {2025}
}
```

