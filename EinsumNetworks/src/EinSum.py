import torch
import torch.nn.functional as F
from EinsumNetwork import Graph, EinsumNetwork
import os

class EinsumExperiment:
    def __init__(self, device: str, num_var: int):
        self.max_num_epochs = 5
        self.batch_size = 100
        self.device = device

        depth = 3
        num_repetitions = 20
        K = 10
        online_em_frequency = 1
        online_em_stepsize = 0.05
        exponential_family = EinsumNetwork.NormalArray
        exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

        self.graph = Graph.random_binary_trees(num_var=num_var, depth=depth, num_repetitions=num_repetitions)

        args = EinsumNetwork.Args(
                num_var=num_var,
                num_dims=1,
                num_classes=10,
                num_sums=K,
                num_input_distributions=K,
                exponential_family=exponential_family,
                exponential_family_args=exponential_family_args,
                online_em_frequency=online_em_frequency,
                online_em_stepsize=online_em_stepsize
                )

        einet = EinsumNetwork.EinsumNetwork(self.graph, args)
        einet.initialize()
        einet.to(device)
        self.einet = einet

    def store(self, save_dir: str):
        graph_file = os.path.join(save_dir, "einet.pc")
        Graph.write_gpickle(self.graph, graph_file)
        model_file = os.path.join(save_dir, "einet.mdl")
        torch.save(self.einet, model_file)
        print("Stored Einet")

    def load(self, save_dir: str):
        graph_file = os.path.join(save_dir, "einet.pc")
        self.graph = Graph.read_gpickle(graph_file)
        model_file = os.path.join(save_dir, "einet.mdl")
        self.einet = torch.load(model_file)
        self.einet.to(self.device)
        print("Loaded Einet")
    
    def train_eval(self, train: torch.Tensor, target_train: torch.Tensor, test: torch.Tensor, target_test: torch.Tensor):
        train = train.to(self.device)
        target_train = target_train.to(self.device)
        test = test.to(self.device)
        target_test = target_test.to(self.device)

        random_input = torch.rand(test.shape[0], test.shape[1]).to(self.device)
        for epoch_count in range(self.max_num_epochs):
            if epoch_count % 2 == 0:
                # evaluate
                train_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, train)
                test_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, test)
                random_input_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, random_input)

                print("[{}]   train LL {}  test LL {} random_input LL {}".format(epoch_count,
                                                                            train_ll / train.shape[0],
                                                                            test_ll / test.shape[0],
                                                                            random_input_ll / random_input.shape[0]))
                
                print("test accuracy: ", EinsumNetwork.eval_accuracy_batched(self.einet, test, target_test, self.batch_size))

            # train
            idx_batches = torch.randperm(train.shape[0]).split(self.batch_size)
            for batch_count, idx in enumerate(idx_batches):
                batch_x = train[idx, :]
                outputs = self.einet.forward(batch_x)
                ll_sample = EinsumNetwork.log_likelihoods(outputs, target_train[idx])
                log_likelihood = ll_sample.sum()
                objective = log_likelihood
                objective.backward()
                self.einet.em_process_batch()
            self.einet.em_update()

        self.store(".")

    def eval(self, test: torch.Tensor, target_test: torch.Tensor, name: str):
        test = test.to(self.device)
        target_test = target_test.to(self.device)
        test_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, test, target_test)
        test_acc = EinsumNetwork.eval_accuracy_batched(self.einet, test, target_test, self.batch_size)
        print(f"{name} LL: {test_ll / test.shape[0]}, {name} accuracy: {test_acc}")
