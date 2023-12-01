import torch
import torch.nn.functional as F
from EinsumNetwork import Graph, EinsumNetwork
import os


class EinsumExperiment:
    def __init__(self, device: str, in_dim: int, out_dim: int):
        self.max_num_epochs = 5
        self.batch_size = 100
        self.device = device

        depth = 3
        num_repetitions = 20
        K = 10
        online_em_frequency = 1
        online_em_stepsize = 0.05
        exponential_family = EinsumNetwork.NormalArray
        exponential_family_args = {"min_var": 1e-6, "max_var": 0.1}

        self.graph = Graph.random_binary_trees(
            num_var=in_dim, depth=depth, num_repetitions=num_repetitions
        )

        args = EinsumNetwork.Args(
            num_var=in_dim,
            num_dims=1,
            num_classes=out_dim,
            num_sums=K,
            num_input_distributions=K,
            exponential_family=exponential_family,
            exponential_family_args=exponential_family_args,
            online_em_frequency=online_em_frequency,
            online_em_stepsize=online_em_stepsize,
            # use_em=False
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

    def train_eval(
        self,
        train: torch.Tensor,
        target_train: torch.Tensor,
        test: torch.Tensor,
        target_test: torch.Tensor,
    ):
        train = train.to(self.device)
        target_train = target_train.to(self.device)
        test = test.to(self.device)
        target_test = target_test.to(self.device)

        random_input = torch.rand(test.shape[0], test.shape[1]).to(self.device)
        # TODO: Can not get training via SGD to work...
        # optimizer = torch.optim.Adam(self.einet.parameters(), lr=0.2)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     milestones=[int(self.max_num_epochs * 0.3), int(self.max_num_epochs * 0.5), int(self.max_num_epochs * 0.7), int(self.max_num_epochs * 0.9)], optimizer=optimizer, gamma=0.5)
        for epoch_count in range(self.max_num_epochs):
            if epoch_count % 2 == 0:
                # evaluate
                train_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, train)
                test_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, test)
                random_input_ll = EinsumNetwork.eval_loglikelihood_batched(
                    self.einet, random_input
                )

                print(
                    "[{}]   train LL {}  test LL {} random_input LL {}".format(
                        epoch_count,
                        train_ll / train.shape[0],
                        test_ll / test.shape[0],
                        random_input_ll / random_input.shape[0],
                    )
                )

                print(
                    "test accuracy: ",
                    EinsumNetwork.eval_accuracy_batched(
                        self.einet, test, target_test, self.batch_size
                    ),
                )

            # train
            idx_batches = torch.randperm(train.shape[0]).split(self.batch_size)
            for batch_count, idx in enumerate(idx_batches):
                # optimizer.zero_grad()
                batch_x = train[idx, :]
                outputs = self.einet.forward(batch_x)
                ll_sample = EinsumNetwork.log_likelihoods(outputs, target_train[idx])
                log_likelihood = ll_sample.sum()
                objective = log_likelihood
                objective.backward()
                # loss = F.cross_entropy(outputs, target_train[idx])
                # loss.backward()
                # optimizer.step()
                self.einet.em_process_batch()
            self.einet.em_update()
            # lr_scheduler.step()
        self.store(".")

    def eval(self, test: torch.Tensor, target_test: torch.Tensor, name: str):
        test = test.to(self.device)
        target_test = target_test.to(self.device)
        test_ll_class = EinsumNetwork.eval_loglikelihood_batched(
            self.einet, test, target_test
        )
        test_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, test)
        test_acc = EinsumNetwork.eval_accuracy_batched(
            self.einet, test, target_test, self.batch_size
        )
        print(
            f"{name} classification LL: {test_ll_class / test.shape[0]}, LL: {test_ll / test.shape[0]}, Accuracy: {test_acc}"
        )

    def explain_mpe(self, test: torch.Tensor, exp_vars: list, name: str):
        # set all explaining features to 1
        self.einet.set_marginalization_idx(exp_vars)
        test = test.to(self.device)
        mpe = self.einet.mpe(x=test)
        self.einet.set_marginalization_idx([])

        return mpe[:, exp_vars]

    def explain_ll(self, test: torch.Tensor, exp_vars: list, name: str):
        test = test.to(self.device)
        # set all explaining features to 1
        full_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, test)
        full_ll /= test.shape[0]
        self.einet.set_marginalization_idx(exp_vars)
        marginal_ll = EinsumNetwork.eval_loglikelihood_batched(self.einet, test)
        marginal_ll /= test.shape[0]
        self.einet.set_marginalization_idx([])

        return full_ll, marginal_ll

    def draw_sample(self, num_samples: int):
        samples = self.einet.sample(num_samples=num_samples).cpu().numpy()
        samples = samples[:, : 22 * 22]
        return samples.reshape((-1, 22, 22))
