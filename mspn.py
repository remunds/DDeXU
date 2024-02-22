import numpy as np
from spn.algorithms.MPE import mpe
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.Inference import log_likelihood


class MSPN_utils:
    def __init__(self, backbone, mspn, explaining_vars):
        self.backbone = backbone
        self.spn = mspn
        self.explaining_vars = explaining_vars

    def eval_acc(self, data):
        class_data = data.copy()
        class_data[:, 0] = np.nan
        mpe_data = mpe(self.spn, class_data)
        acc = (mpe_data[:, 0] == data[:, 0]).mean()
        return acc

    def eval_ll_marg(self, data, return_all=False):
        """
        Returns LL(x) for each data point x
        """
        ll = log_likelihood(self.spn, data)
        if return_all:
            return ll
        else:
            return ll.mean()

    def explain_ll(self, data):
        ll_marg = self.eval_ll_marg(data)
        expl_ll = []
        for v in self.explaining_vars:
            data_clone = data.copy()
            data_clone[:, v] = np.nan  # explaining var
            ll = log_likelihood(self.spn, data_clone).mean()
            expl_ll.append(ll)
        return map(lambda x: x - ll_marg, expl_ll)

    def explain_mpe(self, data, return_all=False):
        """
        Returns MPE(x) for each data point x
        """
        mpe_data = data.copy()
        mpe_data[:, 0] = np.nan  # class
        mpe_data[:, self.explaining_vars] = np.nan

        mpe_expl = mpe(self.spn, mpe_data)
        mpe_expl = mpe_expl[:, self.explaining_vars]
        if return_all:
            return mpe_expl
        else:
            return mpe_expl.mean(axis=0)

    def create_data(self, ds, dl, device):
        embeddings = self.backbone.get_embeddings(dl, device)
        data = embeddings.cpu().detach().numpy()
        targets = np.array([t[1] for t in ds])
        # Put targets in first column
        data = np.concatenate([targets[:, None], data], axis=1)
        return data
