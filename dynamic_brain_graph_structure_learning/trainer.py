import os, sys
from typing import Any, List, Mapping
from absl import logging
import torch as pt
import torch.nn.functional as F
import numpy as np
from tqdm import trange

from config import get_cfg
from dbgs_learner import DBGSLearner
from utils import get_t_repetition

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)


class Trainer():

    def __init__(self,
                 cfg: Mapping[str, Any], masking_generator) -> None:

        """
        Train dbgsl

        Args:
            
        """

        self.cfg = cfg
        self.masking_generator = masking_generator
        self.cfg.t_repetition = get_t_repetition(cfg)
        self.model = DBGSLearner(cfg)
        if os.path.exists(cfg.state_dict_path):
            self.model.load_state_dict(pt.load(cfg.state_dict_path, map_location=f'{cfg.device}'))
        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def train_step(self):

        """
        Take a single optimization step on a single batch
        """

        self.optimizer.zero_grad()
        # target = pt.rand(self.cfg.batch_size, 1)
        data_path = '/home/matteoc/graphs-nn/data/hcp/raw'
        file_name = '100206_0.npy'
        file_path = os.path.join(data_path, file_name)
        x = np.load(file_path)[:, :self.cfg.T]
        tube_mask = self.masking_generator()
        shuffled = np.random.permutation(tube_mask.reshape(x.T.shape).T)
        masked_data = x * shuffled  
        x = pt.tensor(x.reshape(1, x.shape[0], x.shape[1]))
        x = x.float()
        masked_data = pt.tensor(masked_data.reshape(1, masked_data.shape[0], masked_data.shape[1])) 
        out = self.model(masked_data)   # out = self.model(masked_data)
        out = out.float()
        # print("NAN --> ", pt.isnan(out))
        instance_loss = pt.nn.MSELoss()
        loss = instance_loss(out, x)
        # loss = pt.nn.MSELoss(out, x)
        loss.backward()
        self.optimizer.step()

        return loss

    def post_train(self, losses: List) -> None:
        pass
        # pt.save(self.model.state_dict(), self.cfg.state_dict_path)
        # pickle.dump(losses, open(f'{self.cfg.train_result_path}', 'wb'))

    def train(self) -> Mapping[str, Any]:

        self.model.train()
        losses = []

        for t in trange(self.cfg.n_episodes):

            loss = self.train_step()
            losses.append(loss.item())
            
            if t % 50 == 0:
                logging.info(f'iteration: {t} --- loss: {loss.item()}')

        self.post_train(losses)