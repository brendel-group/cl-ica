"""Modified https://github.com/bethgelab/slow_disentanglement/blob/master/scripts/solver.py
and removed functions not needed for training with a contrastive loss."""

import os
import shutil
import torch
import torch.optim as optim
from torch.autograd import Variable
from kitti_masks.model import BetaVAE_H as BetaVAE
import losses


class Solver(object):
    def __init__(self, args, data_loader=None):
        self.ckpt_dir = args.ckpt_dir
        self.output_dir = args.output_dir
        self.data_loader = data_loader
        self.dataset = args.dataset
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.nc = args.num_channel
        params = []

        # for adam
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.net = BetaVAE(self.z_dim, self.nc, args.box_norm).to(self.device)
        self.optim = optim.Adam(
            params + list(self.net.parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

        self.ckpt_name = args.ckpt_name
        if False and self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.log_step = args.log_step
        self.save_step = args.save_step

        self.loss = losses.LpSimCLRLoss(
            p=args.p, tau=1.0, simclr_compatibility_mode=True
        )

    def train(self):
        self.net_mode(train=True)
        out = False  # whether to exit training loop
        failure = False  # whether training was stopped
        running_loss = 0
        log = open(os.path.join(self.output_dir, "log.csv"), "a", 1)
        log.write("Total Loss\n")

        while not out:
            for x, _ in self.data_loader:  # don't use label
                x = Variable(x.to(self.device))
                mu = self.net(x)
                z1_rec = mu[::2]
                z2_con_z1_rec = mu[1::2]
                z3_rec = torch.roll(z1_rec, 1, 0)
                vae_loss, _, _ = self.loss(
                    None, None, None, z1_rec, z2_con_z1_rec, z3_rec
                )
                running_loss += vae_loss.item()

                self.optim.zero_grad()
                vae_loss.backward()
                self.optim.step()

                self.global_iter += 1
                if self.global_iter % self.log_step == 0:
                    running_loss /= self.log_step
                    log.write("%.6f" % running_loss + "\n")

                    running_loss = 0

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint("last")

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        if failure:
            shutil.rmtree(self.ckpt_dir)

        return failure

    def save_checkpoint(self, filename, silent=True):
        model_states = {
            "net": self.net.state_dict(),
        }
        optim_states = {
            "optim": self.optim.state_dict(),
        }
        states = {
            "iter": self.global_iter,
            "model_states": model_states,
            "optim_states": optim_states,
        }

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode="wb+") as f:
            torch.save(states, f)
        if not silent:
            print(
                "=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter)
            )

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint["iter"]
            self.net.load_state_dict(checkpoint["model_states"]["net"])
            self.optim.load_state_dict(checkpoint["optim_states"]["optim"])
            print(
                "=> loaded checkpoint '{} (iter {})'".format(
                    file_path, self.global_iter
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError("Only bool type is supported. True or False")

        if train:
            self.net.train()
        else:
            self.net.eval()
