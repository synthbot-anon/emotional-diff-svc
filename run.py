import os
import random

import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from utils.hparams import set_hparams, hparams
from training.task.SVC_task import SVCTask
from training.task.base_task import BaseTask
from utils.pl_utils import BaseTrainer, LatestModelCheckpoint


set_hparams(print_hparams=False)


def run_task():
    assert hparams['task_cls'] == 'training.task.SVC_task.SVCTask'

    random.seed(hparams['seed'])
    np.random.seed(hparams['seed'])
    task = SVCTask()
    work_dir = hparams['work_dir']
    trainer = BaseTrainer(checkpoint_callback=LatestModelCheckpoint(
                                filepath=work_dir,
                                verbose=True,
                                monitor='val_loss',
                                mode='min',
                                num_ckpt_keep=hparams['num_ckpt_keep'],
                                save_best=hparams['save_best'],
                                period=1 if hparams['save_ckpt'] else 100000
                            ),
                            logger=TensorBoardLogger(
                                save_dir=work_dir,
                                name='lightning_logs',
                                version='lastest'
                            ),
                            gradient_clip_val=hparams['clip_grad_norm'],
                            val_check_interval=hparams['val_check_interval'],
                            row_log_interval=hparams['log_interval'],
                            max_updates=hparams['max_updates'],
                            num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams[
                                'validate'] else 10000,
                            accumulate_grad_batches=hparams['accumulate_grad_batches'])

    trainer.checkpoint_callback.task = task
    trainer.fit(task)


if __name__ == '__main__':
    run_task()

