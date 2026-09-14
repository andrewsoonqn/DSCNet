"""Exercise production training-loop ordering with bounded fake epoch work."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dscnet.training import standard, optimized
from dscnet.evaluation import validation_comparisons as vc


class ValidationRetentionLoopTests(unittest.TestCase):
    def run_loop(self, module, amp=False, fail=False, gap=1, resume=False):
        events = []
        with TemporaryDirectory() as directory, ExitStack() as stack:
            args = SimpleNamespace(start_train_epoch=1, n_epochs=2, start_verify_epoch=1,
                verify_gap=gap, batch_size=1, lr=.001, min_lr=.00001, poly_decay_power=.9,
                beta=.1, min_beta=.01, beta_decay_power=1, rlr_factor=.5,
                rlr_threshold=.01, rlr_patience=5, rlr_cooldown=0, use_rlrop=False,
                if_retrain=not resume, log_name='test', Dir_Log=directory + '/',
                earlystop_threshold=0, earlystop_patience=10, use_earlystop=False,
                Image_Va_txt='images', Label_Va_txt='labels', Meanstd_path='normalization',
                save_path=directory, Dir_Weights=directory, model_name='latest.pt', model_name_max='best.pt')
            net = torch.nn.Linear(1, 1)
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
            stack.enter_context(patch.object(torch.amp, 'GradScaler', return_value=Mock()))
            for name in ['Dataloader', 'DataLoader', 'Get_logger', 'Close_logger']:
                stack.enter_context(patch.object(module, name))
            stack.enter_context(patch.object(module, 'train_epoch_amp' if amp else 'train_epoch', return_value=.1))
            predict_name = 'new_predict' if module is optimized else 'predict_amp' if amp else 'predict'
            prediction = stack.enter_context(patch.object(module, predict_name, side_effect=lambda *a: events.append('predict')))
            stack.enter_context(patch.object(module, 'summarize_predictions', return_value={'dice': .8}))
            stack.enter_context(patch.object(module, 'log_summary'))
            if resume:
                stack.enter_context(patch.object(module, '_resume_training', return_value=(1, .9) if module is optimized else (1, .9, .9, 0)))
            def save(net, args, name, *state):
                epoch = state[2] if module is optimized else state[3]
                torch.save({'epoch': epoch, 'state_dict': net.state_dict()}, Path(directory) / name)
                events.append(('save', name, epoch))
            stack.enter_context(patch.object(module, '_save_training_checkpoint', side_effect=save))
            def retain(args, epoch, is_best):
                self.assertEqual(torch.load(Path(directory) / args.model_name, weights_only=True)['epoch'], epoch)
                if is_best:
                    self.assertEqual(torch.load(Path(directory) / args.model_name_max, weights_only=True)['epoch'], epoch)
                events.append(('retain', epoch, is_best))
                if fail:
                    raise RuntimeError('retention failure after checkpoint')
            stack.enter_context(patch.object(module, 'record_validation', side_effect=retain))
            invoke = lambda: module.Train_net(net, args, torch.device('cpu'), None) if module is optimized else (module.Train_net_amp(net, args) if amp else module.Train_net(net, args))
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'retention failure'):
                    invoke()
                self.assertTrue((Path(directory) / args.model_name).is_file())
                self.assertTrue((Path(directory) / args.model_name_max).is_file())
            else:
                invoke()
            return events, prediction.call_count

    def test_all_training_loops_save_before_retention_and_no_extra_predictions(self):
        for module, amp in [(standard, False), (standard, True), (optimized, False)]:
            with self.subTest(module=module.__name__, amp=amp):
                events, calls = self.run_loop(module, amp)
                self.assertEqual(calls, 2)
                self.assertEqual(events[:4], ['predict', ('save', 'latest.pt', 1), ('save', 'best.pt', 1), ('retain', 1, True)])
                self.assertEqual(events[-2:], [('save', 'latest.pt', 2), ('retain', 2, False)])

    def test_all_loops_keep_checkpoint_on_retention_failure(self):
        for module, amp in [(standard, False), (standard, True), (optimized, False)]:
            with self.subTest(module=module.__name__, amp=amp):
                _, calls = self.run_loop(module, amp, fail=True)
                self.assertEqual(calls, 1)

    def test_optimized_gap_and_resume_do_not_invent_best_or_extra_validation(self):
        events, calls = self.run_loop(optimized, gap=2, resume=True)
        self.assertEqual(calls, 1)
        self.assertEqual([event for event in events if isinstance(event, tuple) and event[0] == 'retain'], [('retain', 2, False)])
        self.assertNotIn(('save', 'best.pt', 2), events)
