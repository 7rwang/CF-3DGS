import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print("sys.path:", sys.path)

import unittest
from unittest.mock import MagicMock, patch
from trainer.cf3dgs_trainer import CFGaussianTrainer




class TestCFGaussianTrainer(unittest.TestCase):
    def setUp(self):
        self.data_root = ""
        self.model_cfg = MagicMock()
        self.pipe_cfg = MagicMock()
        self.optim_cfg = MagicMock()

        self.trainer = CFGaussianTrainer(
            data_root=self.data_root,
            model_cfg=self.model_cfg,
            pipe_cfg=self.pipe_cfg,
            optim_cfg=self.optim_cfg
        )
        
    def test_train_step_success(self):
        """测试 train_step 方法在正常情况下的行为"""
        # 模拟输入参数
        gs_render = MagicMock()
        viewpoint_cam = MagicMock()
        iteration = 1
        pipe = MagicMock()
        optim_opt = MagicMock()
        colors_preco = MagicMock()

        self.trainer.train_step = MagicMock(return_value="expected_result")

        # 调用 train_step 方法
        result = self.trainer.train_step(
            gs_render=gs_render,
            viewpoint_cam=viewpoint_cam,
            iteration=iteration,
            pipe=pipe,
            optim_opt=optim_opt,
            colors_preco=colors_preco
        )
        # 断言 train_step 被正确调用
        self.trainer.train_step.assert_called_once_with(
            gs_render=gs_render,
            viewpoint_cam=viewpoint_cam,
            iteration=iteration,
            pipe=pipe,
            optim_opt=optim_opt,
            colors_preco=colors_preco
        )
        # 断言返回值正确
        self.assertEqual(result, "expected_result", "train_step 应返回预期结果")

    def test_train_step_exception(self):
        """测试 train_step 方法在异常情况下的行为"""
        # 模拟输入参数
        gs_render = MagicMock()
        viewpoint_cam = MagicMock()
        iteration = 1
        pipe = MagicMock()
        optim_opt = MagicMock()
        colors_preco = MagicMock()
        # 模拟 train_step 抛出异常
        self.trainer.train_step = MagicMock(side_effect=Exception("Test Exception"))
        # 使用 assertRaises 检查异常
        with self.assertRaises(Exception) as context:
            self.trainer.train_step(
                gs_render=gs_render,
                viewpoint_cam=viewpoint_cam,
                iteration=iteration,
                pipe=pipe,
                optim_opt=optim_opt,
                colors_preco=colors_preco
            )
        # 断言异常信息正确
        self.assertEqual(str(context.exception), "Test Exception", "应抛出预期的异常信息")

    


