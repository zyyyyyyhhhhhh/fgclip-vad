import os
import torch
import time
from datetime import datetime

from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
# import torch.distributed.nn as nn_dist
import torch.nn.functional as F

class CLIPTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化batch级别的loss监控
        self.batch_losses = []
        self.batch_step = 0
        self.total_steps = 0
        self.epoch = 0
        
        # ✅ 新增：独立的累积步数计数器（每次optimizer.step()后递增）
        self.accumulated_step = 0  # 用于记录真正的训练步数
        
        # ✅ 初始化TensorBoard
        if self.args.local_rank in [-1, 0]:  # 只在主进程记录
            tensorboard_dir = os.path.join(self.args.output_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            print(f"[TensorBoard] 日志目录: {tensorboard_dir}")
            print(f"[TensorBoard] 启动命令: tensorboard --logdir {tensorboard_dir} --port 6006")
        else:
            self.tb_writer = None
        
        # 创建详细的loss日志文件
        if self.args.local_rank in [-1, 0]:  # 只在主进程记录
            self.loss_log_file = os.path.join(self.args.output_dir, "batch_losses.log")
            with open(self.loss_log_file, "w") as f:
                f.write("=" * 100 + "\n")
                f.write("FG-CLIP Training - Detailed Batch Loss Log\n")
                f.write("=" * 100 + "\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Batch Size: {self.args.per_device_train_batch_size}\n")
                f.write(f"Gradient Accumulation Steps: {self.args.gradient_accumulation_steps}\n")
                f.write(f"Effective Batch Size: {self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps}\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"{'Epoch':<6} {'Global_Step':<12} {'Batch':<8} {'Loss':<12} {'Loss_Global':<12} {'Loss_Region':<12} {'Loss_Hard':<12} {'LR':<12} {'Time':<10}\n")
                f.write("-" * 100 + "\n")

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        return super()._get_train_sampler()

        
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # print(decay_parameters)

            if self.args.text_model_lr is not None:
                text_model_parameters = [name for name, _ in opt_model.named_parameters() if "text_model" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in text_model_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in text_model_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in text_model_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.text_model_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in text_model_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.text_model_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            # print(optimizer_grouped_parameters)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        super(CLIPTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(CLIPTrainer, self)._save(output_dir, state_dict)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        重写training_step来监控每个batch的详细loss
        
        Args:
            model: 训练的模型
            inputs: 输入数据batch
            num_items_in_batch: batch中的样本数(新版transformers需要)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 记录开始时间
        start_time = time.time()
        
        # 前向传播
        with self.compute_loss_context_manager():
            loss_dict = self.compute_loss(model, inputs, return_outputs=True)
            
            if isinstance(loss_dict, tuple):
                loss = loss_dict[0]
                outputs = loss_dict[1]
            else:
                loss = loss_dict
                outputs = None
        
        # ✅ 在反向传播前立即提取loss详情(因为反向传播后tensor可能被清空)
        loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
        loss_global = 0.0
        loss_region = 0.0
        loss_hard = 0.0
        
        # ✅ 调试: 检查outputs和loss_dict
        if outputs is not None:
            has_loss_dict = hasattr(outputs, 'loss_dict')
            if has_loss_dict:
                loss_dict_details = outputs.loss_dict
                loss_global = float(loss_dict_details.get('loss_global', 0.0))
                loss_region = float(loss_dict_details.get('loss_region', 0.0))
                loss_hard = float(loss_dict_details.get('loss_hard_neg', 0.0))
            else:
                # 如果没有loss_dict,打印一次警告
                if self.batch_step == 0:
                    print(f"⚠️ Warning: outputs has no 'loss_dict' attribute. Type: {type(outputs)}, Dir: {dir(outputs)}")
        else:
            if self.batch_step == 0:
                print(f"⚠️ Warning: outputs is None!")
        
        # 梯度缩放(如果使用混合精度)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        # 记录batch信息
        batch_time = time.time() - start_time
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # 📊 Memory Bank状态监控
        mb_status = ""
        if hasattr(model, 'module'):  # DDP模式
            actual_model = model.module
        else:
            actual_model = model
        
        if hasattr(actual_model, 'use_memory_bank') and actual_model.use_memory_bank:
            queue_ptr = int(actual_model.queue_ptr.item())
            queue_full = actual_model.queue_is_full.item()
            effective_size = actual_model.memory_bank_size if queue_full else queue_ptr
            mb_status = f" | MB: {effective_size}/128 {'✓' if queue_full else '...'}"
        
        self.batch_step += 1
        
        # ✅ 修复：在累积最后一步时更新accumulated_step
        # 判断是否是梯度累积的最后一步（即将执行optimizer.step()）
        is_last_accumulation_step = (self.batch_step % self.args.gradient_accumulation_steps == 0)
        if is_last_accumulation_step:
            self.accumulated_step += 1
        
        self.total_steps = self.state.global_step  # 保留原有逻辑（向后兼容）
        
        # 每个batch都打印到终端
        if self.args.local_rank in [-1, 0]:
            self.batch_losses.append(loss_value)
            
            # ✅ TensorBoard记录（使用accumulated_step确保数据一致性）
            if self.tb_writer is not None:
                # 使用accumulated_step作为x轴（真实的训练步数）
                self.tb_writer.add_scalar('Loss/Total', loss_value, self.accumulated_step)
                self.tb_writer.add_scalar('Loss/Global', loss_global, self.accumulated_step)
                self.tb_writer.add_scalar('Loss/Region', loss_region, self.accumulated_step)
                self.tb_writer.add_scalar('Loss/HardNeg', loss_hard, self.accumulated_step)
                
                # 记录学习率
                self.tb_writer.add_scalar('Training/LearningRate', current_lr, self.accumulated_step)
                
                # 记录训练速度
                self.tb_writer.add_scalar('Training/BatchTime', batch_time, self.accumulated_step)
                
                # ✅ 修复：从训练开始就记录Memory Bank状态（而不是等use_memory_bank=True）
                # 这样可以清晰看到MB从禁用→启用→填充的完整过程
                if hasattr(actual_model, 'queue_ptr') and hasattr(actual_model, 'queue_is_full'):
                    queue_ptr = int(actual_model.queue_ptr.item())
                    queue_full = actual_model.queue_is_full.item()
                    mb_enabled = getattr(actual_model, 'use_memory_bank', False)
                    
                    # 计算有效队列大小
                    if mb_enabled:
                        # MB已启用：记录实际填充大小
                        effective_size = actual_model.memory_bank_size if queue_full else queue_ptr
                    else:
                        # MB未启用：记录为0（表示未激活状态）
                        effective_size = 0
                    
                    # 记录到TensorBoard（使用accumulated_step）
                    self.tb_writer.add_scalar('MemoryBank/Size', effective_size, self.accumulated_step)
                    self.tb_writer.add_scalar('MemoryBank/Full', int(queue_full), self.accumulated_step)
                    self.tb_writer.add_scalar('MemoryBank/Enabled', int(mb_enabled), self.accumulated_step)  # 新增：是否启用
                    
                    # 如果MB已启用，额外记录队列指针位置（调试用）
                    if mb_enabled:
                        self.tb_writer.add_scalar('MemoryBank/QueuePtr', queue_ptr, self.accumulated_step)
                
                # 每10个batch计算移动平均
                if self.batch_step % 10 == 0 and len(self.batch_losses) >= 10:
                    avg_loss = sum(self.batch_losses[-10:]) / 10
                    self.tb_writer.add_scalar('Loss/MovingAvg10', avg_loss, self.accumulated_step)
            
            # 终端实时输出(每个梯度累积步) - 显示累积平均loss
            # ✅ 使用accumulated_step显示真实的训练步数（与TensorBoard一致）
            print(f"[ACCUMULATED][Epoch {self.epoch}][Step {self.accumulated_step}][Batch {self.batch_step}] "
                  f"AvgLoss: {loss_value:.6f} | Global: {loss_global:.6f} | Region: {loss_region:.6f} | "
                  f"Hard: {loss_hard:.6f}{mb_status} | LR: {current_lr:.2e} | Time: {batch_time:.2f}s")
            
            # 写入详细日志文件
            with open(self.loss_log_file, "a") as f:
                f.write(f"{self.epoch:<6} {self.total_steps:<12} {self.batch_step:<8} "
                       f"{loss_value:<12.6f} {loss_global:<12.6f} {loss_region:<12.6f} "
                       f"{loss_hard:<12.6f} {current_lr:<12.2e} {batch_time:<10.2f}\n")
            
            # 每10个batch统计一次
            if self.batch_step % 10 == 0:
                avg_loss = sum(self.batch_losses[-10:]) / min(10, len(self.batch_losses))
                print(f"\n{'='*100}")
                print(f"[统计] 最近10个batch平均Loss: {avg_loss:.6f}")
                print(f"{'='*100}\n")
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写compute_loss来返回详细的loss信息
        """
        outputs = model(**inputs)
        
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            # 处理可能的其他输出格式
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                raise ValueError("Model output doesn't contain 'loss' field")
        
        return (loss, outputs) if return_outputs else loss
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时的回调 - 将TensorBoard writer注入到模型中"""
        if self.tb_writer is not None and args.local_rank in [-1, 0]:
            # 获取实际的模型（处理DDP包装）
            if hasattr(self.model, 'module'):
                actual_model = self.model.module
            else:
                actual_model = self.model
            
            # 将TensorBoard writer传递给模型（用于micro-batch级别的记录）
            actual_model.tb_writer = self.tb_writer
            print(f"[TensorBoard] ✅ TensorBoard writer已注入到模型，启用micro-batch级别记录")
            print(f"[TensorBoard] 📊 你将看到两套曲线：")
            print(f"             - MicroBatch/* : 每个forward的原始loss（高频波动）")
            print(f"             - Loss/*        : 梯度累积后的平均loss（平滑曲线）")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Epoch开始时的回调"""
        self.epoch = state.epoch if state.epoch is not None else 0
        self.batch_step = 0
        
        if args.local_rank in [-1, 0]:
            print(f"\n{'='*100}")
            print(f"🚀 开始 Epoch {self.epoch + 1}")
            print(f"{'='*100}\n")
            
            with open(self.loss_log_file, "a") as f:
                f.write(f"\n{'='*100}\n")
                f.write(f"Epoch {self.epoch + 1} Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*100}\n\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时的回调 - 关闭TensorBoard"""
        if self.tb_writer is not None:
            print(f"\n[TensorBoard] 关闭 TensorBoard writer...")
            self.tb_writer.close()
            print(f"[TensorBoard] ✓ 已关闭")
        
        if args.local_rank in [-1, 0]:
            print(f"\n{'='*100}")
            print(f"✅ 训练完成！")
            print(f"{'='*100}\n")
            
            with open(self.loss_log_file, "a") as f:
                f.write(f"\n{'='*100}\n")
                f.write(f"Training Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*100}\n")
    
    def __del__(self):
        """析构函数 - 确保TensorBoard writer被关闭"""
        if hasattr(self, 'tb_writer') and self.tb_writer is not None:
            try:
                self.tb_writer.close()
            except:
                pass
