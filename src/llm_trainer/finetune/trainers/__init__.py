"""TRL-style trainer package exports with lazy loading.

This package mirrors HuggingFace TRL's trainer structure while remaining
compatible with LLM Trainer's architecture.
"""
from typing import TYPE_CHECKING
from transformers.utils import _LazyModule

_import_structure = {
    # Core trainers and configs
    "sft_trainer": ["SFTTrainer"],
    "sft_config": ["SFTConfig"],
    "dpo_trainer": ["DPOTrainer"],
    "dpo_config": ["DPOConfig"],

    # Additional trainers/configs (placeholders with consistent API)
    "ppo_trainer": ["PPOTrainer"],
    "ppo_config": ["PPOConfig"],
    "reward_trainer": ["RewardTrainer"],
    "reward_config": ["RewardConfig"],
    "orpo_trainer": ["ORPOTrainer"],
    "orpo_config": ["ORPOConfig"],
    "kto_trainer": ["KTOTrainer"],
    "kto_config": ["KTOConfig"],
    "grpo_trainer": ["GRPOTrainer"],
    "grpo_config": ["GRPOConfig"],
    "rloo_trainer": ["RLOOTrainer"],
    "rloo_config": ["RLOOConfig"],
    "xpo_trainer": ["XPOTrainer"],
    "xpo_config": ["XPOConfig"],
    "cpo_trainer": ["CPOTrainer"],
    "cpo_config": ["CPOConfig"],
    "bco_trainer": ["BCOTrainer"],
    "bco_config": ["BCOConfig"],
    "prm_trainer": ["PRMTrainer"],
    "prm_config": ["PRMConfig"],
    "nash_md_trainer": ["NashMDTrainer"],
    "nash_md_config": ["NashMDConfig"],
    "online_dpo_trainer": ["OnlineDPOTrainer"],
    "online_dpo_config": ["OnlineDPOConfig"],

    # Supporting modules
    "callbacks": [
        "BEMACallback",
        "LogCompletionsCallback",
        "MergeModelCallback",
        "RichProgressCallback",
        "SyncRefModelCallback",
        "WinRateCallback",
    ],
    "utils": [
        "RunningMoments",
        "compute_accuracy",
        "disable_dropout_in_model",
        "empty_cache",
        "peft_module_casting_to_bf16",
    ],
    "judges": [
        "AllTrueJudge",
        "BaseBinaryJudge",
        "BaseJudge",
        "BasePairwiseJudge",
        "BaseRankJudge",
        "HfPairwiseJudge",
        "OpenAIPairwiseJudge",
        "PairRMJudge",
    ],
}

if TYPE_CHECKING:
    from .sft_trainer import SFTTrainer
    from .sft_config import SFTConfig
    from .dpo_trainer import DPOTrainer
    from .dpo_config import DPOConfig

    from .ppo_trainer import PPOTrainer
    from .ppo_config import PPOConfig
    from .reward_trainer import RewardTrainer
    from .reward_config import RewardConfig
    from .orpo_trainer import ORPOTrainer
    from .orpo_config import ORPOConfig
    from .kto_trainer import KTOTrainer
    from .kto_config import KTOConfig
    from .grpo_trainer import GRPOTrainer
    from .grpo_config import GRPOConfig
    from .rloo_trainer import RLOOTrainer
    from .rloo_config import RLOOConfig
    from .xpo_trainer import XPOTrainer
    from .xpo_config import XPOConfig
    from .cpo_trainer import CPOTrainer
    from .cpo_config import CPOConfig
    from .bco_trainer import BCOTrainer
    from .bco_config import BCOConfig
    from .prm_trainer import PRMTrainer
    from .prm_config import PRMConfig
    from .nash_md_trainer import NashMDTrainer
    from .nash_md_config import NashMDConfig
    from .online_dpo_trainer import OnlineDPOTrainer
    from .online_dpo_config import OnlineDPOConfig

    from .callbacks import (
        BEMACallback,
        LogCompletionsCallback,
        MergeModelCallback,
        RichProgressCallback,
        SyncRefModelCallback,
        WinRateCallback,
    )
    from .utils import (
        RunningMoments,
        compute_accuracy,
        disable_dropout_in_model,
        empty_cache,
        peft_module_casting_to_bf16,
    )
    from .judges import (
        AllTrueJudge,
        BaseBinaryJudge,
        BaseJudge,
        BasePairwiseJudge,
        BaseRankJudge,
        HfPairwiseJudge,
        OpenAIPairwiseJudge,
        PairRMJudge,
    )
else:
    import sys as _sys

    _module = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    _sys.modules[__name__] = _module
