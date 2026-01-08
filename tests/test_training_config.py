from llm_trainer.config import TrainingConfig


def test_report_to_default_is_none() -> None:
    cfg = TrainingConfig()
    assert cfg.report_to is None


def test_report_to_provided_list_is_independent() -> None:
    # Provide explicit lists and ensure they're independent
    cfg1 = TrainingConfig(report_to=["tensorboard"])
    cfg2 = TrainingConfig(report_to=["tensorboard"])

    assert cfg1.report_to == ["tensorboard"]
    assert cfg2.report_to == ["tensorboard"]
    assert cfg1.report_to is not cfg2.report_to

    # type guard before mutation
    assert isinstance(cfg1.report_to, list)
    cfg1.report_to.append("wandb")
    assert cfg1.report_to == ["tensorboard", "wandb"]
    assert cfg2.report_to == ["tensorboard"]
