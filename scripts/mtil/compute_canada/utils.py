def set_ppo(template, key=None, val=None, hyperparam_keys=None, hyperparam_map=None):
    assert (key is not None) != (
        hyperparam_keys is not None and hyperparam_map is not None
    )
    if key is not None:
        if key == "objective":
            template["learner_config"]["pi_loss_setting"]["objective"] = val
        if key == "load_pretrain":
            pretrain_config = {
                "checkpoint_path": val,
                "load_components": ["policy", "vf"],
                "freeze_obs_rms": False,
            }
            template["learner_config"]["load_pretrain"] = pretrain_config
    elif hyperparam_keys is not None:
        if "lr" in hyperparam_keys:
            template["optimizer_config"]["policy"]["lr"]["scheduler_kwargs"][
                "value"
            ] = hyperparam_map("lr")
            template["optimizer_config"]["vf"]["lr"]["scheduler_kwargs"][
                "value"
            ] = hyperparam_map("lr")

        if "max_grad_norm" in hyperparam_keys:
            template["optimizer_config"]["policy"]["max_grad_norm"] = hyperparam_map(
                "max_grad_norm"
            )
            template["optimizer_config"]["vf"]["max_grad_norm"] = hyperparam_map(
                "max_grad_norm"
            )

        if "obs_rms" in hyperparam_keys:
            template["learner_config"]["obs_rms"] = hyperparam_map("obs_rms")

        if "value_rms" in hyperparam_keys:
            template["learner_config"]["value_rms"] = hyperparam_map("value_rms")

        if "opt_batch_size" in hyperparam_keys:
            template["learner_config"]["opt_batch_size"] = hyperparam_map(
                "opt_batch_size"
            )

        if "opt_epochs" in hyperparam_keys:
            template["learner_config"]["opt_epochs"] = hyperparam_map("opt_epochs")

        if "vf_clip_param" in hyperparam_keys:
            template["learner_config"]["vf_loss_setting"][
                "clip_param"
            ] = hyperparam_map("vf_clip_param")

        if "ent_coef" in hyperparam_keys:
            template["learner_config"]["ent_loss_setting"] = hyperparam_map("ent_coef")

        if "beta" in hyperparam_keys:
            template["learner_config"]["pi_loss_setting"]["beta"] = hyperparam_map(
                "beta"
            )

        if "clip_param" in hyperparam_keys:
            template["learner_config"]["pi_loss_setting"][
                "clip_param"
            ] = hyperparam_map("clip_param")


def set_bc(template, key=None, val=None, hyperparam_keys=None, hyperparam_map=None):
    assert (key is not None) != (
        hyperparam_keys is not None and hyperparam_map is not None
    )
    if key is not None:
        if key == "loss":
            template["learner_config"]["losses"] = [val, "l2"]
    elif hyperparam_keys is not None:
        if "lr" in hyperparam_keys:
            template["optimizer_config"]["lr"]["scheduler_kwargs"][
                "value"
            ] = hyperparam_map("lr")

        if "max_grad_norm" in hyperparam_keys:
            template["optimizer_config"]["max_grad_norm"] = hyperparam_map(
                "max_grad_norm"
            )

        if "obs_rms" in hyperparam_keys:
            template["learner_config"]["obs_rms"] = hyperparam_map("obs_rms")

        if "batch_size" in hyperparam_keys:
            template["learner_config"]["batch_size"] = hyperparam_map("batch_size")

        if "l2" in hyperparam_keys:
            template["learner_config"]["loss_settings"][1][
                "coefficient"
            ] = hyperparam_map("l2")


def set_mtbc(template, key=None, val=None, hyperparam_keys=None, hyperparam_map=None):
    assert (key is not None) != (
        hyperparam_keys is not None and hyperparam_map is not None
    )
    if key is not None:
        if key == "loss":
            template["learner_config"]["losses"] = [val, "l2"]
    elif hyperparam_keys is not None:
        if "lr" in hyperparam_keys:
            template["optimizer_config"]["lr"]["scheduler_kwargs"][
                "value"
            ] = hyperparam_map("lr")

        if "max_grad_norm" in hyperparam_keys:
            template["optimizer_config"]["encoder"]["max_grad_norm"] = hyperparam_map(
                "max_grad_norm"
            )
            template["optimizer_config"]["predictor"]["max_grad_norm"] = hyperparam_map(
                "max_grad_norm"
            )

        if "obs_rms" in hyperparam_keys:
            template["learner_config"]["obs_rms"] = hyperparam_map("obs_rms")

        if "batch_size" in hyperparam_keys:
            template["learner_config"]["batch_size"] = hyperparam_map("batch_size")

        if "l2" in hyperparam_keys:
            template["learner_config"]["loss_settings"][1][
                "coefficient"
            ] = hyperparam_map("l2")
