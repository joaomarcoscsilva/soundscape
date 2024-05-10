import hydra
import jax
import supervised
from omegaconf import DictConfig, open_dict
from typing import Callable, Any
from functools import partial

# ConfigGenerator = Callable[[jax.random.PRNGKey], DictConfig]
ConfigGenerator = Any


def gen_settings(rng, hpsettings):
    exp = hpsettings.experiment.copy()

    _rng, rng = jax.random.split(rng)
    exp.seed = int(jax.random.randint(_rng, (), 0, 10000))

    _rng, rng = jax.random.split(rng)
    exp.optimizer.optim_name = (
        "sgd" if jax.random.uniform(_rng) < hpsettings.sgd_chance else "adamw"
    )

    _rng, rng = jax.random.split(rng)
    exp.optimizer.log_learning_rate = float(jax.random.uniform(
        _rng, minval=hpsettings.log_lr.min, maxval=hpsettings.log_lr.max
    ))

    _rng, rng = jax.random.split(rng)
    exp.optimizer.log_weight_decay = float(jax.random.uniform(
        _rng, minval=hpsettings.log_wd.min, maxval=hpsettings.log_wd.max
    ))

    if exp.optimizer.optim_name == "sgd":
        _rng, rng = jax.random.split(rng)
        with open_dict(exp):
            exp.optimizer.sub_log_momentum = float(jax.random.uniform(
                _rng,
                minval=hpsettings.sub_log_momentum.min,
                maxval=hpsettings.sub_log_momentum.max,
            ))

    _rng, rng = jax.random.split(rng)
    augment = jax.random.uniform(_rng) < hpsettings.augment.chance

    if augment:
        _rng, rng = jax.random.split(rng)
        mixup = jax.random.uniform(_rng) < hpsettings.augment.mixup_chance 
        if mixup:
            _rng, rng = jax.random.split(rng)
            exp.augmentation.mixup_alpha = float(jax.random.uniform(
                _rng,
                minval=hpsettings.augment.mixup.min,
                maxval=hpsettings.augment.mixup.max,
            ))
        else:
            exp.augmentation.mixup_alpha = None

        _rng, rng = jax.random.split(rng)
        cutmix = jax.random.uniform(_rng) < hpsettings.augment.cutmix_chance 
        if cutmix:
            _rng, rng = jax.random.split(rng)
            exp.augmentation.cutmix_alpha = float(jax.random.uniform(
                _rng,
                minval=hpsettings.augment.cutmix.min,
                maxval=hpsettings.augment.cutmix.max,
            ))
        else:
            exp.augmentation.cutmix_alpha = None

        _rng, rng = jax.random.split(rng)
        exp.augmentation.crop_type = (
            "random"
            if jax.random.uniform(_rng) < hpsettings.augment.random_crop_chance
            else "center"
        )

        _rng, rng = jax.random.split(rng)
        exp.augmentation.cropped_length = float(jax.random.uniform(
            _rng,
            minval=hpsettings.augment.cropped_length.min,
            maxval=hpsettings.augment.cropped_length.max,
        ))
    
    else:
        exp.augmentation.mixup_alpha = None
        exp.augmentation.cutmix_alpha = None
        exp.augmentation.crop_type = "center"
        exp.augmentation.cropped_length = 3.0

    return exp


def hpsearch(rng, gen: ConfigGenerator, settings):
    """
    Run a hyperparameter search given a configuration generator.

    The model object (the callable, not weights) is reused,
    which prevents frequent jax.jit recompilations.
    """

    kept_model = None

    for i in range(settings.num_runs):
        _rng, rng = jax.random.split(rng)
        config = gen(_rng)
        print(config)

        _rng, model_state, env = supervised.instantiate(config)

        if settings.keep_model:
            if kept_model is None:
                kept_model = env.model
            else:
                env = env._replace(model=kept_model)

        results = supervised.train(_rng, model_state, env)


@hydra.main(
    config_path="../../settings",
    config_name="hpsearch/resnet_leec12.yaml",
    version_base=None,
)
def main(settings):
    rng = jax.random.PRNGKey(settings.hpsearch.seed)
    gen = partial(gen_settings, hpsettings=settings.hpsearch)
    hpsearch(rng, gen, settings.hpsearch)


if __name__ == "__main__":
    main()