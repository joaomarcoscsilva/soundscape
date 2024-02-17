import hydra
import jax
import supervised
from omegaconf import DictConfig

ConfigGenerator = callable[[jax.random.PRNGKey, DictConfig], DictConfig]


def hpsearch(rng, num_runs, gen: ConfigGenerator, keep_model=True):
    """
    Run a hyperparameter search given a configuration generator.

    If `keep_model` is True, the model object (callable, not weights)
    is reused, which prevents jax.jit recompilations.
    """

    kept_model = None

    for _ in range(num_runs):
        rng, gen_rng, train_rng = jax.random.split(rng, 3)
        config = gen(gen_rng, {})
        _, model_state, env = supervised.instantiate(config)

        if keep_model:
            if kept_model is None:
                kept_model = env.model
            else:
                env = env._replace(model=kept_model)

        results = supervised.train(train_rng, model_state, env)
