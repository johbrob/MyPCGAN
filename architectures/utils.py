
def create_model_from_config(config):
    return config.model(**vars(config.args))
