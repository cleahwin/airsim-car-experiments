

def train_on_real_and_sim (model_path: str, sim_data, real_data):
    """
    This function trains the model on a mix of the given data sets.

    Params:
    model_path -- path to save the model to. If path already exists, model will continue training.
    sim_data -- simulator dataset of type
    """

    # Determines if model exists at model_path.