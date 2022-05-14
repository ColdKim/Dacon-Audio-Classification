from .Private1st import Network

def get_model(name):
    model_class = None
    if name == "Private1st":
        model_class = Network
    return model_class