from utilities import run_model
from models.vgg_like import create_model

# Global variables
EPOCHS = 10

model = create_model()

run_model(model, EPOCHS)
