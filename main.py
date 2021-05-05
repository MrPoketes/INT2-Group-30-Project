from utilities import run_model
from models.lewis_model import create_model

# Global variables
EPOCHS = 100

model = create_model()

run_model(model, EPOCHS)
