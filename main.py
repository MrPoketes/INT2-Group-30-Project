from utilities import run_model, load_model

from models.lewis_model import create_model

# Global variables
EPOCHS = 50

#run_model(EPOCHS)
model = load_model("complete_train/cp.ckpt")
