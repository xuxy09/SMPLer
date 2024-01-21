from utils.argument_manager import ArgManager
from trainer import Trainer

arg_manager = ArgManager()

trainer = Trainer(arg_manager.args)
trainer.run()