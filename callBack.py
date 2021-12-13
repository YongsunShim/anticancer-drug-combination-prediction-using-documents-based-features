from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec): 
    def __init__(self): 
        self.epoch = 0 
        self.loss_to_be_subed = 0 
    
    def on_epoch_end(self, model): 
        loss = model.get_latest_training_loss() 
        loss_now = loss - self.loss_to_be_subed 
        self.loss_to_be_subed = loss 
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now)) 
        self.epoch += 1