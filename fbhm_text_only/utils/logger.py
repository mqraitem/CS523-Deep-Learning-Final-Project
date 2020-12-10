'''
  * Maan Qraitem
'''

class Logger(): 
  def __init__(self, log_file): 
    self.log_file = log_file 
    self.f = open(log_file, "w")

  def write_model(self, model): 
    print(model, file=self.f)

  def write_stats(self, stats): 
    self.f.write(stats + "\n") 
  
  def close_file(self): 
    self.f.close() 

