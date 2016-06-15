import imp
file = open( "heppy_crab_config.py", 'r' )
cfg = imp.load_source( 'cfg', "heppy_crab_config.py", file)
config = cfg.config
import os
import re
dataset=os.environ["DATASET"]
m=re.match("\/(.*)\/(.*)\/(.*)",dataset)
if not m : 
  print "NO GOOD DATASET"

sample=m.group(1)+"__"+m.group(2)

config.General.requestName+= "_"+sample
config.Data.inputDataset = dataset
config.Data.publishDataName += "_"+sample
