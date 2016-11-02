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

replacePatterns=[
("pythia","Py"),
("80X_mcRun2_asymptotic","80r2as"),
("RunIISpring16","spr16"),
("MiniAODv2","MAv2"),
("miniAODv2","MAv2"),
("PUSpring16","puspr16"),
("RAWAODSIM",""),
("reHLT","HLT")
]

for (s,r) in replacePatterns :
  sample=re.sub(s,r,sample)

config.General.requestName+= "_"+sample
if len(config.General.requestName) > 100 :
  config.General.requestName=config.General.requestName[:90]+config.General.requestName[-10:] 

print config.General.requestName
config.Data.inputDataset = dataset
config.Data.outputDatasetTag += "_"+sample
