import imp
file = open( "heppy_crab_config.py", 'r' )
cfg = imp.load_source( 'cfg', "heppy_crab_config.py", file)
config = cfg.config
import os
import re
dataset=os.environ["DATASET"]
m=re.match("\/(.*)\/(.*)\/(.*)",dataset)
if not m:
    raise Exception("could not parse DATASET: {0}".format(dataset))

site=os.environ["SITE"]
m2=re.match("T.*", site)
if not m2:
    raise Exception("could not parse SITE: {0}".format(dataset))

lfnbase = os.environ["LFNBASE"]
m3 = re.match("\/store.*", lfnbase)


sample=m.group(2)
#sample=sample[:120]

config.General.workArea += "_".join(["", site])
config.General.requestName += "_".join(["", sample])
config.Data.inputDataset = dataset
config.Data.outLFNDirBase = lfnbase

#config.Data.publishDataName += "_"+sample
config.Site.storageSite = site

print config
