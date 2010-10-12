import FWCore.ParameterSet.Config as cms

process = cms.Process('merge')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')

process.MessageLogger = cms.Service("MessageLogger",
     destinations = cms.untracked.vstring('log')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('-s_DIGI2RAW.root')
)

process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    fileName = cms.untracked.string('embedded_RECO_merged.root'),
)


import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis')

options.register ('dir',
                    "none", # default value, false
                    VarParsing.VarParsing.multiplicity.singleton,
                    VarParsing.VarParsing.varType.string,         
                    "dir to merge")

options.parseArguments()
if options.dir == "none":
  raise Exception("Wrong usage") 


process.source.fileNames = cms.untracked.vstring()
import glob
fl=glob.glob(options.dir+"/*.root") 
if len(fl) == 0:
  raise Exception("No root files found in directory "+options.dir) 
else:
  print "Will process "+str(len(fl))+" files"
  
for f in fl:
   process.source.fileNames.extend(["file:"+f])

#print fl
#print process.source.fileNames
#name=str('embedded_RECO_merged_'+options.dir)
name="embedded_RECO_merged_"+options.dir
name=name.replace("/","_").replace(".","_")

process.MessageLogger.destinations = cms.untracked.vstring("log_"+name)

process.output.fileName = cms.untracked.string( name + ".root" )


print "Saving to: "
print process.output.fileName




# Additional output definition

# Other statements
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.out_step)

