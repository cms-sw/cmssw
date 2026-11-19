###############################################################################
# Way to use this:
#   cmsRun runHGCalSimHitDump_cfg.py geomName=Run4D120
#   Options for geomName Run4D120, Run4D121. Run4D122, Run4D123, Run4D126,
#                        Run4D127, Run4D128,
# 
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geomName',
                 "Run4D120",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geomName of operations: Run4D120, Run4D121, Run4D122, Run4D123, Run4D126, Run4D127, Run4D128")

### get and parse the command line arguments
 
options.parseArguments()
print(options)

####################################################################

geomName = options.geomName
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
fileName = "file:hgcV" + geomName + ".root"

print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)
print("i/p File Name:   ", fileName)

process = cms.Process("HGCalSimHitDump",ERA)

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load(geomFile) 
process.load('SimG4CMS.Calo.hgcalSimHitDump_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HitStudy=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(fileName),
)

process.analysis_step   = cms.Path(process.hgcalSimHitDump)

process.hgcalSimHitDump.MaxEvent = 10

# Schedule definition                                                          
process.schedule = cms.Schedule(process.analysis_step)
