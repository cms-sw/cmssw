###############################################################################
# Way to use this:
#   cmsRun runZDCStudy_cfg.py type=Standard
#
#   Options for type Standard, Forward, Legacy
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "Standard",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: Standard, Forward, Legacy")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD

process = cms.Process("ZDCStudy")

inFile = "file:zdc" + options.type + ".root"
outFile = options.type + ".root"
print("Input file:  ", inFile)
print("Output file: ", outFile)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

process.MessageLogger.HitStudy=dict()

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(inFile)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(outFile)
)

process.load("SimG4CMS.Forward.zdcSimHitStudy_cfi")

process.zdcSimHitStudy.Verbose = True
 
process.schedule = cms.Path(process.zdcSimHitStudy)

