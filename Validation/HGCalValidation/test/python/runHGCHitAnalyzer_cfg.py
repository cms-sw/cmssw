###############################################################################
# Way to use this:
#   cmsRun runHGCHitAnalyzer_cfg.py geometry=D88
#
#   Options for geometry D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('HGCHitAnalyzer',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
inFile = "file:step3" + options.geometry + ".root"
outFile = "relValTTbar" + options.geometry + ".root"

print("Geometry file: ", geomFile)
print("Input file:    ", inFile)
print("Output file:   ", outFile)

# import of standard configurations
process.load(geomFile)
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')    
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')
process.MessageLogger.cerr.FwkReport.reportEvery = 10

#    

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(inFile)
)

process.load('Validation.HGCalValidation.hgcHitValidation_cfi')

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(outFile),
                                   closeFileFast = cms.untracked.bool(True)
)

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

#process.hgcHitAnalysis.ietaExcludeBH = [16,92,93,94,95,96,97,98,99,100]
#process.hgcHitAnalysis.ietaExcludeBH = [16, 32, 33]

process.p = cms.Path(process.hgcHitAnalysis)


