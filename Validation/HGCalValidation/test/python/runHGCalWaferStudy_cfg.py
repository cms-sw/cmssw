###############################################################################
# Way to use this:
#   cmsRun runHGCalWaferStudy_cfg.py geometry=D88
#
#   Options for geometry D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

############################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D93",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D88, D92 D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

############################################################
# Use the options
from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
process = cms.Process('HGCGeomAnalysis',Phase2C11I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
fileInput = "file:step2" + options.geometry + "tt.root"
fileName = "hgcWafer" + options.geometry + "tt.root"

print("Geometry file: ", geomFile)
print("Input file:    ", fileInput)
print("Output file:   ", fileName)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hgcalWaferStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalValidation=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fileInput)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.raw2digi_step = cms.Path(process.RawToDigi)
process.analysis_step = cms.Path(process.hgcalWaferStudy)
process.hgcalWaferStudy.verbosity = 1
process.hgcalWaferStudy.nBinHit   = 60
process.hgcalWaferStudy.nBinDig   = 60
process.hgcalWaferStudy.layerMinSim = cms.untracked.vint32(1,1)
process.hgcalWaferStudy.layerMaxSim = cms.untracked.vint32(10,10)
process.hgcalWaferStudy.layerMinDig = cms.untracked.vint32(1,1)
process.hgcalWaferStudy.layerMaxDig = cms.untracked.vint32(10,10)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.analysis_step)
