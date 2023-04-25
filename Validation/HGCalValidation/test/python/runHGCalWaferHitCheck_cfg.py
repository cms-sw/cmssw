###############################################################################
# Way to use this:
#   cmsRun runHGCalWaferHitCheck_cfg.py geometry=D88
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
                 "D93",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('WaferHitCheck',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
fileInput = "file:step1" + options.geometry + "tt.root"

print("Geometry file: ", geomFile)
print("Input file:    ", fileInput)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hgcWaferHitCheck_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalValidation=dict()
#   process.MessageLogger.HGCalGeom=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fileInput)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.analysis_step = cms.Path(process.hgcalWaferHitCheckEE)
#process.analysis_step = cms.Path(process.hgcalWaferHitCheckHEF)
process.hgcalWaferHitCheckEE.verbosity = 1
process.hgcalWaferHitCheckHEF.verbosity = 1
#process.hgcalWaferHitCheckEE.inputType = 2
#process.hgcalWaferHitCheckHEF.inputType = 2
#process.hgcalWaferHitCheckEE.source = cms.InputTag("simHGCalUnsuppressedDigis", "EE")
#process.hgcalWaferHitCheckHEF.source = cms.InputTag("simHGCalUnsuppressedDigis","HEfront")
#process.hgcalWaferHitCheckEE.inputType = 3                                   
#process.hgcalWaferHitCheckHEF.inputType = 3
#process.hgcalWaferHitCheckEE.source = cms.InputTag("HGCalRecHit", "HGCEERecHits")
#process.hgcalWaferHitCheckHEF.source = cms.InputTag("HGCalRecHit", "HGCHEFRecHits")

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step)
