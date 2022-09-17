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

if (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    fileInput = 'file:step1D88tt.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    fileInput = 'file:step1D92tt.root'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
    fileInput = 'file:step1D93tt.root'

print("Input file: ", fileInput)

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
