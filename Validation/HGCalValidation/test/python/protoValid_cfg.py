###############################################################################
# Way to use this:
#   cmsRun protoValid_cfg.py geometry=D88 type=hgcalSimHitStudy defaultInput=1
#
#   Options for geometry D88, D92, D93
#               type hgcalGeomCheck, hgcalSimHitStudy, hgcalDigiStudy,
#                    hgcalRecHitStudy, hgcalSiliconValidation
#               defaultInput 1, 0
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

############################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D88, D92, D93")
options.register('type',
                 "hgcalGeomCheck",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "type of operations: hgcalGeomCheck, hgcalSimHitStudy, hgcalDigiStudy, hgcalRecHitStudy, hgcalSiliconValidation")
options.register('defaultInput',
                 1, # default Value = true
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "change files path in case of defaultInput=0 for using DIGI o/p")

### get and parse the command line arguments
options.parseArguments()

print(options)

############################################################
# Use the options

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('PROD',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
fileCheck = "testHGCalSimWatcher" + options.geometry + ".root"
if (options.type == "hgcalSimHitStudy"):
    fileName = "hgcSimHit" + options.geometry + ".root"
elif (options.type == "hgcalDigiStudy"):
    fileName = "hgcDigi" + options.geometry + ".root"
elif (options.type == "hgcalRecHitStudy"):
    fileName = "hgcRecHit" + options.geometry + ".root"
elif (options.type == "hgcalSiliconValidation"):
    if (options.defaultInput == 0):
        fileName = "hgcDigValid" + options.geometry + ".root"
    else:
        fileName = "hgcSilValid" + options.geometry + ".root"
else:
    fileName = "hgcGeomCheck" + options.geometry + ".root"

print("Geometry file: ", geomFile)
print("Output file:   ", fileName)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.EventContent.EventContent_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

if (options.type == "hgcalSimHitStudy"):
    process.load('Validation.HGCalValidation.hgcSimHitStudy_cfi')
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring('file:step1.root') )
    process.analysis_step = cms.Path(process.hgcalSimHitStudy)
elif (options.type == "hgcalDigiStudy"):
    process.load('Configuration.StandardSequences.RawToDigi_cff')
    process.load('Validation.HGCalValidation.hgcDigiStudy_cfi')
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring('file:step2.root') )
    process.analysis_step = cms.Path(process.RawToDigi+process.hgcalDigiStudyEE+process.hgcalDigiStudyHEF+process.hgcalDigiStudyHEB)
elif (options.type == "hgcalRecHitStudy"):
    process.load('Validation.HGCalValidation.hgcalRecHitStudy_cff')
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring('file:step3.root') )
    process.analysis_step = cms.Path(process.hgcalRecHitStudyEE+process.hgcalRecHitStudyFH+process.hgcalRecHitStudyBH)
elif (options.type == "hgcalSiliconValidation"):
    if (options.defaultInput == 0):
        fileIn = "file:step2.root"
    else:
        fileIn = "file:step1.root"
    process.load('Validation.HGCalValidation.hgcalSiliconValidation_cfi')
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(fileIn) )
    process.analysis_step = cms.Path(process.hgcalSiliconAnalysisEE+process.hgcalSiliconAnalysisHEF)
else:
    process.load('Validation.HGCalValidation.hgcGeomCheck_cff')
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(fileCheck)
    )
    process.analysis_step = cms.Path(process.hgcGeomCheck)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True) )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step)
