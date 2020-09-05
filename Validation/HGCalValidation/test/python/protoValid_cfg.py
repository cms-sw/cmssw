###############################################################################
# Way to use this:
#   cmsRun protoValid_cfg.py geometry=D62 type=hgcalSimHitStudy defaultInput=1
#
#   Options for geometry D49, D58, D59, D62
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
                 "D62",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D49, D58, D59, D62")
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

if (options.geometry == "D49"):
    from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
    process = cms.Process('PROD',Phase2C9)
    process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD49.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD49.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD49.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD49.root'
        else:
            fileName = 'hgcSilValidD49.root'
    else:
        fileName = 'hgcGeomCheckD49.root'
elif (options.geometry == "D58"):
    from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
    process = cms.Process('PROD',Phase2C12)
    process.load('Configuration.Geometry.GeometryExtended2026D58_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D58Reco_cff')
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD58.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD58.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD58.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD58.root'
        else:
            fileName = 'hgcSilValidD58.root'
    else:
        fileName = 'hgcGeomCheckD58.root'
elif (options.geometry == "D59"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D59_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D59Reco_cff')
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD59.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD59.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD59.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD59.root'
        else:
            fileName = 'hgcSilValidD59.root'
    else:
        fileName = 'hgcGeomCheckD59.root'
else:
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D62_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D62Reco_cff')
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD62.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD62.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD62.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD62.root'
        else:
            fileName = 'hgcSilValidD62.root'
    else:
        fileName = 'hgcGeomCheckD62.root'

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
    process.MessageLogger.categories.append('HGCalGeom')

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
                                fileNames = cms.untracked.vstring('file:testHGCalSimWatcher.root')
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
