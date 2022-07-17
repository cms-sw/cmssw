###############################################################################
# Way to use this:
#   cmsRun protoValid_cfg.py geometry=D77 type=hgcalSimHitStudy defaultInput=1
#
#   Options for geometry D49, D68, D77, D83, D84, D88, D92, D93
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
                  "geometry of operations: D49, D68, D77, D83, D84, D88, D92, D93")
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
    fileCheck = 'testHGCalSimWatcherV11.root'
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
elif (options.geometry == "D68"):
    from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
    process = cms.Process('PROD',Phase2C12)
    process.load('Configuration.Geometry.GeometryExtended2026D68_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D68Reco_cff')
    fileCheck = 'testHGCalSimWatcherV12.root'
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD68.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD68.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD68.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD68.root'
        else:
            fileName = 'hgcSilValidD68.root'
    else:
        fileName = 'hgcGeomCheckD68.root'
elif (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    fileCheck = 'testHGCalSimWatcherV15.root'
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD83.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD83.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD83.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD83.root'
        else:
            fileName = 'hgcSilValidD83.root'
    else:
        fileName = 'hgcGeomCheckD83.root'
elif (options.geometry == "D84"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D84_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D84Reco_cff')
    fileCheck = 'testHGCalSimWatcherV13.root'
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD84.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD84.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD84.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD84.root'
        else:
            fileName = 'hgcSilValidD84.root'
    else:
        fileName = 'hgcGeomCheckD84.root'
elif (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    fileCheck = 'testHGCalSimWatcherV16.root'
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD88.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD88.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD88.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD88.root'
        else:
            fileName = 'hgcSilValidD88.root'
    else:
        fileName = 'hgcGeomCheckD88.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D92_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    fileCheck = 'testHGCalSimWatcherV17.root'
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD92.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD92.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD92.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD92.root'
        else:
            fileName = 'hgcSilValidD92.root'
    else:
        fileName = 'hgcGeomCheckD92.root'
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D93_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
    fileCheck = 'testHGCalSimWatcherV17.root'
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD93.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD93.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD93.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD93.root'
        else:
            fileName = 'hgcSilValidD93.root'
    else:
        fileName = 'hgcGeomCheckD93.root'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    fileCheck = 'testHGCalSimWatcherV14.root'
    if (options.type == "hgcalSimHitStudy"):
        fileName = 'hgcSimHitD77.root'
    elif (options.type == "hgcalDigiStudy"):
        fileName = 'hgcDigiD77.root'
    elif (options.type == "hgcalRecHitStudy"):
        fileName = 'hgcRecHitD77.root'
    elif (options.type == "hgcalSiliconValidation"):
        if (options.defaultInput == 0):
            fileName = 'hgcDigValidD77.root'
        else:
            fileName = 'hgcSilValidD77.root'
    else:
        fileName = 'hgcGeomCheckD77.root'

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
