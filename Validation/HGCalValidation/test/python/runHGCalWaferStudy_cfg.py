###############################################################################
# Way to use this:
#   cmsRun runHGCalWaferStudy_cfg.py geometry=D88
#
#   Options for geometry D49, D68, D77, D83, D84, D88, D92, D93
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
                  "geometry of operations: D49, D68, D77, D83, D84, D88, D92")

### get and parse the command line arguments
options.parseArguments()

print(options)

############################################################
# Use the options

import FWCore.ParameterSet.Config as cms

if (options.geometry == "D49"):
    from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
    process = cms.Process('HGCGeomAnalysis',Phase2C9)
    process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    fileName = 'hgcWaferD49.root'
elif (options.geometry == "D68"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('HGCGeomAnalysis',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D68_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D68Reco_cff')
    fileName = 'hgcWaferD68.root'
elif (options.geometry == "D70"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('HGCGeomAnalysis',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D70_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D70Reco_cff')
    fileName = 'hgcWaferD70.root'
elif (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('HGCGeomAnalysis',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    fileName = 'hgcWaferD83.root'
elif (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('HGCGeomAnalysis',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    fileName = 'hgcWaferD88.root'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('HGCGeomAnalysis',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D92_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('HGCGeomAnalysis',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D93_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
else:
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('HGCGeomAnalysis',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    fileName = 'hgcWaferD77.root'

print("Output file: ", fileName)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hgcalWaferStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalValidation=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:step2.root',
        )
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
