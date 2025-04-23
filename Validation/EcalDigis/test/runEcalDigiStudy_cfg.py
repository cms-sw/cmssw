###############################################################################
# Way to use this:
#   cmsRun runEcalDigiStudy_cfg.py IPType=FullSimSignalwithFullSimPU
#
#   Options for IPType FastSimSignalwithFastSimPU, FullSimSignalwithFastSimPU,
#                      FullSimSignalwithFullSimPU
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('IPType',
                 "FullSimSignalwithFullSimPU",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "IPType of operations: FastSimSignalwithFastSimPU, FullSimSignalwithFastSimPU, FullSimSignalwithFullSimPUD98")

process = cms.Process('DigiStudy')

fileInput = "file:/eos/user/s/sarkar/Simulation/PUMixing/" + options.IPType + ".root"
fileName = "EC" + options.IPType + ".root"

print("Input file:    ", fileInput)
print("Output file:   ", fileName)

process.load("Configuration.Geometry.GeometryExtended2023Reco_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.EcalDigis.ecalDigiStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2023_realistic', '')

#process.MessageLogger.EcalDigiStudy=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fileInput) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.raw2digi_step = cms.Path(process.RawToDigi)
process.analysis_step = cms.Path(process.ecalDigiStudy)
#process.ecalDigiStudy.verbose = True

# Schedule definition
process.schedule = cms.Schedule(
    process.raw2digi_step,
    process.analysis_step)
