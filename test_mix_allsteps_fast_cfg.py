import FWCore.ParameterSet.Config as cms

process = cms.Process("DIGI")

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('FastSimulation.Configuration.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('FastSimulation.Configuration.Geometries_MC_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load("FastSimulation/Configuration/Reconstruction_Tk_cff")
process.load('FastSimulation.Configuration.Digi_cff')
process.load('FastSimulation.Configuration.SimL1Emulator_cff')
process.load('FastSimulation.Configuration.DigiToRaw_cff')
process.load('FastSimulation.Configuration.RawToDigi_cff')
process.load("FastSimulation/Configuration/Reconstruction_NoTk_cff")

#process.load("FastSimulation/Configuration/Reconstruction_Tk_cff")
#process.load("FastSimulation/Configuration/RawToDigi_cff")
#process.load("FastSimulation/Configuration/Reconstruction_NoTk_cff")

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:sim.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('digi.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.step0 = cms.Path(process.fastTkReconstruction)
process.step = cms.Path(process.pdigi)
process.step2 = cms.Path(process.SimL1Emulator)
process.step3 = cms.Path(process.DigiToRaw )
process.step4 = cms.Path(process.RawToDigi )
process.step5 = cms.Path(process.reconstruction )
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

process.schedule = cms.Schedule(process.step0,process.step,process.step2,process.step3,process.step4,process.step5,process.FEVTDEBUGoutput_step)
# Automatic addition of the customisation function from FastSimulation.Configuration.MixingModule_Full2Fast
from FastSimulation.Configuration.MixingModule_Full2Fast import prepareDigiRecoMixing 

#call to customisation function prepareDigiRecoMixing imported from FastSimulation.Configuration.MixingModule_Full2Fast
process = prepareDigiRecoMixing(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# End of customisation functions
