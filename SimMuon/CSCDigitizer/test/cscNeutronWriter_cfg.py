# Auto generated configuration file
# using: 
# Revision: 1.108 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: MinBias.cfi --step GEN,SIM --eventcontent FEVTSIM
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/Sim_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("Configuration.Generator.MinBias_cfi")
process.prod = cms.EDAnalyzer("SimHitCaloHitDumper")

process.g4SimHits.StackingAction.NeutronThreshold = 0.
process.g4SimHits.StackingAction.MaxTrackTime = 1e9
process.g4SimHits.SteppingAction.MaxTrackTime = 1e9
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_HP'
process.g4SimHits.Physics.FlagBERT = True
#  cuts on generator-level particles
process.g4SimHits.Generator.ApplyPCuts = False
process.g4SimHits.Generator.ApplyEtaCuts = False
#only affects weighting of energy deposit, so unneeded
#process.g4SimHits.CaloSD.NeutronThreshold = 0.

process.load("SimMuon.CSCDigitizer.cscNeutronWriter_cfi")

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 'keep PSimHits_cscNeutronWriter_*_*'),
    fileName = cms.untracked.string('cscNeutronWriter.root')
)

process.RandomNumberGeneratorService.cscNeutronWriter = cms.PSet(initialSeed = cms.untracked.uint32(112358))
# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'MC_3XY_V26::All'
process.source = cms.Source("EmptySource")

# Path and EndPath definitions
process.generation_step = cms.Path(process.generator*process.pgen)
process.simulation_step = cms.Path(process.psim*process.prod*process.cscNeutronWriter)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.endjob_step,process.out_step)
