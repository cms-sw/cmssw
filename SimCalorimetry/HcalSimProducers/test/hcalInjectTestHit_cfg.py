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
process.load('RecoLocalCalo/Configuration/hcalLocalReco_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

#process.source = cms.Source("EmptySource")
process.load("Configuration.Generator.SingleNuE10_cfi")

process.load('SimCalorimetry.Configuration.hcalDigiSequence_cff')
process.simHcalUnsuppressedDigis.injectTestHits = True
process.simHcalUnsuppressedDigis.doNoise = False
process.simHcalUnsuppressedDigis.doEmpty = False
process.simHcalUnsuppressedDigis.doTimeSlew = False

process.hcalRecHitDump = cms.EDAnalyzer("HcalRecHitDump")
process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.hbhereco.digiLabel = 'simHcalDigis'
process.horeco.digiLabel = 'simHcalDigis'
process.hfreco.digiLabel = 'simHcalDigis'
process.zdcreco.digiLabel = 'simHcalUnsuppressedDigis'


# Other statements
process.GlobalTag.globaltag = 'STARTUP_31X::All'

# Path and EndPath definitions
process.generation_step = cms.Path(process.generator*process.pgen)
process.simulation_step = cms.Path(process.psim*process.mix*process.hcalDigiSequence*process.eca*process.hcalLocalRecoSequence*process.hcalRecHitDump)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.simulation_step)
#process.schedule = cms.Schedule(process.simulation_step)
