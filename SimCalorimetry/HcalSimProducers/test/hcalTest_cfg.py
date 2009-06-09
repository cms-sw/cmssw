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
process.load('RecoLocalCalo/Configuration//hcalLocalReco_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)


readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
readFiles.extend( [
#       '/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0007/2CAA01DE-514F-DE11-9807-0030487A1FEC.root',
#       '/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0006/0A13DB82-884E-DE11-8562-000423D6A6F4.root' ] );
#'/store/relval/CMSSW_3_1_0_pre9/RelValSinglePiPt10/GEN-SIM-RECO/IDEAL_31X_v1/0000/FA7A805B-584E-DE11-A682-0018F3D096E6.root',
#       '/store/relval/CMSSW_3_1_0_pre9/RelValSinglePiPt10/GEN-SIM-RECO/IDEAL_31X_v1/0000/7E5EBE7B-5F4F-DE11-A2A3-003048678B84.root' ] );
       '/store/relval/CMSSW_3_1_0_pre9/RelValSinglePiPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/DABD367E-574E-DE11-9B60-001A92971B26.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValSinglePiPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/6A5D4F75-594E-DE11-93E2-0018F3D095EC.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValSinglePiPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/5224C821-584E-DE11-AC6C-0018F3D096D4.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValSinglePiPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/50FAB27C-574E-DE11-985A-0018F3D096C8.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValSinglePiPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/500B2875-5F4F-DE11-80AB-003048678A72.root' ] );

process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


process.load('SimCalorimetry.Configuration.hcalDigiSequence_cff')
process.simHcalUnsuppressedDigis.doNoise = False
process.simHcalUnsuppressedDigis.doEmpty = False

process.hcalRecHitDump = cms.EDAnalyzer("HcalRecHitDump")
process.hcalHitAnalyzer =  cms.EDAnalyzer("HcalHitAnalyzer", process.hcalSimBlock)
# Other statements
process.GlobalTag.globaltag = 'STARTUP_31X::All'
process.hbhereco.digiLabel = 'simHcalDigis'
process.horeco.digiLabel = 'simHcalDigis'
process.hfreco.digiLabel = 'simHcalDigis'
process.zdcreco.digiLabel = 'simHcalUnsuppressedDigis'


# Path and EndPath definitions
process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.prod = cms.EDFilter("SimHitCaloHitDumper")

#process.psim to run geant4
process.simulation_step = cms.Path(process.mix*process.hcalLocalRecoSequence*process.hcalHitAnalyzer)

# Schedule definition
#process.schedule = cms.Schedule(process.generation_step,process.simulation_step)
process.schedule = cms.Schedule(process.simulation_step)
