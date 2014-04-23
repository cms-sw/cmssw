import FWCore.ParameterSet.Config as cms

process = cms.Process('DIGI')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
#process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023HGCalMuon_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('RecoLocalCalo/Configuration/hcalLocalReco_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            fileNames=cms.untracked.vstring('file:/afs/cern.ch/user/p/psilva/work/HGCal/MuonEvents.root')
                            )

process.load('SimCalorimetry.Configuration.hgcDigiSequence_cff')

# Path and EndPath definitions
#process.simulation_step = cms.Path(process.mix*process.hgcalDigiSequence*process.eca*process.hcalLocalRecoSequence*process.hcalRecHitDump)
process.simulation_step = cms.Path(process.hgcDigiSequence)

# Schedule definition
process.schedule = cms.Schedule(process.simulation_step)
