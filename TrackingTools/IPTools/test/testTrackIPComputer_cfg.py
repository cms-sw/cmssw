import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("rfio:/castor/cern.ch/cms//store/relval/CMSSW_2_2_4/RelValTTbar_Tauola/GEN-SIM-RECO/STARTUP_V8_LowLumiPileUp_v1/0001/54D5C532-73F4-DD11-B065-001D09F251E0.root")                            
                            )

# conditions ------------------------------------------------------------------

process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V8::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

### for checking the btagging infos still the same
#process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.demo = cms.EDAnalyzer("testTrackIPComputer")


process.p = cms.Path(process.demo)
#process.impactParameterTagInfos) ###for rerunning the btagging

# process.out = cms.OutputModule("PoolOutputModule",
#                                fileName = cms.untracked.string('OrigCalc.root'),
#                                # save only events passing the full path
#                                SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
#                                outputCommands = cms.untracked.vstring('drop *',
#                                                                       'keep *_*_*_Demo')
#                                )

# process.outpath = cms.EndPath(process.out)
