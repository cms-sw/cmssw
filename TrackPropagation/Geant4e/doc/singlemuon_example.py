import FWCore.ParameterSet.Config as cms
import copy



# Basic process setup ----------------------------------------------------------
process = cms.Process("G4eRefit")
process.source = cms.Source("PoolSource", 
#    fileNames = cms.untracked.vstring( '/store/caf/user/lenzip/Geant4e/TwentyKsMC/digi-reco/digireco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_1.root') ,
#    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/mc/Phys14DR/DYJetsToLL_M-50_13TeV-madgraph-pythia8-tauola_v2/GEN-SIM-RAW/AVE30BX50_tsg_PHYS14_ST_V1-v1/30000/000055D9-148B-E411-86DF-20CF3027A560.root'),
    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/relval/CMSSW_7_4_0_pre5_ROOT6/RelValADDMonoJet_d3MD3_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/1A00E748-AFA2-E411-8D85-0025905A60A8.root'),

    skipEvents = cms.untracked.uint32( 0 )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
#-------------------------------------------------------------------------------

# Includes + Global Tag --------------------------------------------------------
process.load("FWCore/MessageService/MessageLogger_cfi")
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
#process.load('RecoJets/Configuration/RecoJetAssociations_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.EventContent.EventContent_cff")

# Includes + Global Tag --------------------------------------------------------
process.load("FWCore/MessageService/MessageLogger_cfi")
process.load("TrackPropagation.Geant4e.geantRefit_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')
#-------------------------------------------------------------------------------

# Reduce amount of messages ----------------------------------------------------
#process.MessageLogger.default = cms.untracked.PSet(ERROR = cms.untracked.PSet(limit = cms.untracked.int32(5)))
#process.MessageLogger.cerr.FwkReport.reportEvery = 100
#-----------------------------------------------------------
process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
  'keep *', 
  #'keep *_laserAlignmentT0Producer_*_*'
  ),
  fileName = cms.untracked.string( 'test.root' )
)


## measurementtrackerevent needs to be added explicitly here
# this solution thanks to: https://github.com/lceard/cmssw/commit/b47b82220a01c78e5415bf3fdc9b7c58c362a70e
process.g4RefitPath = cms.Path( process.MeasurementTrackerEvent * process.geant4eTrackRefit )
process.e = cms.EndPath(process.out)
process.schedule = cms.Schedule( process.g4RefitPath, process.e )
