import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

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
process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        '/store/user/lenzip/lenzip20Ks/lenzip20KsRECO2/d39ee304f24de94f181c58ee7fe85f7d/reco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_100_1_s2f.root'
    )
)

process.load("RecoTracker.TrackProducer.TrackRefitter_cfi")

#from RecoVertex.V0Producer.generalV0Candidates_cff import *

#process.generalV0CandidatesModified = generalV0Candidates.clone(
#  trackRecoAlgorithm = "TrackRefitter",
#  tkDCACut = 100.,
#  impactParameterSigCut = 0.,
#  tkChi2Cut = 100.,
#  tkNhitsCut = 0,
#  vtxChi2Cut = 20.,
#  innerHitPosCut = -1,
#  mPiPiCut = 10
#)

#process.v0 = cms.Path(process.generalV0CandidatesModified)
process.out = cms.OutputModule("PoolOutputModule",
                fileName = cms.untracked.string("prova_refit_st.root"),
                outputCommands = cms.untracked.vstring("drop *", "keep *_*_*_Demo")

        )

#process.demo = cms.EDAnalyzer('KsAnalyzer',
#                   vertex = cms.untracked.InputTag('generalV0Candidates','Kshort'),
#                   dEtaMaxCut = cms.untracked.double(10)

#)

#process.TFileService = cms.Service("TFileService",
#               fileName = cms.string('prova_st.root')
#)
process.p = cms.Path(process.TrackRefitter)
process.e = cms.EndPath(process.out)
process.schedule = cms.Schedule( process.p,  process.e )
