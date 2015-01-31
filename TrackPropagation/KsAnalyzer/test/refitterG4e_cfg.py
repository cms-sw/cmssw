import FWCore.ParameterSet.Config as cms

process = cms.Process("kappa")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) )

myfilelist = cms.untracked.vstring()

myfilelist.extend( [
'root://xrootd.ba.infn.it//store/user/lenzip/lenzip5Ks/lenzip5KsRECO2/323f4e3f0cfc0aa7f22eba77a06ff9df/reco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_82_5_8jA.root'
])

process.source = cms.Source("PoolSource",
    fileNames = myfilelist,
    skipEvents = cms.untracked.uint32( 282 )
)

process.load("TrackPropagation.Geant4e.geantRefit_cff")

process.g4RefitPath = cms.Path( process.geant4eTrackRefit )

from RecoVertex.V0Producer.generalV0Candidates_cff import *

process.generalV0CandidatesModified = generalV0Candidates.clone(
  trackRecoAlgorithm = cms.InputTag('Geant4eTrackRefitter'),
 # tkChi2Cut = cms.double(10000000),
  selectLambdas = cms.bool(False),
 # tkNhitsCut = cms.int32(0),
 # impactParameterSigCut = cms.double(0.),
 # tkDCACut = cms.double(100.),
 # vtxChi2Cut = cms.double(100.0)
)

process.v0 = cms.Path(process.generalV0CandidatesModified)

process.out = cms.OutputModule("PoolOutputModule",
                fileName = cms.untracked.string("prova_refit_G4e.root"),
                outputCommands = cms.untracked.vstring("drop *", "keep *_*_*_kappa", "keep *_generalTracks_*_*", "keep *_generalV0Candidates_*_*")

        )

process.e = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.g4RefitPath, process.v0, process.e)


