import FWCore.ParameterSet.Config as cms

process = cms.Process("PIONS")

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

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'root://xrootd.ba.infn.it//store/user/lenzip/lenzip5Ks/lenzip5KsRECO2/323f4e3f0cfc0aa7f22eba77a06ff9df/reco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_82_5_8jA.root'
    ),
    skipEvents = cms.untracked.uint32( 282 )
)

process.PionTracksProducer = cms.EDProducer('PionTracksProducer',
    src = cms.InputTag('generalV0Candidates','Kshort')
)

process.piPath = cms.Path(process.PionTracksProducer)

process.load("TrackPropagation.Geant4e.geantRefit_cff")
#Geant4eTrackRefitter.src = cms.InputTag("PionTracksProducer","pionTrack","PIONS")
process.g4RefitPath = cms.Path( process.geant4eTrackRefit )

from RecoVertex.V0Producer.generalV0Candidates_cff import *
process.generalV0CandidatesModified = generalV0Candidates.clone(
  trackRecoAlgorithm = cms.InputTag('Geant4eTrackRefitter'),
  selectLambdas = cms.bool(False)
)
process.v0 = cms.Path(process.generalV0CandidatesModified)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root'),
    outputCommands = cms.untracked.vstring("drop *", "keep *_*_*_PIONS", "keep *_generalTracks_*_*", "keep *_generalV0Candidates_*_*")
)

  
process.p = cms.Path(process.PionTracksProducer)

process.e = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.piPath, process.g4RefitPath, process.v0, process.e)

