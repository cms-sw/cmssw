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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

myfilelist = cms.untracked.vstring()

myfilelist.extend( [
       'root://xrootd.ba.infn.it//store/user/lenzip/lenzip5Ks/lenzip5KsRECO2/323f4e3f0cfc0aa7f22eba77a06ff9df/reco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_82_5_8jA.root' 
])

process.source = cms.Source("PoolSource",
    fileNames = myfilelist,
    skipEvents = cms.untracked.uint32( 0 )
)

process.load("TrackPropagation.Geant4e.geantRefit_cff")
#process.load('RecoVertex/V0Producer/generalV0Candidates_cff')

process.g4RefitPath = cms.Path( process.geant4eTrackRefit )

from RecoVertex.V0Producer.generalV0Candidates_cff import *

process.generalV0CandidatesModified = generalV0Candidates.clone(
  trackRecoAlgorithm = cms.InputTag('Geant4eTrackRefitter')
)

process.v0 = cms.Path(process.generalV0CandidatesModified)

process.dEta05 = cms.EDAnalyzer('KsAnalyzer',
               vertex = cms.untracked.InputTag('generalV0CandidatesModified','Kshort'),
               dEtaMaxCut = cms.untracked.double(0.5)
)

process.dEta07 = cms.EDAnalyzer('KsAnalyzer',
               vertex = cms.untracked.InputTag('generalV0CandidatesModified','Kshort'),
               dEtaMaxCut = cms.untracked.double(0.7)
)

process.dEta03 = cms.EDAnalyzer('KsAnalyzer',
               vertex = cms.untracked.InputTag('generalV0CandidatesModified','Kshort'),
               dEtaMaxCut = cms.untracked.double(0.3)
)

process.dEta5 = cms.EDAnalyzer('KsAnalyzer',
               vertex = cms.untracked.InputTag('generalV0CandidatesModified','Kshort'),
               dEtaMaxCut = cms.untracked.double(5)
)

process.TFileService = cms.Service("TFileService",
               fileName = cms.string('/afs/cern.ch/work/l/lviliani/Geant4e_G4-9.5/CMSSW_5_3_17/src/TrackPropagation/KsAnalyzer/test/outputKsAnalyzerG4e_75_1nh.root')
)

process.p = cms.Path(process.dEta05*process.dEta03*process.dEta07*process.dEta5)

process.schedule = cms.Schedule(process.g4RefitPath, process.v0, process.p)

