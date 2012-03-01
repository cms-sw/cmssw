import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIM')


process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load('Configuration.StandardSequences.GeometryExtended_cff')
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandalonSelection_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'START44_V9B::All'


process.maxEvents = cms.untracked.PSet(
    output = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('file:/scratch/hh/current/cms/user/aburgmei/testFiles/_DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola_Fall11-PU_S6_START44_V9B-v1_GEN-SIM-RECO/00337699-5B3D-E111-BF90-E0CB4E1A1194.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('skimmed.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

process.p = cms.Path(process.goldenZmumuSelectionSequence)
process.outp = cms.EndPath(process.out)

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)
