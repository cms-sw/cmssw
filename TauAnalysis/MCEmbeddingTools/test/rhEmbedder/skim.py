import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIM')


process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load('Configuration.StandardSequences.GeometryExtended_cff')
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandalonSelection_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'START42_V13::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('file:F89E8977-949B-E011-87BA-E0CB4E1A1187.root')
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


