import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIMTF')


process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load('Configuration.StandardSequences.GeometryExtended_cff')
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandalonSelectionLegacy_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.GlobalTag.globaltag = 'START44_V9B::All'
#process.GlobalTag.globaltag = 'START42_V13::All'
process.GlobalTag.globaltag =  'FT_R_42_V13A::All'


process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1000)
)

# Input source
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('file:/scratch/scratch0/tfruboes/DATA_tmp/DoubleMu/Run2011A-ZMu-May10ReReco-v1/RAW-RECO/FE04F164-5C7C-E011-9969-0026189438F7.root')
    #fileNames = cms.untracked.vstring('file:/scratch/scratch0/tfruboes/DATA_tmp/20120227_DYmumu/fruboes-20120227_DYmumu-300f5f448f2e374c2acc9a40b5f771bd/USER/ZmumuTF_9_1_YFD.root')
    #fileNames = cms.untracked.vstring('file:/scratch/hh/current/cms/user/aburgmei/testFiles/_DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola_Fall11-PU_S6_START44_V9B-v1_GEN-SIM-RECO/00337699-5B3D-E111-BF90-E0CB4E1A1194.root')
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
