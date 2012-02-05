import FWCore.ParameterSet.Config as cms
process = cms.Process("ana")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
    'file:/storage/cluster/burgmeier/_DYToMuMu_M-20_CT10_TuneZ2_7TeV-powheg-pythia_Spring11-PU_S1_START311_V1G1-v1_AODSIM/B0298296-594F-E011-AA02-1CC1DE1CEFC8.root'
#		'file:/storage/6/zeise/temp/DYToMuMu_M-20_CT10_TuneZ2_7TeV-powheg-pythia_Fall10-START38_V12-v1_A201343B-91CF-DF11-A06F-0018FE287FB8.root',
#		'file:/storage/6/zeise/temp/DYToMuMu_M-20_CT10_TuneZ2_7TeV-powheg-pythia_Fall10-START38_V12-v1_FE0A9759-77CB-DF11-9899-00163E131401.root',
        )
                            )

#process.options = cms.untracked.PSet(
#        SkipEvent = cms.untracked.vstring('ProductNotFound')
#)

process.load('TauAnalysis.Skimming.goldenZmmSelectionVBTFrelPFIsolation_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.GlobalTag.globaltag = cms.string('START42_V12::All')
process.goldenZmumuSkimPath = cms.Path(
	process.goldenZmumuSelectionSequence
)

goldenZmumuEventSelection = cms.untracked.PSet(
	SelectEvents = cms.untracked.PSet(
		SelectEvents = cms.vstring('goldenZmumuSkimPath')
	)
)

from TauAnalysis.Skimming.goldenZmmEventContent_cff import *
process.goldenZmumuSkimOutputModule = cms.OutputModule("PoolOutputModule",
	goldenZmumuEventContent,
	goldenZmumuEventSelection,
	fileName = cms.untracked.string('goldenZmumuEvents.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.o = cms.EndPath(process.goldenZmumuSkimOutputModule)
