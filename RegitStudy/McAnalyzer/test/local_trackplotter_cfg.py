import FWCore.ParameterSet.Config as cms

process = cms.Process("Plot")
process.load("FWCore.MessageService.MessageLogger_cfi")

'''
process.MessageLogger.categories = cms.untracked.vstring('plot',
                                                         'FwkJob', 'FwkReport', 'FwkSummary', 'Root_NoDictionary')

process.MessageLogger.cout = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True),
    threshold = cms.untracked.string('DEBUG'),
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(1000000)
        ),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(1000000)
        ),
    plot = cms.untracked.PSet(
        limit = cms.untracked.int32(1000000)
        )
    )
'''

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# fake input file
process.source = cms.Source("EmptySource")

process.plot = cms.EDAnalyzer('McMatchTrackPlotter',
                              inFileName      = cms.untracked.string("/afs/cern.ch/work/e/echapon/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/src/RegitStudy/McAnalyzer/test/ntuples_jpsi/ntuples_pp.root"),
                              typeNumber      = cms.untracked.string("Type1"),
                              mergedRootFiles =cms.untracked.bool(False),
                              doBarel         = cms.untracked.bool(True),
                              doEndcap        = cms.untracked.bool(False),
                              doMixedCoverage = cms.untracked.bool(False), # works with one of the doBarel or doEndcap being false
                              doParentSpecificCut = cms.untracked.bool(False),
                              doSingle        = cms.untracked.bool(True),
                              doPair          = cms.untracked.bool(True),
                              cutSet          = cms.untracked.int32(-1),
                              hitFractionMatch= cms.untracked.double(0.5),
                              pdgPair         = cms.untracked.int32(443),
                              ptCutSingle     = cms.untracked.double(0.),
                              ptCutPair       = cms.untracked.double(0.),
                              massCutMax      = cms.untracked.double(4),
                              massCutMin      = cms.untracked.double(2.5),
                              reco2Sim        = cms.untracked.bool(True),
                              sim2Reco        = cms.untracked.bool(True)
                              )

process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string("/afs/cern.ch/work/e/echapon/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/src/RegitStudy/McAnalyzer/test/plots.root")
                                   )

process.p = cms.Path(process.plot)
