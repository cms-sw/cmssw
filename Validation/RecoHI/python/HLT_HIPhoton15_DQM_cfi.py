import FWCore.ParameterSet.Config as cms

##########################################################
# See HLT Config Browser, for up-to-date HLT paths
#  http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/browser/
#
# This config is for
#  HLT_HIPhoton15
#    A single photon trigger, requiring at least one HLT photon with ET > 15 GeV.
#    No isolation is required.
#
# This path contains 5 steps:
#  HLT_HIPhoton15 = cms.Path( HLTBeginSequenceBPTX +
#                             hltHIL1sPhoton15 +
#                             hltHIPrePhoton15 +
#                             HLTDoCaloSequence +
#                             HLTDoHIEcalClusSequence +
#                             hltHIPhoton15 +
#                             HLTEndSequence )
#
# The two filter steps:
#   1. hltHIL1sPhoton15
#   2. hltHIPhoton15
#
# ...are what go into the "HLTCollectionLabels" below.
##########################################################


HLT_HIPhoton15_DQM = cms.EDAnalyzer("EmDQM",
    triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),
    pdgGen = cms.int32(22),     
    genEtaAcc = cms.double(2.5),
    genEtAcc = cms.double(10.0),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(200.0),
    useHumanReadableHistTitles = cms.untracked.bool(False),

    # Filters from collections listed above.  theHLTOutputTypes defined at the following:
    # http://cmslxr.fnal.gov/lxr/source/DataFormats/HLTReco/interface/TriggerTypeDefs.h#030      
    filters = cms.VPSet(
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltHIL1sPhoton15","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(-82),       
            HLTCollectionHumanName = cms.untracked.string("Level 1"),
            ncandcut = cms.int32(1)
        ),
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltHIPhoton15","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(81),
            HLTCollectionHumanName = cms.untracked.string("Photon 15"),
            ncandcut = cms.int32(1)
        )
    )
)
