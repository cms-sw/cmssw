import FWCore.ParameterSet.Config as cms

# File: CaloMET.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET.
metAnalyzer = cms.EDFilter("METTester",
    OutputFile = cms.untracked.string('METTester_met.root'),
    InputMETLabel = cms.InputTag("met"),
    METType = cms.untracked.string('CaloMET')
)

metNoHFAnalyzer = cms.EDFilter("METTester",
    OutputFile = cms.untracked.string('METTester_metNoHF.root'),
    InputMETLabel = cms.InputTag("metNoHF"),
    METType = cms.untracked.string('CaloMET')
)

metOptAnalyzer = cms.EDFilter("METTester",
    OutputFile = cms.untracked.string('METTester_metOpt.root'),
    InputMETLabel = cms.InputTag("metOpt"),
    METType = cms.untracked.string('CaloMET')
)

metOptNoHFAnalyzer = cms.EDFilter("METTester",
    OutputFile = cms.untracked.string('METTester_metOptNoHF.root'),
    InputMETLabel = cms.InputTag("metOptNoHF"),
    METType = cms.untracked.string('CaloMET')
)


