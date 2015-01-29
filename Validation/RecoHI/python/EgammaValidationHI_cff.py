import FWCore.ParameterSet.Config as cms

##########################################################
# produce generated paricles in acceptance               #
##########################################################

higenp = cms.EDFilter("PdgIdAndStatusCandViewSelector",
    status = cms.vint32(3),
    src = cms.InputTag("genParticles"),
    pdgId = cms.vint32(22)
)

hifiducial = cms.EDFilter("EtaPtMinCandViewSelector",
    src = cms.InputTag("higenp"),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5), 
    ptMin = cms.double(2.0) 
)

hltlabel = "HLT"

from Validation.RecoHI.HLT_HIPhoton15_DQM_cfi import *
HLT_HIPhoton15_DQM.cutcollection = cms.InputTag("hifiducial")
HLT_HIPhoton15_DQM.cutnum = cms.int32(1)
HLT_HIPhoton15_DQM.pdgGen = cms.int32(22)
HLT_HIPhoton15_DQM.triggerobject.setProcessName(hltlabel)

for filterpset in HLT_HIPhoton15_DQM.filters:
    getattr(filterpset,'HLTCollectionLabels').setProcessName(hltlabel)
    for isocollections in getattr(filterpset,'IsoCollections'):
        isocollections.setProcessName(hltlabel)

# prevalidation sequence for filters / producers
hiEgammaPrevalidationSequence = cms.Sequence( higenp * hifiducial )

# validation sequence
hiEgammaValidationSequence = cms.Sequence( HLT_HIPhoton15_DQM )
