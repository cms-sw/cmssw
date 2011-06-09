import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
exoticaMuHLT = hlt.hltHighLevel.clone()
exoticaMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
#Define the HLT path to be used.
exoticaMuHLT.HLTPaths = cms.vstring('HLT_Mu30*')

#Define the Reco quality cut
exoticaRecoMuonFilter = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('pt > 32.0'),
    filter = cms.bool(True)            
                                      
)

exoMuSequence = cms.Sequence(
    exoticaMuHLT * exoticaRecoMuonFilter
#    exoticaMuHLT 
#    exoticaRecoMuonFilter
)

