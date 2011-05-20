import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
lowetModR = hlt.hltHighLevel.clone()
lowetModR.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
lowetModR.HLTPaths = cms.vstring(
    "HLT_Ele32_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT*",)
lowetModR.andOr = cms.bool( True )
lowetModR.throw = cms.bool( False )

lowetsel = cms.EDFilter("GsfElectronSelector",
                        src =cms.InputTag("gsfElectrons"),
                        cut =cms.string("superCluster().get().energy()*sin(theta())>35")
                        )

lowfilter = cms.EDFilter("CandViewCountFilter",
                         src = cms.InputTag("lowetsel"),
                         minNumber = cms.uint32(1),
                         )


exoEleLowetSeqReco=cms.Sequence(lowetModR * lowetsel * lowfilter)
