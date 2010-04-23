import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
exoticaHTHLT = hltHighLevel
#Define the HLT path to be used.
#exoticaHTHLT.HLTPaths =['HLT_L1Jet6U']
exoticaHTHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")

#Define the HLT quality cut 

exoticaHLTHTFilter = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    member  = cms.InputTag("hltJet15UHt","","HLT"),				
    cut     = cms.string("pt>250"),                     
    minN    = cms.int32(1)                  
)
                               
# Define the Reco quality cut using L2L3CorJetAK5Calo
from JetMETCorrections.Configuration.L2L3Corrections_Summer09_7TeV_ReReco332_cff import *
#from JetMETCorrections.Configuration.L2L3Corrections_900GeV_cff import *
exoticaRecoHT = cms.EDProducer("METProducer",
   src = cms.InputTag('L2L3CorJetAK5Calo'),
   METType = cms.string('MET'),
   alias = cms.string('EXOHT'),
   noHF = cms.bool(False),
   globalThreshold = cms.double(30.0),
   InputType = cms.string('CaloJetCollection')
)

exoticaRecoHTFilter = cms.EDFilter("HLTGlobalSumsMET",
   inputTag = cms.InputTag("exoticaRecoHT"),
   saveTag = cms.untracked.bool( True ),                             
   observable = cms.string( "sumEt" ),                           
   Min = cms.double(150.0),  
   Max = cms.double( -1.0 ),                             
   MinN = cms.int32(1)
)

#Define group sequence, using HLT bits + either HLT/Reco quality cut. 
exoticaHTHLTQualitySeq = cms.Sequence(
   exoticaHTHLT+exoticaHLTHTFilter
   
)
exoticaHTRecoQualitySeq = cms.Sequence(
	exoticaHTHLT + L2L3CorJetAK5Calo + exoticaRecoHT + exoticaRecoHTFilter
)

