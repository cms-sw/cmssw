import FWCore.ParameterSet.Config as cms


#
# --- Muons from Slava
#

from L1Trigger.L1ExtraFromDigis.l1extraMuExtended_cfi import *
from SLHCUpgradeSimulations.L1TrackTrigger.l1TkMuonsExt_cff import *

# this is based on all GMTs available (BX=0 is hardcoded )                                                                               
l1tkMusFromExtendedAllEta =  cms.Sequence(l1extraMuExtended * l1TkMuonsExt )

# this is based on CSCTF record directly (no GMT sorting) and creates TkMus in |eta| > 1.1                         
l1tkMusFromExtendedForward = cms.Sequence(l1extraMuExtended * l1TkMuonsExtCSC )


#
# --- FInal collection of L1TkMuons
#

L1TkMuonsMerge = cms.EDProducer("L1TkMuonMerger",
   TkMuonCollections = cms.VInputTag( 
				      #cms.InputTag("L1TkMuonsDT","DTMatchInwardsTTTrackFullReso"),
                                      cms.InputTag("l1TkMuonsExt",""),
                                      cms.InputTag("l1TkMuonsExtCSC","") ),
   absEtaMin = cms.vdouble( 0. , 1.1),      # Padova's not ready yet
   absEtaMax = cms.vdouble( 1.1 , 5.0)
)


L1TkMuons = cms.Sequence( l1tkMusFromExtendedAllEta + l1tkMusFromExtendedForward + L1TkMuonsMerge )


