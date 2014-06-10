import FWCore.ParameterSet.Config as cms


# ---------------------------------------------------------------------------

#
# --- Muons from Slava
#

from L1Trigger.L1ExtraFromDigis.l1extraMuExtended_cfi import *
from SLHCUpgradeSimulations.L1TrackTrigger.l1TkMuonsExt_cff import *

# this is based on all GMTs available (BX=0 is hardcoded )                                                                               
l1tkMusFromExtendedAllEta =  cms.Sequence(l1extraMuExtended * l1TkMuonsExt )

# this is based on CSCTF record directly (no GMT sorting) and creates TkMus in |eta| > 1.1                         
l1tkMusFromExtendedForward = cms.Sequence(l1extraMuExtended * l1TkMuonsExtCSC )

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

#
# --- Muons from the Padova algorithm
#

#################################################################################################
# now, all the DT related stuff
#################################################################################################
# to produce, in case, collection of L1MuDTTrack objects:
#process.dttfDigis = cms.Path(process.simDttfDigis)

# the DT geometry
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from SimMuon.DTDigitizer.muonDTDigis_cfi import *

##process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff")

#################################################################################################
# define the producer of DT + TK objects
#################################################################################################
from L1Trigger.DTPlusTrackTrigger.DTPlusTrackProducer_cfi import *
#DTPlusTk_step = cms.Path(process.DTPlusTrackProducer)

L1TkMuonsDT = cms.EDProducer("L1TkMuonDTProducer")

L1TkMuonsDTSequence = cms.Sequence( DTPlusTrackProducer + L1TkMuonsDT )

# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------

#
# --- FInal collection of L1TkMuons
#

	# --- using only Slava's muons :
L1TkMuonsMerge =  cms.EDProducer("L1TkMuonMerger",
   TkMuonCollections = cms.VInputTag( cms.InputTag("l1TkMuonsExt",""),
                                      cms.InputTag("l1TkMuonsExtCSC","") ),
   absEtaMin = cms.vdouble( 0. , 1.1),      
   absEtaMax = cms.vdouble( 1.1 , 5.0)
)


	# --- or using the Padova muons in the central region:
L1TkMuonsMergeWithDT = cms.EDProducer("L1TkMuonMerger",
   TkMuonCollections = cms.VInputTag( 
				      cms.InputTag("L1TkMuonsDT","DTMatchInwardsTTTrackFullReso"),
                                      cms.InputTag("l1TkMuonsExt",""),
                                      cms.InputTag("l1TkMuonsExtCSC","") ),
   absEtaMin = cms.vdouble( 0. , 1.1, 1.25),     
   absEtaMax = cms.vdouble( 1.1 , 1.25, 5.0)
)

# ---------------------------------------------------------------------------


L1TkMuons = cms.Sequence( l1tkMusFromExtendedAllEta + l1tkMusFromExtendedForward + L1TkMuonsDTSequence + L1TkMuonsMerge )


