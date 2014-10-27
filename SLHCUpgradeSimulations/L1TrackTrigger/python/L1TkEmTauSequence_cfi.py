import FWCore.ParameterSet.Config as cms


# --- to run L1EGCrystalClusterProducer, one needs the ECAL RecHits:
from Configuration.StandardSequences.Reconstruction_cff import *


L1EGammaCrystalsProducer = cms.EDProducer("L1EGCrystalClusterProducer",
   EtminForStore = cms.double(-1.0),                                 
   debug = cms.untracked.bool(False),
   useECalEndcap = cms.bool(True)
)


L1TkEmTauProducer = cms.EDProducer( 'L1TkEmTauProducer' ,
                                  L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
                                  L1EmInputTag = cms.InputTag("L1EGammaCrystalsProducer","EGCrystalCluster"),
                                  ptleadcut = cms.double(5.0),
                                  ptleadcone = cms.double(0.3),
                                  masscut = cms.double(1.77),
                                  emptcut = cms.double(5.0),
                                  trketacut = cms.double(2.3),
                                  pttotcut = cms.double(5.0),
                                  isocone = cms.double(0.25),
                                  isodz = cms.double(0.6),
                                  relisocut = cms.double(0.15),
                                  chisqcut = cms.double(40.0),
                                  nstubcut = cms.int32(5),
                                  dzcut = cms.double(0.8)
)


TkEmTauSequence = cms.Sequence( calolocalreco +  L1EGammaCrystalsProducer + L1TkEmTauProducer)
