import FWCore.ParameterSet.Config as cms


# --- for the local calo reconstruction
from Configuration.StandardSequences.Reconstruction_cff import *
towerMaker.hbheInput = cms.InputTag("hbheprereco")
towerMakerWithHO.hbheInput = cms.InputTag("hbheprereco")


# --- to reconstruct the HLT HI jets 
from RecoHI.HiJetAlgos.HiRecoJets_TTI_cff import *


# --- Put the HI HLT jets into "L1Jets"
L1JetsFromHIHLTJets = cms.EDProducer("L1JetsFromHIHLTJets",
        ETAMIN = cms.double(0),
        ETAMAX = cms.double(3.),
        HIJetsInputTag = cms.InputTag("iterativeConePu5CaloJets")
)

# --- Produce L1TkJets 
from SLHCUpgradeSimulations.L1TrackTrigger.L1TkJetProducer_cfi import L1TkJets
L1TkJetsHI = L1TkJets.clone()
L1TkJetsHI.L1CentralJetInputTag = cms.InputTag("L1JetsFromHIHLTJets")
L1TkJetsHI.JET_HLTETA = cms.bool(True)


# --- Produce HT and MHT with and without the vertex constraint
from SLHCUpgradeSimulations.L1TrackTrigger.L1TkHTMissProducer_cfi import *

L1TkHTMissCaloHI = L1TkHTMissCalo.clone()
L1TkHTMissCaloHI.L1TkJetInputTag = cms.InputTag("L1TkJetsHI","Central")

L1TkHTMissVtxHI = L1TkHTMissVtx.clone()
L1TkHTMissVtxHI.L1TkJetInputTag = cms.InputTag("L1TkJetsHI","Central")


# --- the full sequence with the L1Jets :
L1TkCaloSequence = cms.Sequence( L1TkJets + L1TkHTMissCalo + L1TkHTMissVtx )

# --- the full sequence with the HLT HI jets :
L1TkCaloSequenceHI = cms.Sequence( calolocalreco + hiRecoJets + L1JetsFromHIHLTJets + L1TkJetsHI + L1TkHTMissCaloHI + L1TkHTMissVtxHI )

# --- a sequence to just reconstruct the HLT HI jets :
L1TkHIJets = cms.Sequence( calolocalreco + hiRecoJets + L1JetsFromHIHLTJets + L1TkJetsHI)


