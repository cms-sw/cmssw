import FWCore.ParameterSet.Config as cms

L1TkHTMissCalo = cms.EDProducer("L1TkHTMissProducer",
     L1TkJetInputTag = cms.InputTag("L1TkJets","Central"),
     JET_ETAMAX = cms.double(2.2),          # maximum eta of jets for HT
     JET_PTMIN = cms.double(15.0),          # minimum pt of jets for HT [GeV]
     DoVtxConstrain = cms.bool(False),      # turn on/off applying any vertex constraint
     DeltaZ = cms.double(1.0),              # require jets to have |z_jet - z_ref| below DeltaZ [cm]
     PrimaryVtxConstrain = cms.bool(False), # use primary vertex instead of leading jet as reference z position
     L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex")
)

L1TkHTMissVtx = L1TkHTMissCalo.clone()
L1TkHTMissVtx.DoVtxConstrain = cms.bool( True )



