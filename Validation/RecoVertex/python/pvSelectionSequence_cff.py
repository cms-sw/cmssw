import FWCore.ParameterSet.Config as cms

# one PV

goodVertices = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
#   cut = cms.string("!isFake && ndof >= 5 && abs(z) <= 15 && position.Rho <= 2"),  # old cut
#   cut = cms.string("!isFake && ndof >= 5 && abs(z) <= 24 && position.Rho <= 2"),  
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 30 && position.Rho <= 2"),  
   filter = cms.bool(False),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

noFakeVertices = goodVertices.clone(cut=cms.string("!isFake"))

goodVerticesD0s5 = goodVertices.clone(src = cms.InputTag("offlinePrimaryVerticesD0s5"))
goodVerticesD0s51mm = goodVertices.clone(src = cms.InputTag("offlinePrimaryVerticesD0s51mm"))
goodVerticesDA100um = goodVertices.clone(src = cms.InputTag("offlinePrimaryVerticesDA100um"))


seqPVSelection = cms.Sequence(goodVertices + noFakeVertices + goodVerticesD0s5 + goodVerticesD0s51mm + goodVerticesDA100um)
