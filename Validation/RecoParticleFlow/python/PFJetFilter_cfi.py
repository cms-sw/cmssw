import FWCore.ParameterSet.Config as cms

pfJetFilter = cms.EDFilter("PFJetFilter",
    # Gen Jets
    InputTruthLabel = cms.InputTag('ak4GenJets'),
    # Reco Jets
    InputRecoLabel = cms.InputTag('iterativeCone5PFJets'),
    # Pseudo-rapidity cut for the reconstructed jet.
    minEta = cms.double(-1),
    maxEta = cms.double(2.5),
    # Pt cut for the reconstructed jet
    recPt = cms.double(10.0),
    # Pt cut for the generated jet
    genPt = cms.double(15.0),
    # No reconstructed jets with pt > recPt closer than this value
    deltaRMin = cms.double(1.0),
    # Gen Jet and Reco Jet must be closer than this value
    deltaRMax = cms.double(0.2),
    # Reco Pt at least 6 sigma below or 5 sigma above gen Pt
    minDeltaEt = cms.double(-6.),
    maxDeltaEt = cms.double(+6.),
    # debug level
    verbose = cms.bool(False)
                           
)
