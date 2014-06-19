import FWCore.ParameterSet.Config as cms

L1CaloTauCorrectionsProducer = cms.EDProducer("L1CaloTauCorrectionsProducer",
     L1TausInputTag = cms.InputTag("SLHCL1ExtraParticles","Taus")
)

# Setup the L1TkTauFromCalo producer:
L1TkTauFromCaloProducer = cms.EDProducer("L1TkTauFromCaloProducer",
      #L1TausInputTag                   = cms.InputTag("SLHCL1ExtraParticles","Taus"),
      L1TausInputTag                   = cms.InputTag("L1CaloTauCorrectionsProducer","CalibratedTaus"),
      L1TrackInputTag                  = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
      L1TkTrack_ApplyVtxIso            = cms.bool( True  ),      # Produce vertex-isolated L1TkTaus?
      L1TkTrack_VtxIsoZ0Max            = cms.double( 1.0  ),     # Max vertex z for L1TkTracks for VtxIsolation [cm]
      L1TkTrack_NStubsMin              = cms.uint32(  5   ),     # Min number of stubs per L1TkTrack [unitless]
      L1TkTrack_PtMin_AllTracks        = cms.double(  2.0 ),     # Min pT applied on all L1TkTracks [GeV]
      L1TkTrack_PtMin_SignalTracks     = cms.double(  10.0),     # Min pT applied on signal L1TkTracks [GeV]
      L1TkTrack_PtMin_IsoTracks        = cms.double(  2.0 ),     # Min pT applied on isolation L1TkTracks [GeV]
      L1TkTrack_RedChiSquareEndcapMax  = cms.double(  5.0 ),     # Max red-chi squared for L1TkTracks in Endcap
      L1TkTrack_RedChiSquareBarrelMax  = cms.double(  2.0 ),     # Max red-chi squared for L1TkTracks in Barrel
      L1TkTrack_VtxZ0Max               = cms.double( 30.0 ),     # Max vertex z for L1TkTracks [cm] 
      DeltaR_L1TkTau_L1TkTrack         = cms.double( 0.10 ),     # Cone size for L1TkTracks assigned to L1TkTau
      DeltaR_L1TkTauIsolation          = cms.double( 0.40 ),     # Isolation cone size for L1TkTau
      DeltaR_L1TkTau_L1CaloTau         = cms.double( 0.15 ),     # Matching cone for L1TkTau and L1CaloTau
      L1CaloTau_EtMin                  = cms.double( 5.0  ),     # Min eT applied on all L1CaloTaus [GeV]
      RemoveL1TkTauTracksFromIsoCalculation = cms.bool( False ), # Remove tracks used in L1TkTau construction from VtxIso calculation?
)


CaloTauSequence = cms.Sequence( L1CaloTauCorrectionsProducer + L1TkTauFromCaloProducer )

