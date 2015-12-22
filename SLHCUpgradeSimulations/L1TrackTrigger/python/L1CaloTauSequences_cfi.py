import FWCore.ParameterSet.Config as cms


L1CaloTauCorrectionsProducer = cms.EDProducer("L1CaloTauCorrectionsProducer",
     L1TausInputTag = cms.InputTag("SLHCL1ExtraParticles","Taus")
)

# Setup the L1TkTauFromCalo producer:
L1TkTauFromCaloProducer = cms.EDProducer("L1TkTauFromCaloProducer",
      L1CaloTaus_InputTag                = cms.InputTag("L1CaloTauCorrectionsProducer","CalibratedTaus"),
      L1TkTracks_InputTag                = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
      #L1TkPV_InputTag                    = cms.InputTag("L1TkPrimaryVertex"),
      L1TkPV_InputTag                    = cms.InputTag("Dummy"),
      L1TkTracks_NFitParameters          = cms.uint32(  4    ),  # Number of fit parameters: 4 or 5? (pT, eta, phi, z0, d0)
      L1TkTracks_PtMin                   = cms.double(  2.0  ),  # Min pT applied on all L1TkTracks [GeV]
      L1TkTracks_AbsEtaMax               = cms.double(  2.3  ),  # Max |eta| applied on all L1TkTracks [unitless]
      L1TkTracks_POCAzMax                = cms.double(  30.0 ),  # Max POCA-z for L1TkTracks [cm] 
      L1TkTracks_NStubsMin               = cms.uint32(  4    ),  # Min number of stubs per L1TkTrack [unitless]
      L1TkTracks_RedChiSquareBarrelMax   = cms.double(  10.0 ),  # Max chi squared for L1TkTracks in Barrel [unitless] 
      L1TkTracks_RedChiSquareEndcapMax   = cms.double(  8.0  ),  # Max chi squared for L1TkTracks in Endcap [unitless]
      L1TkTracks_DeltaPOCAzFromPV        = cms.double(10000  ),  # Max POCA-z distance from the z-vertex of the L1TkPV [cm]. (1000.0 to disable)
      L1TkTau_MatchingTk_PtMin           = cms.double( 15.0  ),  # Min pT of CaloTau-matching L1TkTracks [GeV]
      L1TkTau_MatchingTk_DeltaRMax       = cms.double(  0.1  ),  # Max deltaR of CaloTau-matching L1TkTracks [GeV] 
      L1TkTau_SignalTks_PtMin            = cms.double(  2.0  ),  # Min pT of L1TkTau signal-cone L1TkTracks [GeV]
      L1TkTau_SignalTks_DeltaRMax        = cms.double(  0.15 ),  # Max opening of the L1TkTau signal-cone for adding L1TkTracks [unitless]
      L1TkTau_SignalTks_InvMassMax       = cms.double(  1.77 ),  # Max invariant mass of the L1TkTau when adding L1TkTracks [GeV/c^2]
      L1TkTau_SignalTks_DeltaPOCAzMax    = cms.double(  0.5  ),  # Max POCAz difference between MatchingTk and additional L1TkTau signal-cone L1TkTracks [cm]
      L1TkTau_IsolationTks_PtMin         = cms.double(  2.0  ),  # Min pT of L1TkTau isolation-annulus L1TkTracks [GeV]
      L1TkTau_IsolationTks_DeltaRMax     = cms.double(  0.30 ),  # Max opening of the L1TkTau isolation-annulus [unitless]
      L1TkTau_IsolationTks_DeltaPOCAzMax = cms.double( +1.0  ),  # Max POCAz difference between MatchingTk and L1TkTracks in isolation cone [cm] (-ve to disable)
)


#CaloTauSequence = cms.Sequence( L1TkPrimaryVertex + L1CaloTauCorrectionsProducer + L1TkTauFromCaloProducer )

CaloTauSequence = cms.Sequence( L1CaloTauCorrectionsProducer + L1TkTauFromCaloProducer )
