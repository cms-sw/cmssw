
import FWCore.ParameterSet.Config as cms

L1TkStubElectrons = cms.EDProducer("L1TkElectronStubsProducer",
	label = cms.string("EG"),	# labels the collection of L1TkEmParticleProducer that is produced.
                                        # e.g. EG or IsoEG if all objects are kept, or
                                        # EGIsoTrk or IsoEGIsoTrk if only the EG or IsoEG
                                        # objects that pass a cut RelIso < RelIsoCut are written
                                        # into the new collection.
        L1EGammaInputTag = cms.InputTag("SLHCL1ExtraParticlesNewClustering","EGamma"),      # input EGamma collection
					# When the standard sequences are used :
                                                #   - for the Run-1 algo, use ("l1extraParticles","NonIsolated")
                                                #     or ("l1extraParticles","Isolated")
                                                #   - for the "old stage-2" algo (2x2 clustering), use 
                                                #     ("SLHCL1ExtraParticles","EGamma") or ("SLHCL1ExtraParticles","IsoEGamma")
                                                #   - for the new clustering algorithm of Jean-Baptiste et al,
                                                #     use ("SLHCL1ExtraParticlesNewClustering","IsoEGamma") or
                                                #     ("SLHCL1ExtraParticlesNewClustering","EGamma").
        ETmin = cms.double( -1.0 ),             # Only the L1EG objects that have ET > ETmin in GeV
                                                # are considered. ETmin < 0 means that no cut is applied.
        L1StubInputTag  = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
        MCTruthInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),                                   
        StubMinPt          = cms.double(5.0), # minimum Pt to select Stubs
        StubEGammaDeltaPhi = cms.double(0.05),     # delta Phi of stub and EG
        StubEGammaDeltaZ   = cms.double(15.0),     # Z-intercept o
        StubEGammaPhiMiss  = cms.double(0.0015),     # delta Phi between a stub-pair and EG  
        StubEGammaZMiss    = cms.double(0.7),     # Z difference between a stub-pair and EG                    
        BeamSpotInputTag   = cms.InputTag("BeamSpotFromSim", "BeamSpot"), # beam spot InputTag                                            
        
        L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis", "Level1TTTracks"),
	RelativeIsolation = cms.bool( True ),	# default = True. The isolation variable is relative if True,
						# else absolute.
        IsoCut = cms.double( -0.15 ), 		# Cut on the (Trk-based) isolation: only the L1TkEmParticle for which
                                                # the isolation is below RelIsoCut are written into
                                                # the output collection. When RelIsoCut < 0, no cut is applied.
						# When RelativeIsolation = False, IsoCut is in GeV.
        # Determination of the isolation w.r.t. L1Tracks :
        PTMINTRA = cms.double( 2. ),	# in GeV
	DRmin = cms.double( 0.05),
	DRmax = cms.double( 0.4 ),
	DeltaZ = cms.double( 0.4 )    # in cm. Used for tracks to be used isolation calculation
)
