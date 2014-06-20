import FWCore.ParameterSet.Config as cms

L1TkElectrons = cms.EDProducer("L1TkElectronTrackProducer",
	label = cms.string("EG"),	# labels the collection of L1TkEmParticleProducer that is produced.
					# (not really needed actually)
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
     	#L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
	L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigisLargerPhi","Level1TTTracks"),
        # Quality cuts on Track and Track L1EG matching criteria                                
        TrackChi2           = cms.double(1e10), # minimum Chi2 to select tracks
        TrackMinPt          = cms.double(10.0), # minimum Pt to select tracks                                     
	useTwoStubsPT       = cms.bool( True ),
        TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0), # functional Delta Phi cut parameters to match Track with L1EG objects
        TrackEGammaDeltaR   = cms.vdouble(0.08, 0.0, 0.0), # functional Delta R cut parameters to match Track with L1EG objects
        TrackEGammaDeltaEta = cms.double(1e10), # Delta Eta cutoff to match Track with L1EG objects
                                                # are considered. (unused in default configuration)
	RelativeIsolation = cms.bool( True ),	# default = True. The isolation variable is relative if True,
						# else absolute.
        IsoCut = cms.double( -0.10 ), 		# Cut on the (Trk-based) isolation: only the L1TkEmParticle for which
                                                # the isolation is below RelIsoCut are written into
                                                # the output collection. When RelIsoCut < 0, no cut is applied.
						# When RelativeIsolation = False, IsoCut is in GeV.
        # Determination of the isolation w.r.t. L1Tracks :
        PTMINTRA = cms.double( 2. ),	# in GeV
	DRmin = cms.double( 0.03),
	DRmax = cms.double( 0.2 ),
	DeltaZ = cms.double( 0.6 )    # in cm. Used for tracks to be used isolation calculation
)
# for  LowPt Electron
L1TkElectronsLoose = L1TkElectrons.clone()
L1TkElectronsLoose.TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0)
L1TkElectronsLoose.TrackEGammaDeltaR = cms.vdouble(0.12, 0.0, 0.0)
L1TkElectronsLoose.TrackMinPt = cms.double( 3.0 )
