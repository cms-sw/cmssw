import FWCore.ParameterSet.Config as cms

defaultMuonHistoParameters = cms.PSet(

    minNTracks = cms.int32(0), 
    maxNTracks = cms.int32(100),  
    nintNTracks = cms.int32(100),  
    #
    minFTracks = cms.int32(0),                                                                                                      
    maxFTracks = cms.int32(20),                                                                                                    
    nintFTracks = cms.int32(20),    
    #
    nintdR = cms.int32(200),
    mindR = cms.double(0.),
    maxdR = cms.double(10.),
    #
    useFabsEta = cms.bool(False),
    minEta = cms.double(-2.5),
    maxEta = cms.double(2.5),
    nintEta = cms.int32(50),
    #
    minPt = cms.double(0.9),
    maxPt = cms.double(2000.),
    nintPt = cms.int32(50),
    useLogPt=cms.untracked.bool(True),
    useInvPt = cms.bool(False),
    #
    minPhi = cms.double(-3.1416),
    maxPhi = cms.double(3.1416),
    nintPhi = cms.int32(36),
    #
    minDxy = cms.double(-2.),
    maxDxy = cms.double(2.),
    nintDxy = cms.int32(40),
    #
    minDz = cms.double(-30.),
    maxDz = cms.double(30.),
    nintDz = cms.int32(60),
    #
    minRpos = cms.double(0.),
    maxRpos = cms.double(4.),
    nintRpos = cms.int32(40),
    #
    minZpos = cms.double(-30.),
    maxZpos = cms.double(30.),
    nintZpos = cms.int32(60),
    #
    minPU = cms.double(-0.5),                            
    maxPU = cms.double(199.5),
    nintPU = cms.int32(100),
    #
    # switches to be set according to the input Track collection to properly count the number of hits in Eff vs N(SimHits)
    usetracker = cms.bool(True),
    usemuon = cms.bool(True),
    #
    # here set for GLB tracks, redefined for TRK and STA tracks
    minNHit = cms.double(-0.5),                            
    maxNHit = cms.double(80.5),
    nintNHit = cms.int32(81),
    #
    # select doing TRK/MUO hits plots
    do_TRKhitsPlots = cms.bool(True),
    do_MUOhitsPlots = cms.bool(True),
    #
    minDTHit = cms.double(-0.5),                            
    maxDTHit = cms.double(50.5),
    nintDTHit = cms.int32(51),
    #
    minCSCHit = cms.double(-0.5),                            
    maxCSCHit = cms.double(50.5),
    nintCSCHit = cms.int32(51),
    #
    minRPCHit = cms.double(-0.5),                            
    maxRPCHit = cms.double(10.5),
    nintRPCHit = cms.int32(11),
    #
    minLayers = cms.double(-0.5),                            
    maxLayers = cms.double(20.5),
    nintLayers = cms.int32(21),
    #
    minPixels = cms.double(-0.5),                            
    maxPixels = cms.double(5.5),
    nintPixels = cms.int32(6),
    #
    ptRes_nbin = cms.int32(200),                 
    ptRes_rangeMin = cms.double(-0.5),
    ptRes_rangeMax = cms.double(0.5),
    #
    phiRes_nbin = cms.int32(200),
    phiRes_rangeMin = cms.double(-0.01),
    phiRes_rangeMax = cms.double(0.01),
    #
    etaRes_nbin = cms.int32(100),               
    etaRes_rangeMin = cms.double(-0.02),
    etaRes_rangeMax = cms.double(0.02),
    #
    cotThetaRes_nbin = cms.int32(100),
    cotThetaRes_rangeMin = cms.double(-0.01),
    cotThetaRes_rangeMax = cms.double(0.01),
    #
    dxyRes_nbin = cms.int32(100),         
    dxyRes_rangeMin = cms.double(-0.02), 
    dxyRes_rangeMax = cms.double(0.02),    
    #
    dzRes_nbin = cms.int32(100),             
    dzRes_rangeMin = cms.double(-0.05),
    dzRes_rangeMax = cms.double(0.05)
)

#####################################################################################
# TRK tracks
trkMuonHistoParameters =  defaultMuonHistoParameters.clone(
    usetracker = True,
    usemuon = False,
    nintNHit = 41,
    maxNHit = 40.5,
    do_TRKhitsPlots = True,
    do_MUOhitsPlots = False
)
#####################################################################################
# GEMmuon tracks
gemMuonHistoParameters =  trkMuonHistoParameters.clone(
    usetracker = True,
    usemuon = False,
    minEta = -2.8,
    maxEta = +2.8,
    nintEta = 48,
    #nintNHit = 41,   # this is the tracker default
    #maxNHit = 40.5,
    do_TRKhitsPlots = True,
    do_MUOhitsPlots = True  # is this used in the current code ?
)
#####################################################################################
# ME0muon tracks
me0MuonHistoParameters =  trkMuonHistoParameters.clone(
    usetracker = True,
    usemuon = False,
    minEta = -2.8,
    maxEta = +2.8,
    nintEta = 56,
    #nintNHit = 41,   # this is the tracker default
    #maxNHit = 40.5,
    do_TRKhitsPlots = True,
    do_MUOhitsPlots = True  # is this used in the current code ?
)
#####################################################################################
# STA tracks
staMuonHistoParameters = defaultMuonHistoParameters.clone(
    usetracker = False,
    usemuon = True,
    nintNHit = 61,
    maxNHit = 60.5,
    do_TRKhitsPlots = False,
    do_MUOhitsPlots = True,
    ##
    nintDxy = 40,
    minDxy = -10.,
    maxDxy = 10.,
    ##
    ptRes_nbin = 200,
    ptRes_rangeMin = -1.,
    ptRes_rangeMax = 5.,
    ##
    phiRes_nbin = 200,
    phiRes_rangeMin = -0.1,
    phiRes_rangeMax = 0.1,
    ##
    etaRes_nbin = 100,
    etaRes_rangeMin = -0.1,
    etaRes_rangeMax = 0.1,
    ##
    cotThetaRes_nbin = 100,
    cotThetaRes_rangeMin = -0.1,
    cotThetaRes_rangeMax = 0.1,
    ##
    dxyRes_nbin = 100,
    dxyRes_rangeMin = -10.,
    dxyRes_rangeMax = 10.,
    ##
    dzRes_nbin = 100,
    dzRes_rangeMin = -25.,
    dzRes_rangeMax = 25.
)
#####################################################################################
# STA seeds (here hits are counting DT,CSC segments rather than individual hit layers)
staSeedMuonHistoParameters = staMuonHistoParameters.clone(
    nintNHit = 7,
    maxNHit = 6.5,
    nintDTHit = 7,
    maxDTHit = 6.5,
    nintCSCHit = 7,
    maxCSCHit = 6.5,
    nintRPCHit = 7,
    maxRPCHit = 6.5
)
#####################################################################################
# STA Upd tracks
staUpdMuonHistoParameters = staMuonHistoParameters.clone(
    dxyRes_nbin = 100,
    dxyRes_rangeMin = -1.,
    dxyRes_rangeMax = 1.
)
#####################################################################################
# GLB tracks
glbMuonHistoParameters =  defaultMuonHistoParameters.clone(
    usetracker = True,
    usemuon = True,
    nintNHit = 81,
    maxNHit = 80.5,
    do_TRKhitsPlots = True,
    do_MUOhitsPlots = True
)
#####################################################################################                                             
# Reco Muon tracks                                                                                                                
recoMuonHistoParameters =  defaultMuonHistoParameters.clone(                                                                     
    usetracker = True,                                                                                         
    usemuon = True,                                                                                        
    nintNHit = 81,                                                                                             
    maxNHit = 80.5,                                                                                            
    do_TRKhitsPlots = True,                                                                                    
    do_MUOhitsPlots = True         
)
#####################################################################################
# Displaced TRK tracks
displacedTrkMuonHistoParameters = trkMuonHistoParameters.clone(
    nintDxy = 85,
    minDxy = -85.,
    maxDxy = 85.,
    #
    nintDz = 84,
    minDz = -210.,
    maxDz = 210.,
    #
    nintRpos = 85,
    minRpos = 0.,
    maxRpos = 85.,
    #
    nintZpos = 84,
    minZpos = -210.,
    maxZpos = 210.
)
#####################################################################################
# Displaced muons: STA tracks
displacedStaMuonHistoParameters = staMuonHistoParameters.clone(
    nintDxy = 85,
    minDxy = -85.,
    maxDxy = 85.,
    #
    nintDz = 84,
    minDz = -210.,
    maxDz = 210.,
    #
    nintRpos = 85,
    minRpos = 0.,
    maxRpos = 85.,
    #
    nintZpos = 84,
    minZpos = -210.,
    maxZpos = 210.
)
#####################################################################################
# Displaced muons: STA seeds (here hits are counting DT,CSC segments rather than individual hit layers)
displacedStaSeedMuonHistoParameters = displacedStaMuonHistoParameters.clone(
    nintNHit = 7,
    maxNHit = 6.5,
    nintDTHit = 7,
    maxDTHit = 6.5,
    nintCSCHit = 7,
    maxCSCHit = 6.5,
    nintRPCHit = 7,
    maxRPCHit = 6.5
)
#####################################################################################
# Displaced muons: GLB tracks
displacedGlbMuonHistoParameters = glbMuonHistoParameters.clone(
    nintDxy = 85,
    minDxy = -85.,
    maxDxy = 85.,
    #
    nintDz = 84,
    minDz = -210.,
    maxDz = 210.,
    #
    nintRpos = 85,
    minRpos = 0.,
    maxRpos = 85.,
    #
    nintZpos = 84,
    minZpos = -210.,
    maxZpos = 210.
)
#####################################################################################
# COSMIC muons
#####################################################################################
# cosmics: TRK tracks (2-legs)
trkCosmicMuonHistoParameters = trkMuonHistoParameters.clone(
    nintDxy = 40,
    minDxy = -10., 
    maxDxy = 10.,
    #
    nintDz = 50,
    minDz = -50.,
    maxDz = 50.,
    #
    nintRpos = 40, 
    minRpos = 0.,
    maxRpos = 10.,
    #
    nintZpos = 50,
    minZpos = -50.,
    maxZpos = 50.
)
#####################################################################################
# cosmics: STA tracks (2-legs)
staCosmicMuonHistoParameters = staMuonHistoParameters.clone(
    nintDxy = 40,
    minDxy = -10., 
    maxDxy = 10.,
    #,
    nintDz = 50,
    minDz = -50.,
    maxDz = 50.,
    #
    nintRpos = 40, 
    minRpos = 0.,
    maxRpos = 10.,
    #
    nintZpos = 50,
    minZpos = -50.,
    maxZpos = 50.
)
#####################################################################################
# cosmics: GLB tracks (2-legs)
glbCosmicMuonHistoParameters = glbMuonHistoParameters.clone(
    nintDxy = 40,
    minDxy = -10., 
    maxDxy = 10.,
    #
    nintDz = 50,
    minDz = -50.,
    maxDz = 50.,
    #
    nintRpos = 40, 
    minRpos = 0.,
    maxRpos = 10.,
    #
    nintZpos = 50,
    minZpos = -50.,
    maxZpos = 50.
)
#####################################################################################
# cosmics: TRK tracks (1-leg)
trkCosmic1LegMuonHistoParameters = trkCosmicMuonHistoParameters.clone(
    nintNHit = 81,
    maxNHit = 80.5,
    #
    nintLayers = 31,
    maxLayers = 30.5,
    #
    nintPixels = 11,
    maxPixels = 10.5
)
#####################################################################################
# cosmics: STA tracks (1-leg)
staCosmic1LegMuonHistoParameters = staCosmicMuonHistoParameters.clone(
    nintNHit = 121,
    maxNHit = 120.5,
    #
    nintDTHit = 101,
    maxDTHit = 100.5,
    #
    nintCSCHit = 101,
    maxCSCHit = 100.5,
    #
    nintRPCHit = 21,
    maxRPCHit = 20.5
)
#####################################################################################
# cosmics: GLB tracks (1-leg)
glbCosmic1LegMuonHistoParameters = glbCosmicMuonHistoParameters.clone(
    nintNHit = 161,
    maxNHit = 160.5,
    #
    nintDTHit = 101,
    maxDTHit = 100.5,
    #
    nintCSCHit = 101,
    maxCSCHit = 100.5,
    #
    nintRPCHit = 21,
    maxRPCHit = 20.5,
    #
    nintLayers = 31, 
    maxLayers = 30.5,
    #
    nintPixels = 11,
    maxPixels = 10.5
)

## Customize ranges for phase 2 samples 
# TRK tracks                                                                                                                     
trkMuonHistoParameters_phase2 = trkMuonHistoParameters.clone(
    minPU = 150,
    maxPU = 250
)
# GEMmuon tracks                                                                                                                 
gemMuonHistoParameters_phase2 = gemMuonHistoParameters.clone(       
    minPU = 150,
    maxPU = 250,
    maxNTracks = 150,
    nintNTracks = 100,
    maxFTracks = 50,
    nintFTracks = 50
)
# STA tracks                                                                                                                      
staMuonHistoParameters_phase2 = staMuonHistoParameters.clone(
    minPU = 150,
    maxPU = 250
)
# STA seeds (here hits are counting DT,CSC segments rather than individual hit layers)                                            
staSeedMuonHistoParameters_phase2 = staSeedMuonHistoParameters.clone(
    minPU = 150,
    maxPU = 250
)
# STA Upd tracks                                                                                                                  
staUpdMuonHistoParameters_phase2 = staUpdMuonHistoParameters.clone(
    minPU = 150, 
    maxPU = 250
)
# GLB tracks                                                                                                                      
glbMuonHistoParameters_phase2 = glbMuonHistoParameters.clone(
    minPU = 150,
    maxPU = 250
)
#RecoMuon tracks
recoMuonHistoParameters_phase2 = recoMuonHistoParameters.clone(
    minPU = 150,
    maxPU = 250,
    maxNTracks = 150,
    nintNTracks = 100,
    maxFTracks = 50,
    nintFTracks = 50
)
# Displaced TRK tracks  
displacedTrkMuonHistoParameters_phase2 = displacedTrkMuonHistoParameters.clone(
    minPU = 150,
    maxPU = 250
)
# Displaced muons: STA tracks                                                                                                    
displacedStaMuonHistoParameters_phase2 = displacedStaMuonHistoParameters.clone(
    minPU = 150,
    maxPU = 250
)
# Displaced muons: GLB tracks                                                                                                     
displacedGlbMuonHistoParameters_phase2 = displacedGlbMuonHistoParameters.clone(
    minPU = 150,
    maxPU = 250
)
