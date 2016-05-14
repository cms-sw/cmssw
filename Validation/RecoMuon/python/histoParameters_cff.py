import FWCore.ParameterSet.Config as cms

defaultMuonHistoParameters = cms.PSet(
    useFabsEta = cms.bool(False),
    minEta = cms.double(-2.5),
    maxEta = cms.double(2.5),
    nintEta = cms.int32(50),
    #
    minPt = cms.double(0.9),
    maxPt = cms.double(2000.),
    nintPt = cms.int32(50),      # TRK ha 40 con scala LOG da 0.05 a 1 TeV - e' ok con scala LOG (vd trk)
    useLogPt=cms.untracked.bool(True),
    useInvPt = cms.bool(False),
    #
    minPhi = cms.double(-3.1416),
    maxPhi = cms.double(3.1416),
    nintPhi = cms.int32(36),
    #
    minDxy = cms.double(-2.),  # per prompt particles e' ok cosi', per displaced muons vanno cambiati !
    maxDxy = cms.double(2.),   # metto 40 bins da -10. a 10.
    nintDxy = cms.int32(40),   # TRK ha 100 bins da -25. a +25 cm !!!
    #
    minDz = cms.double(-30.),  # come TRK e' rispetto al beamspot !!!  
    maxDz = cms.double(30.),
    nintDz = cms.int32(60),
    # TP production Radius        # per non disdpalced  muons ok cosi ' ??
    minRpos = cms.double(0.),     # per displaced per es. da 0. a 85. e 85 bins
    maxRpos = cms.double(4.),    # TRK ha 60 \bins da 0. a 60cm
    nintRpos = cms.int32(40),
    # TP production Z position    # cosi' e \' il TRK (il select per displaced va fino a 210cm in Z !)
    minZpos = cms.double(-30.),    # TRK ha 60 bins da --30 a 30cm   (e' rispetto al BeamSPot -> bunch length !!
    maxZpos = cms.double(30.),     # per diaplced fino a -210 - +210 cm -> 105 bins ?
    nintZpos = cms.int32(60),
    # Number of vertices (PU summary info)
    minPU = cms.double(-0.5),                            
    maxPU = cms.double(199.5),
    nintPU = cms.int32(100),
    #
    # switches to be set according to the input Track collection to properly count the number of hits in Eff vs N(SimHits)
    usetracker = cms.bool(True),
    usemuon = cms.bool(True),
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
    # n TRK layers
    minLayers = cms.double(-0.5),                            
    maxLayers = cms.double(20.5),
    nintLayers = cms.int32(21),
    # n Pixel layers
    minPixels = cms.double(-0.5),                            
    maxPixels = cms.double(5.5),
    nintPixels = cms.int32(6),
    #
    ptRes_nbin = cms.int32(200),       # avevo provato 120 bins da -0.3 a 0.3 : la sigma a 1Tev e' circa 0.15                     
    ptRes_rangeMin = cms.double(-0.5),    # TRK ha -0.1 / 0.1   con 100 bins 
    ptRes_rangeMax = cms.double(0.5),    # provo con 120 tra -3. e 3. x STA  n.b. x pt=1000 e' ancora poco...
    #
    # TRK ha 300 bins da -0.01 a +0.01    # muo aveva 100 bins da -0.05 a 0.05
    phiRes_nbin = cms.int32(200),         # devo allargarla per STA tracks tipo 100 bins da -0.1 a 0.1
    phiRes_rangeMin = cms.double(-0.01),
    phiRes_rangeMax = cms.double(0.01),
    #
    etaRes_nbin = cms.int32(100),               
    etaRes_rangeMin = cms.double(-0.02),     # ok per GLB e TRK limitare a 0.02 !!!  STA invece -0.1 - 0.1
    etaRes_rangeMax = cms.double(0.02),      # TRK ha 200 bins da -0.1 a 0.1 
    #
    #     TRK ha 300 bins  da -0.02 a +0.02     // MUO aveva 120 bins tra -0.01 e 0.01
    cotThetaRes_nbin = cms.int32(100),           # per STA allargo a -0.05 - 0.05  e 80- bins -> allargo a -0.1 +0.1 e 100 bins
    cotThetaRes_rangeMin = cms.double(-0.01),
    cotThetaRes_rangeMax = cms.double(0.01),
    #
    # TRK ha 500 bins tra -0.1 e +0.1     // MUO ora ha 100 bins da -0.02 a 0.02
    dxyRes_nbin = cms.int32(100),         
    dxyRes_rangeMin = cms.double(-0.02),   # per STA provo 100 bins tra -4. e 4.  (400 um) -> non copre -> allargo tra -10 e 10.
    dxyRes_rangeMax = cms.double(0.02),    
    # TRK ha 150 bins tra -0.05  e +0.05  // MUO aveva lo stesso ora
    dzRes_nbin = cms.int32(100),           # riduco a 100 bins                       
    dzRes_rangeMin = cms.double(-0.05),    # per STA faccio 100 bins tra -4. e 4. -> non copre -> allargo tra -25 e 25.
    dzRes_rangeMax = cms.double(0.05)
)

#####################################################################################
# TRK tracks
trkMuonHistoParameters =  defaultMuonHistoParameters.clone()
trkMuonHistoParameters.usetracker = True
trkMuonHistoParameters.usemuon = False
trkMuonHistoParameters.nintNHit = 41
trkMuonHistoParameters.maxNHit = 40.5
trkMuonHistoParameters.do_TRKhitsPlots = True
trkMuonHistoParameters.do_MUOhitsPlots = False
#####################################################################################
# STA tracks
staMuonHistoParameters = defaultMuonHistoParameters.clone()
staMuonHistoParameters.usetracker = False
staMuonHistoParameters.usemuon = True
staMuonHistoParameters.nintNHit = 61
staMuonHistoParameters.maxNHit = 60.5
staMuonHistoParameters.do_TRKhitsPlots = False
staMuonHistoParameters.do_MUOhitsPlots = True
##
staMuonHistoParameters.nintDxy = 40
staMuonHistoParameters.minDxy = -10.
staMuonHistoParameters.maxDxy = 10.
##
staMuonHistoParameters.ptRes_nbin = 200
staMuonHistoParameters.ptRes_rangeMin = -1.
staMuonHistoParameters.ptRes_rangeMax = 5.
##
staMuonHistoParameters.phiRes_nbin = 200
staMuonHistoParameters.phiRes_rangeMin = -0.1
staMuonHistoParameters.phiRes_rangeMax = 0.1
##
staMuonHistoParameters.etaRes_nbin = 100
staMuonHistoParameters.etaRes_rangeMin = -0.1
staMuonHistoParameters.etaRes_rangeMax = 0.1
##
staMuonHistoParameters.cotThetaRes_nbin = 100
staMuonHistoParameters.cotThetaRes_rangeMin = -0.1
staMuonHistoParameters.cotThetaRes_rangeMax = 0.1
##
staMuonHistoParameters.dxyRes_nbin = 100
staMuonHistoParameters.dxyRes_rangeMin = -10.
staMuonHistoParameters.dxyRes_rangeMax = 10.
##
staMuonHistoParameters.dzRes_nbin = 100
staMuonHistoParameters.dzRes_rangeMin = -25.
staMuonHistoParameters.dzRes_rangeMax = 25.
#####################################################################################
# STA seeds (here hits are counting DT,CSC segments rather than individual hit layers)
staSeedMuonHistoParameters = staMuonHistoParameters.clone()
staSeedMuonHistoParameters.nintNHit = 7
staSeedMuonHistoParameters.maxNHit = 6.5
staSeedMuonHistoParameters.nintDTHit = 7
staSeedMuonHistoParameters.maxDTHit = 6.5
staSeedMuonHistoParameters.nintCSCHit = 7
staSeedMuonHistoParameters.maxCSCHit = 6.5
staSeedMuonHistoParameters.nintRPCHit = 7
staSeedMuonHistoParameters.maxRPCHit = 6.5
#####################################################################################
# STA Upd tracks
staUpdMuonHistoParameters = staMuonHistoParameters.clone()
staUpdMuonHistoParameters.dxyRes_nbin = 100
staUpdMuonHistoParameters.dxyRes_rangeMin = -1.
staUpdMuonHistoParameters.dxyRes_rangeMax = 1.
#####################################################################################
# GLB tracks
glbMuonHistoParameters =  defaultMuonHistoParameters.clone()
glbMuonHistoParameters.usetracker = True
glbMuonHistoParameters.usemuon = True
glbMuonHistoParameters.nintNHit = 81
glbMuonHistoParameters.maxNHit = 80.5
glbMuonHistoParameters.do_TRKhitsPlots = True
glbMuonHistoParameters.do_MUOhitsPlots = True
#####################################################################################
# Displaced TRK tracks
displacedTrkMuonHistoParameters = trkMuonHistoParameters.clone()
displacedTrkMuonHistoParameters.nintDxy = 85
displacedTrkMuonHistoParameters.minDxy = -85.
displacedTrkMuonHistoParameters.maxDxy = 85.
#
displacedTrkMuonHistoParameters.nintDz = 84
displacedTrkMuonHistoParameters.minDz = -210.
displacedTrkMuonHistoParameters.maxDz = 210.
#
displacedTrkMuonHistoParameters.nintRpos = 85
displacedTrkMuonHistoParameters.minRpos = 0.
displacedTrkMuonHistoParameters.maxRpos = 85.
#
displacedTrkMuonHistoParameters.nintZpos = 84
displacedTrkMuonHistoParameters.minZpos = -210.
displacedTrkMuonHistoParameters.maxZpos = 210.
#####################################################################################
# Displaced muons: STA tracks
displacedStaMuonHistoParameters = staMuonHistoParameters.clone()
displacedStaMuonHistoParameters.nintDxy = 85
displacedStaMuonHistoParameters.minDxy = -85.
displacedStaMuonHistoParameters.maxDxy = 85.
#
displacedStaMuonHistoParameters.nintDz = 84
displacedStaMuonHistoParameters.minDz = -210.
displacedStaMuonHistoParameters.maxDz = 210.
#
displacedStaMuonHistoParameters.nintRpos = 85
displacedStaMuonHistoParameters.minRpos = 0.
displacedStaMuonHistoParameters.maxRpos = 85.
#
displacedStaMuonHistoParameters.nintZpos = 84
displacedStaMuonHistoParameters.minZpos = -210.
displacedStaMuonHistoParameters.maxZpos = 210.
#####################################################################################
# Displaced muons: STA seeds (here hits are counting DT,CSC segments rather than individual hit layers)
displacedStaSeedMuonHistoParameters = displacedStaMuonHistoParameters.clone()
displacedStaSeedMuonHistoParameters.nintNHit = 7
displacedStaSeedMuonHistoParameters.maxNHit = 6.5
displacedStaSeedMuonHistoParameters.nintDTHit = 7
displacedStaSeedMuonHistoParameters.maxDTHit = 6.5
displacedStaSeedMuonHistoParameters.nintCSCHit = 7
displacedStaSeedMuonHistoParameters.maxCSCHit = 6.5
displacedStaSeedMuonHistoParameters.nintRPCHit = 7
displacedStaSeedMuonHistoParameters.maxRPCHit = 6.5
#####################################################################################
# Displaced muons: GLB tracks
displacedGlbMuonHistoParameters = glbMuonHistoParameters.clone()
displacedGlbMuonHistoParameters.nintDxy = 85
displacedGlbMuonHistoParameters.minDxy = -85.
displacedGlbMuonHistoParameters.maxDxy = 85.
#
displacedGlbMuonHistoParameters.nintDz = 84
displacedGlbMuonHistoParameters.minDz = -210.
displacedGlbMuonHistoParameters.maxDz = 210.
#
displacedGlbMuonHistoParameters.nintRpos = 85
displacedGlbMuonHistoParameters.minRpos = 0.
displacedGlbMuonHistoParameters.maxRpos = 85.
#
displacedGlbMuonHistoParameters.nintZpos = 84
displacedGlbMuonHistoParameters.minZpos = -210.
displacedGlbMuonHistoParameters.maxZpos = 210.
#####################################################################################
# COSMIC muons
#####################################################################################
# cosmics: TRK tracks (2-legs)
trkCosmicMuonHistoParameters = trkMuonHistoParameters.clone()
trkCosmicMuonHistoParameters.nintDxy = 40
trkCosmicMuonHistoParameters.minDxy = -10. 
trkCosmicMuonHistoParameters.maxDxy = 10.
#
trkCosmicMuonHistoParameters.nintDz = 50
trkCosmicMuonHistoParameters.minDz = -50.
trkCosmicMuonHistoParameters.maxDz = 50.
#
trkCosmicMuonHistoParameters.nintRpos = 40 
trkCosmicMuonHistoParameters.minRpos = 0.
trkCosmicMuonHistoParameters.maxRpos = 10.
#
trkCosmicMuonHistoParameters.nintZpos = 50
trkCosmicMuonHistoParameters.minZpos = -50.
trkCosmicMuonHistoParameters.maxZpos = 50.
#####################################################################################
# cosmics: STA tracks (2-legs)
staCosmicMuonHistoParameters = staMuonHistoParameters.clone()
staCosmicMuonHistoParameters.nintDxy = 40
staCosmicMuonHistoParameters.minDxy = -10. 
staCosmicMuonHistoParameters.maxDxy = 10.
#
staCosmicMuonHistoParameters.nintDz = 50
staCosmicMuonHistoParameters.minDz = -50.
staCosmicMuonHistoParameters.maxDz = 50.
#
staCosmicMuonHistoParameters.nintRpos = 40 
staCosmicMuonHistoParameters.minRpos = 0.
staCosmicMuonHistoParameters.maxRpos = 10.
#
staCosmicMuonHistoParameters.nintZpos = 50
staCosmicMuonHistoParameters.minZpos = -50.
staCosmicMuonHistoParameters.maxZpos = 50.
#####################################################################################
# cosmics: GLB tracks (2-legs)
glbCosmicMuonHistoParameters = glbMuonHistoParameters.clone()
glbCosmicMuonHistoParameters.nintDxy = 40
glbCosmicMuonHistoParameters.minDxy = -10. 
glbCosmicMuonHistoParameters.maxDxy = 10.
#
glbCosmicMuonHistoParameters.nintDz = 50
glbCosmicMuonHistoParameters.minDz = -50.
glbCosmicMuonHistoParameters.maxDz = 50.
#
glbCosmicMuonHistoParameters.nintRpos = 40 
glbCosmicMuonHistoParameters.minRpos = 0.
glbCosmicMuonHistoParameters.maxRpos = 10.
#
glbCosmicMuonHistoParameters.nintZpos = 50
glbCosmicMuonHistoParameters.minZpos = -50.
glbCosmicMuonHistoParameters.maxZpos = 50.
#####################################################################################
# cosmics: TRK tracks (1-leg)
trkCosmic1LegMuonHistoParameters = trkCosmicMuonHistoParameters.clone()
trkCosmic1LegMuonHistoParameters.nintNHit = 81
trkCosmic1LegMuonHistoParameters.maxNHit = 80.5
#
trkCosmic1LegMuonHistoParameters.nintLayers = 31
trkCosmic1LegMuonHistoParameters.maxLayers = 30.5
#
trkCosmic1LegMuonHistoParameters.nintPixels = 11
trkCosmic1LegMuonHistoParameters.maxPixels = 10.5
#####################################################################################
# cosmics: STA tracks (1-leg)
staCosmic1LegMuonHistoParameters = staCosmicMuonHistoParameters.clone()
staCosmic1LegMuonHistoParameters.nintNHit = 121
staCosmic1LegMuonHistoParameters.maxNHit = 120.5
#
staCosmic1LegMuonHistoParameters.nintDTHit = 101
staCosmic1LegMuonHistoParameters.maxDTHit = 100.5
#
staCosmic1LegMuonHistoParameters.nintCSCHit = 101
staCosmic1LegMuonHistoParameters.maxCSCHit = 100.5
#
staCosmic1LegMuonHistoParameters.nintRPCHit = 21
staCosmic1LegMuonHistoParameters.maxRPCHit = 20.5
#####################################################################################
# cosmics: GLB tracks (1-leg)
glbCosmic1LegMuonHistoParameters = glbCosmicMuonHistoParameters.clone()
glbCosmic1LegMuonHistoParameters.nintNHit = 161
glbCosmic1LegMuonHistoParameters.maxNHit = 160.5
#
glbCosmic1LegMuonHistoParameters.nintDTHit = 101
glbCosmic1LegMuonHistoParameters.maxDTHit = 100.5
#
glbCosmic1LegMuonHistoParameters.nintCSCHit = 101
glbCosmic1LegMuonHistoParameters.maxCSCHit = 100.5
#
glbCosmic1LegMuonHistoParameters.nintRPCHit = 21
glbCosmic1LegMuonHistoParameters.maxRPCHit = 20.5
#
glbCosmic1LegMuonHistoParameters.nintLayers = 31 
glbCosmic1LegMuonHistoParameters.maxLayers = 30.5
#
glbCosmic1LegMuonHistoParameters.nintPixels = 11
glbCosmic1LegMuonHistoParameters.maxPixels = 10.5
