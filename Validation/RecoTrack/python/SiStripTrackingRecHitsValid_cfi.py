import FWCore.ParameterSet.Config as cms

StripTrackingRecHitsValid = cms.EDAnalyzer("SiStripTrackingRecHitsValid",
    outputFile = cms.string('striptrackingrechitshisto.root'),
    runStandalone = cms.bool(False),
    OutputMEsInRootFile = cms.bool(False),
    TopFolderName = cms.string('SiStrip/RecHitsValidation/TrackingRecHits'),

    TH1Resolx_LF = cms.PSet(
        Nbinx          = cms.int32(1000),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.05),
        layerswitchon  = cms.bool(False)
    ),

    TH1Resolx_MF = cms.PSet(
        Nbinx          = cms.int32(1000),
        xmin           = cms.double(0.),
        xmax           = cms.double(5),
        layerswitchon  = cms.bool(True)
    ),

    TH1Res_LF = cms.PSet(
        Nbinx          = cms.int32(1000),
        xmin           = cms.double(-0.2),
        xmax           = cms.double(+0.2),
        layerswitchon  = cms.bool(False)
    ),

    TH1Res_MF = cms.PSet(
        Nbinx          = cms.int32(1000),
        xmin           = cms.double(-20.0),
        xmax           = cms.double(+20.0),
        layerswitchon  = cms.bool(True)
    ),

    TH1Pull_LF = cms.PSet(
        Nbinx          = cms.int32(1000),
        xmin           = cms.double(-50.0),
        xmax           = cms.double(+50.0),
        layerswitchon  = cms.bool(False)
    ),

    TH1Pull_MF = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-50.0),
        xmax           = cms.double(+50.0),
        layerswitchon  = cms.bool(True)
    ),

    TH1Category = cms.PSet(
        Nbinx          = cms.int32(10),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.0),
        layerswitchon  = cms.bool(False)
    ),

    TH1Trackwidth = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.0),
        layerswitchon  = cms.bool(True)
    ),

    TH1Expectedwidth = cms.PSet(
        Nbinx          = cms.int32(10),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.0),
        layerswitchon  = cms.bool(True)
    ),

    TH1Clusterwidth= cms.PSet(
        Nbinx          = cms.int32(15),
        xmin           = cms.double(0.),
        xmax           = cms.double(15.0),
        layerswitchon  = cms.bool(True)
    ),

    TH1Trackanglealpha = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-100.0),
        xmax           = cms.double(+100.0),
        layerswitchon  = cms.bool(True)
    ),

    TH1Trackanglebeta = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-100.0),
        xmax           = cms.double(+100.0),
        layerswitchon  = cms.bool(True)
    ),

    TProfResolxMFTrackwidthProfile_WClus1 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

    TProfResolxMFTrackwidthProfile_WClus2 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

    TProfResolxMFTrackwidthProfile_WClus3 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

    TProfResolxMFTrackwidthProfile_WClus4 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

    TProfResMFTrackwidthProfile_WClus1 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResMFTrackwidthProfile_WClus2 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResMFTrackwidthProfile_WClus21 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResMFTrackwidthProfile_WClus22 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-5.0),
        ymax           = cms.double(+5.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResMFTrackwidthProfile_WClus23 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-0.5),
        ymax           = cms.double(+0.5),
        layerswitchon  = cms.bool(False)
    ),

     TProfResMFTrackwidthProfile_WClus3 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResMFTrackwidthProfile_WClus4 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResolxMFTrackwidthProfile = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

    TProfResolxMFTrackwidthProfile_Category1 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResolxMFTrackwidthProfile_Category2 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResolxMFTrackwidthProfile_Category3 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResolxMFTrackwidthProfile_Category4 = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResolxMFClusterwidthProfile_Category1 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfResolxMFAngleProfile = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-1.),
        xmax           = cms.double(60.0),
        ymin           = cms.double(0.),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1WclusRphi = cms.PSet(
        Nbinx          = cms.int32(20),
        xmin           = cms.double(0.5),
        xmax           = cms.double(20.5),
        layerswitchon  = cms.bool(True)
    ),

   TH1AdcRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(300.),
        layerswitchon  = cms.bool(True)
    ),

   TH1ResolxLFRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.005),
        layerswitchon  = cms.bool(False)
    ),

   TH1ResolxMFRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

   TH1ResolxMFRphiwclus1 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

   TH1ResolxMFRphiwclus2 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

   TH1ResolxMFRphiwclus3 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

   TH1ResolxMFRphiwclus4 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(2.0),
        layerswitchon  = cms.bool(True)
    ),

   TH1ResLFRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-0.02),
        xmax           = cms.double(+0.02),
        layerswitchon  = cms.bool(False)
    ),

   TH1ResMFRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

   TH1ResMFRphiwclus1 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1ResMFRphiwclus2 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1ResMFRphiwclus3 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1ResMFRphiwclus4 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1PullLFRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-5.0),
        xmax           = cms.double(+5.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1PullMFRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-5.0),
        xmax           = cms.double(+5.0),
        layerswitchon  = cms.bool(True)
    ),

   TH1PullMFRphiwclus1 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1PullMFRphiwclus2 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1PullMFRphiwclus3 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1PullMFRphiwclus4 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.0),
        xmax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1TrackangleRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-20.0),
        xmax           = cms.double(+20.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1TrackanglebetaRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-20.0),
        xmax           = cms.double(+20.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1Trackangle2Rphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-20.0),
        xmax           = cms.double(+20.0),
        layerswitchon  = cms.bool(False)
    ),

    TProfPullTrackangleProfileRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-20.0),
        xmax           = cms.double(+20.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1PullTrackangle2DRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.0),
        layerswitchon  = cms.bool(False)
    ),

  TH1TrackwidthRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.0),
        layerswitchon  = cms.bool(False)
    ),

   TH1ExpectedwidthRphi = cms.PSet(
        Nbinx          = cms.int32(10),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        layerswitchon  = cms.bool(False)
    ),

    TH1ClusterwidthRphi = cms.PSet(
        Nbinx          = cms.int32(10),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        layerswitchon  = cms.bool(False)
    ),

   TH1CategoryRphi = cms.PSet(
        Nbinx          = cms.int32(10),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        layerswitchon  = cms.bool(False)
    ),

   TProfPullTrackwidthProfileRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

   TProfPullTrackwidthProfileRphiwclus1 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

   TProfPullTrackwidthProfileRphiwclus2 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

   TProfPullTrackwidthProfileRphiwclus3 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

   TProfPullTrackwidthProfileRphiwclus4 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

   TProfPullTrackwidthProfileCategory1Rphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfPullTrackwidthProfileCategory2Rphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfPullTrackwidthProfileCategory3Rphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfPullTrackwidthProfileCategory4Rphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileRphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(8.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

  TProfResolxMFTrackwidthProfileWclus1Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

  TProfResolxMFTrackwidthProfileWclus2Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

  TProfResolxMFTrackwidthProfileWclus3Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

  TProfResolxMFTrackwidthProfileWclus4Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(True)
    ),

  TProfResMFTrackwidthProfileWclus1Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResMFTrackwidthProfileWclus2Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResMFTrackwidthProfileWclus3Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResMFTrackwidthProfileWclus4Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileCategory1Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileCategory2Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileCategory3Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileCategory4Rphi = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFAngleProfileRphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.0),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFClusterwidthProfileCategory1Rphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        ymin           = cms.double(-2.0),
        ymax           = cms.double(+2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfrapidityResProfilewclus1 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.5),
        xmax           = cms.double(+2.5),
        ymin           = cms.double(0.0),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfrapidityResProfilewclus2 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.5),
        xmax           = cms.double(+2.5),
        ymin           = cms.double(0.0),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfrapidityResProfilewclus3 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.5),
        xmax           = cms.double(+2.5),
        ymin           = cms.double(0.0),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

  TProfrapidityResProfilewclus4 = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.5),
        xmax           = cms.double(+2.5),
        ymin           = cms.double(0.0),
        ymax           = cms.double(2.0),
        layerswitchon  = cms.bool(False)
    ),

  TH1WclusSas = cms.PSet(
        Nbinx          = cms.int32(20),
        xmin           = cms.double(0.5),
        xmax           = cms.double(20.5),
        layerswitchon  = cms.bool(True)
    ),

  TH1AdcSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(300.),
        layerswitchon  = cms.bool(True)
    ),

  TH1ResolxLFSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.005),
        layerswitchon  = cms.bool(False)
    ),

  TH1ResolxMFSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.5),
        layerswitchon  = cms.bool(True)
    ),

  TH1ResLFSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-0.02),
        xmax           = cms.double(+0.02),
        layerswitchon  = cms.bool(False)
    ),

  TH1ResMFSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-2.),
        xmax           = cms.double(+2.),
        layerswitchon  = cms.bool(True)
    ),

  TH1PullLFSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-4.),
        xmax           = cms.double(+4.),
        layerswitchon  = cms.bool(False)
    ),

  TH1PullMFSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-4.),
        xmax           = cms.double(+4.),
        layerswitchon  = cms.bool(True)
    ),

  TH1TrackangleSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-40.),
        xmax           = cms.double(+40.),
        layerswitchon  = cms.bool(False)
    ),

  TH1TrackanglebetaSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-40.),
        xmax           = cms.double(+40.),
        layerswitchon  = cms.bool(False)
    ),

  TProfPullTrackangleProfileSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-40.),
        xmax           = cms.double(+40.),
        ymin           = cms.double(-4.),
        ymax           = cms.double(+4.),
        layerswitchon  = cms.bool(False)
    ),

  TH1TrackwidthSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.),
        layerswitchon  = cms.bool(False)
    ),

  TH1ExpectedwidthSas = cms.PSet(
        Nbinx          = cms.int32(10),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        layerswitchon  = cms.bool(False)
    ),

  TH1ClusterwidthSas = cms.PSet(
        Nbinx          = cms.int32(10),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        layerswitchon  = cms.bool(False)
    ),

  TH1CategorySas = cms.PSet(
        Nbinx          = cms.int32(10),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        layerswitchon  = cms.bool(False)
    ),

  TProfPullTrackwidthProfileSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(True)
    ),

  TProfPullTrackwidthProfileCategory1Sas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfPullTrackwidthProfileCategory2Sas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfPullTrackwidthProfileCategory3Sas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfPullTrackwidthProfileCategory4Sas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileSas = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(True)
    ),

  TProfResolxMFTrackwidthProfileCategory1Sas = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileCategory2Sas = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileCategory3Sas = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFTrackwidthProfileCategory4Sas = cms.PSet(
        Nbinx          = cms.int32(12),
        xmin           = cms.double(0.),
        xmax           = cms.double(4.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFAngleProfileSas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(1.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TProfResolxMFClusterwidthProfileCategory1Sas = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(10.),
        ymin           = cms.double(-2.),
        ymax           = cms.double(+2.),
        layerswitchon  = cms.bool(False)
    ),

  TH1PosxMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-6.0),
        xmax           = cms.double(+6.0),
        layerswitchon  = cms.bool(False)
    ),

  TH1PosyMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-6.0),
        xmax           = cms.double(+6.0),
        layerswitchon  = cms.bool(False)
    ),

  TH1ResolxMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.05),
        layerswitchon  = cms.bool(False)
    ),

  TH1ResolyMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.05),
        layerswitchon  = cms.bool(False)
    ),

  TH1ResxMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-0.02),
        xmax           = cms.double(+0.02),
        layerswitchon  = cms.bool(False)
    ),

  TH1ResyMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-1.),
        xmax           = cms.double(+1.),
        layerswitchon  = cms.bool(False)
    ),

  TH1PullxMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-5.),
        xmax           = cms.double(+5.),
        layerswitchon  = cms.bool(False)
    ),

  TH1PullyMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-5.),
        xmax           = cms.double(+5.),
        layerswitchon  = cms.bool(False)
    ),

    trajectoryInput = cms.InputTag("generalTracks"),
    associatePixel = cms.bool(False),
    ROUList = cms.vstring('g4SimHitsTrackerHitsTIBLowTof', 
			'g4SimHitsTrackerHitsTIBHighTof', 
			'g4SimHitsTrackerHitsTIDLowTof', 
			'g4SimHitsTrackerHitsTIDHighTof', 
			'g4SimHitsTrackerHitsTOBLowTof', 
			'g4SimHitsTrackerHitsTOBHighTof', 
			'g4SimHitsTrackerHitsTECLowTof', 
			'g4SimHitsTrackerHitsTECHighTof'),
    associateRecoTracks = cms.bool(False),
    #	string trajectoryInput = "rsWithMaterialTracks"
    associateStrip = cms.bool(True)
)


