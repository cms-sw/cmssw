import FWCore.ParameterSet.Config as cms
#GEN-SIM so far...
def customise(process):
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    n=0
    if hasattr(process,'reconstruction') or hasattr(process,'dqmoffline_step'):
        if hasattr(process,'mix'):
            if hasattr(process.mix,'input'):
                n=process.mix.input.nbPileupEvents.averageNumber.value()
        else:
            print 'phase1TkCustoms requires a --pileup option to cmsDriver to run the reconstruction/dqm'
            print 'Please provide one!'
            sys.exit(1)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process,float(n))
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process,n)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process,float(n))
    if hasattr(process,'L1TrackTrigger_step'):
        process=customise_TrackTrigger(process)
    process=customise_condOverRides(process)
    
    return process

def customise_Digi(process):
    process.mix.digitizers.pixel.MissCalibrate = False
    process.mix.digitizers.pixel.LorentzAngle_DB = False
    process.mix.digitizers.pixel.killModules = False
    process.mix.digitizers.pixel.useDB = False
    process.mix.digitizers.pixel.DeadModules_DB = False
    process.mix.digitizers.pixel.NumPixelBarrel = cms.int32(10)
    process.mix.digitizers.pixel.NumPixelEndcap = cms.int32(10)
    process.mix.digitizers.pixel.ThresholdInElectrons_FPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix_L1 = cms.double(2000.0)
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelColEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.AddPixelInefficiencyFromPython = cms.bool(False)
    process.mix.digitizers.strip.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof",
                         'g4SimHitsTrackerHitsPixelEndcapLowTof')
    process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"))
    process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"))
    process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"))
    process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"))
    process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECLowTof"))
    process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECHighTof"))
    process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"))
    process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"))
    
    return process


def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.siPixelRawData)
    process.digi2raw_step.remove(process.rpcpacker)
    return process

def customise_RawToDigi(process):
    process.raw2digi_step.remove(process.siPixelDigis)
    return process

def customise_Reco(process,pileup):



    #use with latest pixel geometry
    process.ClusterShapeHitFilterESProducer.PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par')
    # Need this line to stop error about missing siPixelDigis.
    process.MeasurementTracker.inactivePixelDetectorLabels = cms.VInputTag()

    # new layer list (3/4 pixel seeding) in InitialStep and pixelTracks
    process.pixellayertriplets.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
                                                        'BPix2+BPix3+BPix4',
                                                        'BPix1+BPix3+BPix4',
                                                        'BPix1+BPix2+BPix4',
                                                        'BPix2+BPix3+FPix1_pos',
                                                        'BPix2+BPix3+FPix1_neg',
                                                        'BPix1+BPix2+FPix1_pos',
                                                        'BPix1+BPix2+FPix1_neg',
                                                        'BPix2+FPix1_pos+FPix2_pos',
                                                        'BPix2+FPix1_neg+FPix2_neg',
                                                        'BPix1+FPix1_pos+FPix2_pos',
                                                        'BPix1+FPix1_neg+FPix2_neg',
                                                        'FPix1_pos+FPix2_pos+FPix3_pos',
                                                        'FPix1_neg+FPix2_neg+FPix3_neg' )

    # New tracking.  This is really ugly because it redefines globalreco and reconstruction.
    # It can be removed if change one line in Configuration/StandardSequences/python/Reconstruction_cff.py
    # from RecoTracker_cff.py to RecoTrackerPhase1PU140_cff.py

    # remove all the tracking first
    itIndex=process.globalreco.index(process.trackingGlobalReco)
    grIndex=process.reconstruction.index(process.globalreco)

    process.reconstruction.remove(process.globalreco)
    process.globalreco.remove(process.iterTracking)
    process.globalreco.remove(process.electronSeedsSeq)
    process.reconstruction_fromRECO.remove(process.trackingGlobalReco)
    del process.iterTracking
    del process.ckftracks
    del process.ckftracks_woBH
    del process.ckftracks_wodEdX
    del process.ckftracks_plus_pixelless
    del process.trackingGlobalReco
    del process.electronSeedsSeq
    del process.InitialStep
    del process.LowPtTripletStep
    del process.PixelPairStep
    del process.DetachedTripletStep
    del process.MixedTripletStep
    del process.PixelLessStep
    del process.TobTecStep
    del process.earlyGeneralTracks
    del process.ConvStep
    del process.earlyMuons
    del process.muonSeededStepCore
    del process.muonSeededStepExtra 
    del process.muonSeededStep
    del process.muonSeededStepDebug
    
    # add the correct tracking back in
    process.load("RecoTracker.Configuration.RecoTrackerPhase2BE_cff")

    process.globalreco.insert(itIndex,process.trackingGlobalReco)
    process.reconstruction.insert(grIndex,process.globalreco)
    #Note process.reconstruction_fromRECO is broken
    
    # End of new tracking configuration which can be removed if new Reconstruction is used.


    process.reconstruction.remove(process.castorreco)
    process.reconstruction.remove(process.CastorTowerReco)
    process.reconstruction.remove(process.ak7BasicJets)
    process.reconstruction.remove(process.ak7CastorJetID)

    #the quadruplet merger configuration     
    process.load("RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff")
    process.pixelseedmergerlayers.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.BPix.HitProducer = cms.string("siPixelRecHits" )
    process.pixelseedmergerlayers.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.FPix.HitProducer = cms.string("siPixelRecHits" )    
    
    # Need these until pixel templates are used
    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    # PixelCPEGeneric #
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
    process.PixelCPEGenericESProducer.DoCosmics = False
    # CPE for other steps
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
    # Turn of template use in tracking (iterative steps handled inside their configs)
    process.mergedDuplicateTracks.TTRHBuilder = 'WithTrackAngle'
    process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
    process.muonSeededSeedsInOut.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.muonSeededTracksInOut.TTRHBuilder=cms.string('WithTrackAngle')
    process.muons1stStep.TrackerKinkFinderParameters.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.regionalCosmicTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.cosmicsVetoTracksRaw.TTRHBuilder=cms.string('WithTrackAngle')
    # End of pixel template needed section
    
    process.regionalCosmicTrackerSeeds.OrderedHitsFactoryPSet.LayerPSet.layerList  = cms.vstring('BPix10+BPix9')  # Optimize later
    process.regionalCosmicTrackerSeeds.OrderedHitsFactoryPSet.LayerPSet.BPix = cms.PSet(
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.006),
        useErrorsFromParam = cms.bool(True),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        skipClusters = cms.InputTag("pixelPairStepClusters"),
        hitErrorRPhi = cms.double(0.0027)
    )
    # Make pixelTracks use quadruplets
    process.pixelTracks.SeedMergerPSet = cms.PSet(
        layerListName = cms.string('PixelSeedMergerQuadruplets'),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
        )
    process.pixelTracks.FilterPSet.chi2 = cms.double(50.0)
    process.pixelTracks.FilterPSet.tipMax = cms.double(0.05)
    process.pixelTracks.RegionFactoryPSet.RegionPSet.originRadius =  cms.double(0.02)

    return process

def customise_condOverRides(process):
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_BarrelEndcap_cff')
    process.trackerNumberingSLHCGeometry.layerNumberPXB = cms.uint32(20)
    process.trackerTopologyConstants.pxb_layerStartBit = cms.uint32(20)
    process.trackerTopologyConstants.pxb_ladderStartBit = cms.uint32(12)
    process.trackerTopologyConstants.pxb_moduleStartBit = cms.uint32(2)
    process.trackerTopologyConstants.pxb_layerMask = cms.uint32(15)
    process.trackerTopologyConstants.pxb_ladderMask = cms.uint32(255)
    process.trackerTopologyConstants.pxb_moduleMask = cms.uint32(1023)
    process.trackerTopologyConstants.pxf_diskStartBit = cms.uint32(18)
    process.trackerTopologyConstants.pxf_bladeStartBit = cms.uint32(12)
    process.trackerTopologyConstants.pxf_panelStartBit = cms.uint32(10)
    process.trackerTopologyConstants.pxf_moduleMask = cms.uint32(255)
    return process


def l1EventContent(process):
    #extend the event content

    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):

            getattr(process,b).outputCommands.append('keep *_TTClustersFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTStubsFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTTracksFromPixelDigis_*_*')

            getattr(process,b).outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')

            getattr(process,b).outputCommands.append('drop PixelDigiSimLinkedmDetSetVector_mix_*_*')
            getattr(process,b).outputCommands.append('drop PixelDigiedmDetSetVector_mix_*_*')

            getattr(process,b).outputCommands.append('keep *_simSiPixelDigis_*_*')

    return process

def customise_DQM(process,pileup):
    # We cut down the number of iterative tracking steps
#    process.dqmoffline_step.remove(process.TrackMonStep3)
#    process.dqmoffline_step.remove(process.TrackMonStep4)
#    process.dqmoffline_step.remove(process.TrackMonStep5)
#    process.dqmoffline_step.remove(process.TrackMonStep6)
    			    #The following two steps were removed
                            #process.PixelLessStep*
                            #process.TobTecStep*
#    process.dqmoffline_step.remove(process.muonAnalyzer)
    process.dqmoffline_step.remove(process.jetMETAnalyzer)
#    process.dqmoffline_step.remove(process.TrackMonStep9)
#    process.dqmoffline_step.remove(process.TrackMonStep10)
#    process.dqmoffline_step.remove(process.PixelTrackingRecHitsValid)

    #put isUpgrade flag==true
    process.SiPixelRawDataErrorSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelDigiSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelClusterSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelRecHitSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelTrackResidualSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelHitEfficiencySource.isUpgrade = cms.untracked.bool(True)
    
    from DQM.TrackingMonitor.customizeTrackingMonitorSeedNumber import customise_trackMon_IterativeTracking_PHASE1PU140
    process=customise_trackMon_IterativeTracking_PHASE1PU140(process)
    process.dqmoffline_step.remove(process.Phase1Pu70TrackMonStep2)
    process.dqmoffline_step.remove(process.Phase1Pu70TrackMonStep4)
    process.globalrechitsanalyze.ROUList = cms.vstring(
       'g4SimHitsTrackerHitsPixelBarrelLowTof', 
       'g4SimHitsTrackerHitsPixelBarrelHighTof', 
       'g4SimHitsTrackerHitsPixelEndcapLowTof', 
       'g4SimHitsTrackerHitsPixelEndcapHighTof')
    return process

def customise_Validation(process,pileup):
    process.validation_step.remove(process.PixelTrackingRecHitsValid)
    process.validation_step.remove(process.stripRecHitsValid)
    process.validation_step.remove(process.trackerHitsValid)
    process.validation_step.remove(process.StripTrackingRecHitsValid)
    # We don't run the HLT
    process.validation_step.remove(process.HLTSusyExoVal)
    process.validation_step.remove(process.hltHiggsValidator)
    process.validation_step.remove(process.relvalMuonBits)
    if pileup>30:
        process.trackValidator.label=cms.VInputTag(cms.InputTag("cutsRecoTracksHp"))
        process.tracksValidationSelectors = cms.Sequence(process.cutsRecoTracksHp)
        process.globalValidation.remove(process.recoMuonValidation)
        process.validation.remove(process.recoMuonValidation)
        process.validation_preprod.remove(process.recoMuonValidation)
        process.validation_step.remove(process.recoMuonValidation)
        process.validation.remove(process.globalrechitsanalyze)
        process.validation_prod.remove(process.globalrechitsanalyze)
        process.validation_step.remove(process.globalrechitsanalyze)
        process.validation.remove(process.stripRecHitsValid)
        process.validation_step.remove(process.stripRecHitsValid)
        process.validation_step.remove(process.StripTrackingRecHitsValid)
        process.globalValidation.remove(process.vertexValidation)
        process.validation.remove(process.vertexValidation)
        process.validation_step.remove(process.vertexValidation)
        process.mix.input.nbPileupEvents.averageNumber = cms.double(0.0)
        process.mix.minBunch = cms.int32(0)
        process.mix.maxBunch = cms.int32(0)

    if hasattr(process,'simHitTPAssocProducer'):    
        process.simHitTPAssocProducer.simHitSrc=cms.VInputTag(cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
                                                              cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"))
    return process

def customise_harvesting(process):
    process.dqmHarvesting.remove(process.jetMETDQMOfflineClient)
    process.dqmHarvesting.remove(process.dataCertificationJetMET)
    process.dqmHarvesting.remove(process.sipixelEDAClient)
    process.dqmHarvesting.remove(process.sipixelCertification)
    return (process)

def customise_TrackTrigger(process):
    process.StackedTrackerGeometryESModule.EndcapCutSet = cms.VPSet(
        cms.PSet( EndcapCut = cms.vdouble( 0 ) ), #Use 0 as dummy to have direct access using DetId to the correct element
        cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 4, 4, 5, 6, 6, 7, 5, 6, 6, 9, 11, 11 ) ), #D1
        cms.PSet( EndcapCut = cms.vdouble( 0, 2, 3, 4, 4, 5, 5, 6, 7, 4, 6, 6, 8, 8, 10 ) ), #D2 ...
        cms.PSet( EndcapCut = cms.vdouble( 0, 2, 2, 3, 4, 4, 5, 6, 6, 4, 4, 6, 6, 8, 9 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 2, 2, 3, 3, 4, 4, 5, 5, 7, 4, 5, 6, 6, 8 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 2, 3, 3, 4, 4, 5, 5, 6, 8, 4, 5, 6, 7 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 3, 4, 4, 5, 6, 7, 4, 5, 6, 6 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 2, 3, 4, 4, 5, 5, 6, 8, 4, 5, 6 ) ) 
        # missing rings are not taken into account in numbering, so everything
        # always starts from 1 to N, with increasing r
    )

    process.TTStubAlgorithm_tab2013_PixelDigi_.EndcapCutSet = cms.VPSet(
        cms.PSet( EndcapCut = cms.vdouble( 0 ) ), #Use 0 as dummy to have direct access using DetId to the correct element
        cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 2.5, 3.0, 3.0, 4.5, 5.5, 5.5 ) ), #D1
        cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.5, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0 ) ), #D2 ...
        cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.5 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.5, 2.0, 2.5, 3.0, 3.0, 4.0 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 4.0, 2.0, 2.5, 3.0, 3.5 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.5, 3.0, 3.5, 2.0, 2.5, 3.0, 3.0 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 4.0, 2.0, 2.5, 3.0 ) ) # missing rings are not taken into account in numbering, so everything
        # always starts from 1 to N, with increasing r
        )
      
                                                                                                                                        
    
#    process.StackedTrackerGeometryESModule.EndcapCutSet.append(process.StackedTrackerGeometryESModule.EndcapCutSet[5])
#    process.StackedTrackerGeometryESModule.EndcapCutSet.append(process.StackedTrackerGeometryESModule.EndcapCutSet[5])
#    process.TTStubAlgorithm_tab2013_PixelDigi_.EndcapCutSet.append(process.TTStubAlgorithm_tab2013_PixelDigi_.EndcapCutSet[5])
#    process.TTStubAlgorithm_tab2013_PixelDigi_.EndcapCutSet.append(process.TTStubAlgorithm_tab2013_PixelDigi_.EndcapCutSet[5])
    
    return process
