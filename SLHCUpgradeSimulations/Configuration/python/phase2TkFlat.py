import FWCore.ParameterSet.Config as cms
#import SLHCUpgradeSimulations.Configuration.customise_PFlow as customise_PFlow

#GEN-SIM so far...
def customise(process):
    print "!!!You are using the SUPPORTED FLAT version of the Phase2 Tracker !!!"
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
    process=customise_condOverRides(process)

    return process

def customise_Digi(process):
    process.digitisation_step.remove(process.mix.digitizers.pixel)
    process.load('SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi')
    process.mix.digitizers.pixel=process.phase2TrackerDigitizer
    process.mix.digitizers.strip.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof",
                         'g4SimHitsTrackerHitsPixelEndcapLowTof')
    #Check if mergedtruth is in the sequence first, could be taken out depending on cmsDriver options
    if hasattr(process.mix.digitizers,"mergedtruth") :
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"))

    # keep new digis
    alist=['FEVTDEBUG','FEVTDEBUGHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep Phase2TrackerDigiedmDetSetVector_*_*_*')
    return process


def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.siPixelRawData)
    process.digi2raw_step.remove(process.rpcpacker)
    return process

def customise_RawToDigi(process):
    process.raw2digi_step.remove(process.siPixelDigis)
    return process

def customise_Reco(process,pileup):
    # insert the new clusterizer
    process.load('SimTracker.SiPhase2Digitizer.phase2TrackerClusterizer_cfi')
    
    #process.load('RecoLocalTracker.SubCollectionProducers.jetCoreClusterSplitter_cfi')	
    #clustersTmp = 'siPixelClustersPreSplitting'
     # 0. Produce tmp clusters in the first place.
    #process.siPixelClustersPreSplitting = process.siPixelClusters.clone()
    #process.siPixelRecHitsPreSplitting = process.siPixelRecHits.clone()
    #process.siPixelRecHitsPreSplitting.src = clustersTmp
    #process.pixeltrackerlocalreco.replace(process.siPixelClusters, process.siPixelClustersPreSplitting)
    #process.pixeltrackerlocalreco.replace(process.siPixelRecHits, process.siPixelRecHitsPreSplitting)
    #process.clusterSummaryProducer.pixelClusters = clustersTmp
    itIndex = process.pixeltrackerlocalreco.index(process.siPixelClustersPreSplitting)
    process.pixeltrackerlocalreco.insert(itIndex, process.siPhase2Clusters)
    process.pixeltrackerlocalreco.remove(process.siPixelClustersPreSplitting)
    process.pixeltrackerlocalreco.remove(process.siPixelRecHitsPreSplitting)
    process.trackerlocalreco.remove(process.clusterSummaryProducer)
    # keep new clusters
    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_siPhase2Clusters_*_*')


 
    return process

def customise_condOverRides(process):
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkFlat_cff')
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
    # SiPixelRawDataErrorSource doesn't work with Stacks, so take it out
    process.dqmoffline_step.remove(process.SiPixelRawDataErrorSource)

    ## DQM for stacks doesn't work yet, so skip adding the outer tracker.
    ##add Phase 2 Upgrade Outer Tracker
    #stripIndex=process.DQMOfflinePreDPG.index(process.SiStripDQMTier0)
    #process.load("DQM.Phase2OuterTracker.OuterTrackerSourceConfig_cff")
    #process.dqmoffline_step.insert(stripIndex, process.OuterTrackerSource)

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
    if hasattr(process,"globalrechitsanalyze") : # Validation takes this out if pileup is more than 30
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

    ## This next part doesn't work for stacks yet, so skip adding it.
    ## Include Phase 2 Upgrade Outer Tracker
    #stripVIndex=process.globalValidation.index(process.trackerDigisValidation)
    #process.load("Validation.Phase2OuterTracker.OuterTrackerSourceConfig_cff")
    #process.validation_step.insert(stripVIndex, process.OuterTrackerSource)

    process.pixelDigisValid.src = cms.InputTag('simSiPixelDigis', "Pixel")
    process.tpClusterProducer.pixelSimLinkSrc = cms.InputTag("simSiPixelDigis","Pixel")
    
    # We don't run the HLT
    process.validation_step.remove(process.HLTSusyExoVal)
    process.validation_step.remove(process.hltHiggsValidator)
    process.validation_step.remove(process.relvalMuonBits)
    # TrackerHitAssociator needs updating for stacks, so all of the following
    # need to be taken out. They either require hit association or rely on a
    # module that does.
    process.validation_step.remove(process.globalrechitsanalyze)
    process.validation_step.remove(process.pixRecHitsValid)
    process.validation_step.remove(process.recoMuonValidation)
    
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

    # Include Phase 2 Upgrade Outer Tracker
    strip2Index=process.DQMOffline_SecondStep_PreDPG.index(process.SiStripOfflineDQMClient)
    process.load("DQM.Phase2OuterTracker.OuterTrackerClientConfig_cff")
    process.dqmHarvesting.insert(strip2Index, process.OuterTrackerClient)

    strip2VIndex=process.postValidation.index(process.bTagCollectorSequenceMCbcl)
    process.load("Validation.Phase2OuterTracker.OuterTrackerClientConfig_cff")
    process.validationHarvesting.insert(strip2VIndex, process.OuterTrackerClient)
    return (process)

