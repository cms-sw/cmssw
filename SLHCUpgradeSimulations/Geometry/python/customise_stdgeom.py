import FWCore.ParameterSet.Config as cms

#common stuff here
def customise(process):
    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    #this line is different
    process.load("SLHCUpgradeSimulations.Geometry.upgradeTracking_stdgeom_cff")

    process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
    #this line is different
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(True)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    #this line is different
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(True)
    #two lines removed here
    process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
    process.PixelCPEGenericESProducer.DoCosmics = False

    ## CPE for other steps
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
    process.initialStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.lowPtTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.pixelPairStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.detachedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.mixedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.pixelLessStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.tobTecStepTracks.TTRHBuilder = cms.string('WithTrackAngle')

    # Need these lines to stop some errors about missing siStripDigis collections.
    # should add them to fakeConditions_Phase1_cff
    process.MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
    process.MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
    process.MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
    process.MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
    process.MeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
    process.MeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)

    process.detachedTripletStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
    process.detachedTripletStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
    process.detachedTripletStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
    process.detachedTripletStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
    process.detachedTripletStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
    process.detachedTripletStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
    process.mixedTripletStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
    process.mixedTripletStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
    process.mixedTripletStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
    process.mixedTripletStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
    process.mixedTripletStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
    process.mixedTripletStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
    process.pixelLessStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
    process.tobTecStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()

    ### back to standard job commands ##################################################
    process.DigiToRaw.remove(process.castorRawData)

#this lines different
#    process.DigiToRaw.remove(process.siPixelRawData)
#    process.RawToDigi.remove(process.siPixelDigis)


#this line different
    process.pdigi.remove(process.addPileupInfo)
    
    if hasattr(process,'dqmoffline_step'):
        process.dqmoffline_step.remove(process.SiPixelTrackResidualSource)
        process.dqmoffline_step.remove(process.jetMETAnalyzer)
        process.dqmoffline_step.remove(process.hltMonMuBits)
        process.dqmoffline_step.remove(process.vbtfAnalyzer)
        process.dqmoffline_step.remove(process.hltResults)
        process.dqmoffline_step.remove(process.egHLTOffDQMSource)
        process.dqmoffline_step.remove(process.globalAnalyzer)
        process.dqmoffline_step.remove(process.jetMETHLTOfflineSource)

    if hasattr(process,'validation_step'):
        process.validation_step.remove(process.hltHITval)
        process.validation_step.remove(process.HLTSusyExoVal)
        process.validation_step.remove(process.relvalMuonBits)
    else:
    ## removing large memory usage module if we don't need it
        process.pdigi.remove(process.mergedtruth)

    return(process)


#pileup specific stuff here
def customise_pu50_25ns(process):

    process=customise(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_stdgeom_cff")

### set the number of pileup
    process.mix.input.nbPileupEvents = cms.PSet(
        averageNumber = cms.double(50.0)
        )
    

### if doing inefficiency at <PU>=50
    process.simSiPixelDigis.AddPixelInefficiency = 20
    ## also for strips TIB inefficiency if we want
    ## TIB1,2 inefficiency at 20%
    #process.simSiStripDigis.Inefficiency = 20
    ## TIB1,2 inefficiency at 50%
    #process.simSiStripDigis.Inefficiency = 30
    ## TIB1,2 inefficiency at 99% (i.e. dead)
    #process.simSiStripDigis.Inefficiency = 40
    
    return (process)
