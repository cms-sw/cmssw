import FWCore.ParameterSet.Config as cms

process = cms.Process("Fastsimwdigi")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.RandomNumberGeneratorService.simSiStripDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))
process.RandomNumberGeneratorService.simSiPixelDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))

process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
process.generator.PGunParameters.PartID[0] = 13
## for 4 muons to test with vertex
#process.FlatRandomPtGunSource.PGunParameters.PartID = cms.untracked.vint32(13,-13,13,-13)
## for opposite sign back-to-back dimuon pairs
process.generator.PGunParameters.MinPt = 0.9
process.generator.PGunParameters.MaxPt = 50.0
process.generator.PGunParameters.MinEta = -2.4
process.generator.PGunParameters.MaxEta = 2.4
process.generator.AddAntiParticle = True
#process.generator.Verbosity = 1

# Famos sequences (fake conditions)
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.Geometries_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
# replace with strawman geometry
#process.load("SLHCUpgradeSimulations.Geometry.PhaseI_cmsSimIdealGeometryXML_R39F16_cff")
process.load('SLHCUpgradeSimulations.Geometry.Phase1_R34F16_cmsSimIdealGeometryXML_cff')

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

process.load('SLHCUpgradeSimulations.Geometry.Digi_Phase1_R34F16_cff')
process.simSiPixelDigis.ROUList =  ['famosSimHitsTrackerHits']

## static efficiency
#process.simSiPixelDigis.AddPixelInefficiency = 0         #--Hec (default = -1)
#process.simSiPixelDigis.PixelEff     = 0.99              #--Hec (default = 1)
#process.simSiPixelDigis.PixelColEff  = 0.99              #--Hec (default = 1)
#process.simSiPixelDigis.PixelChipEff = 0.99              #--Hec (default = 1)
#  Note only static is implemented for upgrade geometries
#--PixelIneff = -1 Default Value  (No Inefficiency. eff=100%)
#             = 0  Static Efficiency
#             > 0  Luminosity rate dependent ineff
#            1,2 - low-lumi rate dependent inefficency added
#            10 - high-lumi inefficiency added

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'DESIGN42_V17::All'
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

#process.load("SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_cff")
process.load("SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_R34F16_cff")
process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
process.load("SLHCUpgradeSimulations.Geometry.upgradeTracking_phase1_cff")

process.TrackerDigiGeometryESModule.applyAlignment = False

process.TrackerGeometricDetESModule.fromDDD=cms.bool(True)
process.TrackerDigiGeometryESModule.fromDDD=cms.bool(True)


process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')

# Replace std 10 TeV with 14 TeV pileup files and set the vertex smearing like signal
import FastSimulation.Event.GaussianVertexGenerator_cfi as GaussSmearing
process.famosPileUp.VertexGenerator = cms.PSet( GaussSmearing.myVertexGenerator )

# If you want to turn on/off pile-up
process.famosPileUp.PileUpSimulator.averageNumber = 50.00
# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True

## make occupancies more similar to full simulation
process.famosSimHits.ParticleFilter.etaMax = 3.0
process.famosSimHits.ParticleFilter.pTMin = 0.05
process.famosSimHits.TrackerSimHits.pTmin = 0.05
process.famosSimHits.TrackerSimHits.firstLoop = False



#process.simSiStripDigis.Noise = False
process.simSiStripDigis.ROUList =  ['famosSimHitsTrackerHits']

process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.mergedtruth.simHitCollections.tracker = ['famosSimHitsTrackerHits']
process.mergedtruth.simHitCollections.pixel = []
process.mergedtruth.simHitCollections.muon = []
process.mergedtruth.simHitLabel = 'famosSimHits'

process.load("Validation.RecoTrack.cutsTPEffic_cfi")
process.load("Validation.RecoTrack.cutsTPFake_cfi")
## if mergedBremsstrahlung is False
#process.cutsTPEffic.src = cms.InputTag("mix")
#process.cutsTPFake.src = cms.InputTag("mix")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.TrackAssociatorByHits.ROUList = ['famosSimHitsTrackerHits']

process.load("Validation.RecoTrack.MultiTrackValidator_cff")
#process.multiTrackValidator.label = ['generalTracks']
### if using simple (non-iterative) or old (as in 1_8_4) tracking
process.multiTrackValidator.label = ['ctfWithMaterialTracks']
#process.multiTrackValidator.label = ['cutsRecoTracks']
#process.multiTrackValidator.label_tp_effic = cms.InputTag("cutsTPEffic")
#process.multiTrackValidator.label_tp_fake = cms.InputTag("cutsTPFake")
process.multiTrackValidator.sim = 'famosSimHits'
process.multiTrackValidator.associators = ['TrackAssociatorByHits']
process.multiTrackValidator.UseAssociators = True
process.multiTrackValidator.outputFile = "validphase1_muon_50GeV.root"
process.multiTrackValidator.nint = cms.int32(20)
process.multiTrackValidator.nintpT = cms.int32(25)
process.multiTrackValidator.maxpT = cms.double(50.0)
process.multiTrackValidator.skipHistoFit = cms.untracked.bool(False)
process.multiTrackValidator.runStandalone = cms.bool(True)

##### with John's changes ##############################
process.load("SLHCUpgradeSimulations.Geometry.oldTracking_wtriplets")
process.PixelLayerTriplets.layerList = cms.vstring('BPix1+BPix2+BPix3',
        'BPix1+BPix3+BPix4',
        'BPix2+BPix3+BPix4',
        'BPix1+BPix2+BPix4',
        'BPix1+BPix2+FPix1_pos',
        'BPix1+BPix2+FPix1_neg',
        'BPix1+FPix1_pos+FPix2_pos',
        'BPix1+FPix1_neg+FPix2_neg',
        'BPix1+FPix2_pos+FPix3_pos',
        'BPix1+FPix2_neg+FPix3_neg',
        'FPix1_pos+FPix2_pos+FPix3_pos',
        'FPix1_neg+FPix2_neg+FPix3_neg')
# restrict vertex finding in trackingtruthprod to smaller volume (note: these numbers in mm) 
process.mergedtruth.volumeRadius = cms.double(100.0)
process.mergedtruth.volumeZ = cms.double(900.0)
process.mergedtruth.discardOutVolume = cms.bool(True)

process.cutsTPFake.tip = cms.double(10.0)
process.cutsTPFake.lip = cms.double(90.0)

#NB: tracks are already filtered by the generalTracks sequence
#for additional cuts use the cutsRecoTracks filter:
#process.load("Validation.RecoTrack.cutsRecoTracks_cfi")
#process.cutsRecoTracks.src = cms.InputTag("ctfWithMaterialTracks")
#process.cutsRecoTracks.quality = cms.string('')
#process.cutsRecoTracks.minHit = cms.int32(3)
#process.cutsRecoTracks.minHit = cms.int32(8)
#process.cutsRecoTracks.minHit = cms.int32(6
############ end John's changes ###########################

process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
process.PixelCPEGenericESProducer.SmallPitch = False
process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
process.PixelCPEGenericESProducer.DoCosmics = False

## CPE for other steps
process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.newPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.secPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.thPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.preFilterZeroStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.preFilterStepOneTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.secWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.thWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.fourthWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.fifthWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')

# Need these lines to stop some errors about missing siStripDigis collections.
# should add them to fakeConditions_Phase1_cff
process.MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.MeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.newMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.newMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.newMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.newMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.newMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.newMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.secMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.secMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.secMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.secMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.secMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.secMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.thMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.thMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.thMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.thMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.thMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.thMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.fourthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.fifthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()

#next may not be needed
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.TTRHBuilder = cms.string('WithTrackAngle')

#next may not be needed
process.load("RecoTracker.SiTrackerMRHTools.SiTrackerMultiRecHitUpdator_cff")
process.siTrackerMultiRecHitUpdator.TTRHBuilder = cms.string('WithTrackAngle')

# Need these lines to stop some errors about missing siStripDigis collections.
# should add them to fakeConditions_Phase1_cff
process.MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.MeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.newMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.newMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.newMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.newMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.newMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.newMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.secMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.secMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.secMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.secMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.secMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.secMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.thMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.thMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.thMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.thMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.thMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.thMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.fourthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.fifthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()

### produce an ntuple with hits for analysis
process.ReadLocalMeasurement = cms.EDAnalyzer("StdHitNtuplizer",
   src = cms.InputTag("siPixelRecHits"),
   stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
   rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
   matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
   simtrackHits = cms.InputTag("famosSimHits"),
   #trackProducer = cms.InputTag("generalTracks"),
   ### if using simple (non-iterative) or old (as in 1_8_4) tracking
   trackProducer = cms.InputTag("ctfWithMaterialTracks"),
   OutputFile = cms.string("stdgrechitP1_ntuple.root"),
   ### for using track hit association
   associatePixel = cms.bool(True),
   associateStrip = cms.bool(False),
   associateRecoTracks = cms.bool(False),
   ROUList = cms.vstring('famosSimHitsTrackerHits')
)
process.ReadFastsimHits = cms.EDAnalyzer("FastsimHitNtuplizer",
   HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSRecHits"),
   VerbosityLevel = cms.untracked.int32(1),
   OutputFile = cms.string("fsgrechitP1_ntuple.root")
)

process.o1 = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *',
                                           'drop *_mix_*_*'),
    fileName = cms.untracked.string('./fastsimPI_50mu.root')
)

process.outpath = cms.EndPath(process.o1)

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )
#process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )

process.Timing =  cms.Service("Timing")
process.load("FWCore/MessageService/MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("detailedInfo_phase1_mu50")
### to output debug messages for particular modules
# process.MessageLogger.detailedInfo_strawb_mu50 = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG'))
# process.MessageLogger.debugModules= cms.untracked.vstring("*")

process.anal = cms.EDAnalyzer("EventContentAnalyzer")

# Famos with tracks
process.p0 = cms.Path(process.generator)
process.p1 = cms.Path(process.famosWithTrackerHits)
process.p2 = cms.Path(process.trDigi*process.trackingParticles)
process.p3 = cms.Path(process.trackerlocalreco)
process.p6 = cms.Path(process.oldTracking_wtriplets)
#process.p6 = cms.Path(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks)
process.p7 = cms.Path(process.anal)
#process.p8 = cms.Path(process.cutsTPEffic*process.cutsTPFake*process.cutsRecoTracks*process.multiTrackValidator)
process.p8 = cms.Path(process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
#process.p9 = cms.Path(process.ReadLocalMeasurement)
process.p9 = cms.Path(process.ReadLocalMeasurement)
process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p3,process.p6,process.p8,process.p9)
#process.schedule = cms.Schedule(process.p1,process.p2,process.p3,process.p6,process.p8)

