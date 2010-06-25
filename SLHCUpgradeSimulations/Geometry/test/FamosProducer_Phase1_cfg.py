import FWCore.ParameterSet.Config as cms

process = cms.Process("Fastsimwdigi")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.RandomNumberGeneratorService.simSiStripDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))
process.RandomNumberGeneratorService.simSiPixelDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))

# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=180GeV/c2
#  process.load("FastSimulation.Configuration.HZZllll_cfi")
# Generate ttbar events
#  process.load("FastSimulation/Configuration/ttbar_cfi")
# Generate multijet events with different ptHAT bins
#  process.load("FastSimulation/Configuration/QCDpt80-120_cfi")
#  process.load("FastSimulation/Configuration/QCDpt600-800_cfi")
# Generate Minimum Bias Events
#  process.load("FastSimulation/Configuration/MinBiasEvents_cfi")
# Generate muons with a flat pT particle gun
process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
process.generator.PGunParameters.PartID[0] = 13
#process.generator.PGunParameters.PartID[0] = 211
## for 4 muons to test with vertex
#process.FlatRandomPtGunSource.PGunParameters.PartID = cms.untracked.vint32(13,-13,13,-13)
## for opposite sign back-to-back dimuon pairs
process.generator.PGunParameters.MinPt = 0.9
process.generator.PGunParameters.MaxPt = 50.0
process.generator.PGunParameters.MinEta = -2.4
process.generator.PGunParameters.MaxEta = 2.4
process.generator.AddAntiParticle = True
# Generate di-electrons with pT=35 GeV
# process.load("FastSimulation/Configuration/DiElectrons_cfi")

# from std full sim 
#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_36Y_V9::All'

# Famos sequences (fake conditions)
#process.load("FastSimulation.Configuration.CommonInputsFake_cff")
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
# replace with strawman geometry
process.load("SLHCUpgradeSimulations.Geometry.PhaseI_cmsSimIdealGeometryXML_cff")

process.siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
#    file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt')
    file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/EmptyPixelSkimmedGeometry_phase1.txt')
)
process.es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

process.siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
    file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt')
)
process.es_prefer_fake_lorentz = cms.ESPrefer("SiPixelFakeLorentzAngleESSource","siPixelFakeLorentzAngleESSource")

process.load("CalibTracker.SiStripESProducers.fake.SiStripNoisesFakeESSource_cfi")
process.SiStripNoisesGenerator.NoiseStripLengthSlope=cms.vdouble(51.) #dec mode
process.SiStripNoisesGenerator.NoiseStripLengthQuote=cms.vdouble(630.)

process.siStripNoisesFakeESSource  = cms.ESSource("SiStripNoisesFakeESSource")
process.es_prefer_fake_strip_noise = cms.ESPrefer("SiStripNoisesFakeESSource",
                                                  "siStripNoisesFakeESSource")

process.load("CalibTracker.SiStripESProducers.fake.SiStripQualityFakeESSource_cfi")

process.siStripQualityFakeESSource  = cms.ESSource("SiStripQualityFakeESSource")
process.es_prefer_fake_strip_quality = cms.ESPrefer("SiStripQualityFakeESSource",
                                                     "siStripQualityFakeESSource")

process.load("CalibTracker.SiStripESProducers.fake.SiStripPedestalsFakeESSource_cfi")

process.siStripPedestalsFakeESSource  = cms.ESSource("SiStripPedestalsFakeESSource")
process.es_prefer_fake_strip_pedestal = cms.ESPrefer("SiStripPedestalsFakeESSource",
                                                     "siStripPedestalsFakeESSource")

process.load("CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi")

process.siStripLorentzAngleFakeESSource  = cms.ESSource("SiStripLorentzAngleFakeESSource")
process.es_prefer_fake_strip_LA = cms.ESPrefer("SiStripLorentzAngleFakeESSource",
                                               "siStripLorentzAngleFakeESSource")

process.siStripLorentzAngleSimFakeESSource  = cms.ESSource("SiStripLorentzAngleSimFakeESSource")
process.es_prefer_fake_strip_LA_sim = cms.ESPrefer("SiStripLorentzAngleSimFakeESSource",
                                                   "siStripLorentzAngleSimFakeESSource")

process.load("CalibTracker.SiStripESProducers.fake.SiStripApvGainFakeESSource_cfi")
process.SiStripApvGainGenerator.MeanGain=1.0
process.SiStripApvGainGenerator.SigmaGain=0.0
process.SiStripApvGainGenerator.genMode = cms.string("default")

process.myStripApvGainFakeESSource = cms.ESSource("SiStripApvGainFakeESSource")
process.es_prefer_myStripApvGainFakeESSource  = cms.ESPrefer("SiStripApvGainFakeESSource",
                                                  "myStripApvGainFakeESSource")

process.myStripApvGainSimFakeESSource  = cms.ESSource("SiStripApvGainSimFakeESSource")
process.es_prefer_myStripApvGainSimFakeESSource = cms.ESPrefer("SiStripApvGainSimFakeESSource",
                                                               "myStripApvGainSimFakeESSource")

process.load("CalibTracker.SiStripESProducers.fake.SiStripThresholdFakeESSource_cfi")

process.siStripThresholdFakeESSource  = cms.ESSource("SiStripThresholdFakeESSource")
process.es_prefer_fake_strip_threshold = cms.ESPrefer("SiStripThresholdFakeESSource",
                                                     "siStripThresholdFakeESSource")

process.TrackerDigiGeometryESModule.applyAlignment = False

process.TrackerGeometricDetESModule.fromDDD=cms.bool(True)
process.TrackerDigiGeometryESModule.fromDDD=cms.bool(True)
#print process.TrackerGeometricDetESModule.fromDDD
#print process.TrackerDigiGeometryESModule.fromDDD,process.TrackerDigiGeometryESModule.applyAlignment

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

#process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

# Replace std 10 TeV with 14 TeV pileup files and set the vertex smearing like signal
import FastSimulation.Event.GaussianVertexGenerator_cfi as GaussSmearing
process.famosPileUp.VertexGenerator = cms.PSet( GaussSmearing.myVertexGenerator )
#import FastSimulation.PileUpProducer.PileUpSimulator_cfi as Pileup14TeV
#process.famosPileUp.PileUpSimulator = cms.PSet( Pileup14TeV.PileUpSimulatorBlock.PileUpSimulator )

# Make sure CoM energy is 14 TeV if we are using pythia for the signal source
#process.PythiaSource.comEnergy = cms.untracked.double(14000.0)
#process.PythiaSource.maxEventsToPrint = 1

# If you want to turn on/off pile-up
process.famosPileUp.PileUpSimulator.averageNumber = 5.0
# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True

## make occupancies more similar to full simulation
process.famosSimHits.ParticleFilter.etaMax = 3.0
process.famosSimHits.ParticleFilter.pTMin = 0.05
process.famosSimHits.TrackerSimHits.pTmin = 0.05
process.famosSimHits.TrackerSimHits.firstLoop = False

process.load("SimTracker.Configuration.SimTracker_cff")
process.simSiPixelDigis.ROUList =  ['famosSimHitsTrackerHits']
process.simSiPixelDigis.MissCalibrate = False
process.simSiPixelDigis.LorentzAngle_DB = False
process.simSiPixelDigis.killModules = False
process.simSiPixelDigis.useDB = False
process.simSiPixelDigis.DeadModules_DB = False
process.simSiPixelDigis.NumPixelBarrel = cms.int32(4)
process.simSiPixelDigis.NumPixelEndcap = cms.int32(3)
## set pixel inefficiency if we want it
## 100% efficiency
process.simSiPixelDigis.AddPixelInefficiency = -1
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


#process.simSiStripDigis.Noise = False
process.simSiStripDigis.ROUList =  ['famosSimHitsTrackerHits']

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.siPixelClusters.src = 'simSiPixelDigis'
process.siPixelClusters.MissCalibrate = False
#process.siStripZeroSuppression.RawDigiProducersList[0].RawDigiProducer = 'simSiStripDigis'
#process.siStripZeroSuppression.RawDigiProducersList[1].RawDigiProducer = 'simSiStripDigis'
#process.siStripZeroSuppression.RawDigiProducersList[2].RawDigiProducer = 'simSiStripDigis'
#process.siStripClusters.DigiProducersList[0].DigiProducer= 'simSiStripDigis'

process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag(cms.InputTag('simSiStripDigis','VirginRaw'),
                                                                    cms.InputTag('simSiStripDigis','ProcessedRaw'),
                                                                    cms.InputTag('simSiStripDigis','ScopeMode'))
process.siStripClusters.DigiProducersList = cms.VInputTag(cms.InputTag('simSiStripDigis','ZeroSuppressed'),
                                                          cms.InputTag('siStripZeroSuppression','VirginRaw'),
                                                          cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
                                                          cms.InputTag('siStripZeroSuppression','ScopeMode'))


# change from default of 8bit ADC (255) for stack layers (1=1 bit, 7=3 bits)
# need to change both digitizer and clusterizer
#process.simSiPixelDigis.AdcFullScaleStack = cms.int32(1)
#process.siPixelClusters.AdcFullScaleStack = cms.int32(1)
# probably no need to change default stack layer start
#process.simSiPixelDigis.FirstStackLayer = cms.int32(5)
#process.siPixelClusters.FirstStackLayer = cms.int32(5)

process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.mergedtruth.simHitCollections.tracker = ['famosSimHitsTrackerHits']
process.mergedtruth.simHitCollections.pixel = []
process.mergedtruth.simHitCollections.muon = []
process.mergedtruth.simHitLabel = 'famosSimHits'

process.load("Validation.RecoTrack.cutsTPEffic_cfi")
process.load("Validation.RecoTrack.cutsTPFake_cfi")
## if mergedBremsstrahlung is False
#process.cutsTPEffic.src = cms.InputTag("mergedtruth")
#process.cutsTPFake.src = cms.InputTag("mergedtruth")

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
process.multiTrackValidator.skipHistoFit = False
##### with John's changes ##############################
process.load("SLHCUpgradeSimulations.Geometry.oldTracking_wtriplets")
process.pixellayertriplets.layerList = cms.vstring('BPix1+BPix2+BPix3',
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

### make sure the correct (modified) error routine is used
process.siPixelRecHits.CPE = 'PixelCPEfromTrackAngle'
process.MeasurementTracker.PixelCPE = 'PixelCPEfromTrackAngle'
process.ttrhbwr.PixelCPE = 'PixelCPEfromTrackAngle'
process.mixedlayerpairs.BPix.TTRHBuilder = cms.string('WithTrackAngle')
process.mixedlayerpairs.FPix.TTRHBuilder = cms.string('WithTrackAngle')
process.pixellayertriplets.BPix.TTRHBuilder = cms.string('WithTrackAngle')
process.pixellayertriplets.FPix.TTRHBuilder = cms.string('WithTrackAngle')
process.ctfWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')

#process.MeasurementTracker.stripClusterProducer=cms.string('')
process.MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
#Prevent strips...

#next may not be needed
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.TTRHBuilder = cms.string('WithTrackAngle')

#next may not be needed
process.load("RecoTracker.SiTrackerMRHTools.SiTrackerMultiRecHitUpdator_cff")
process.siTrackerMultiRecHitUpdator.TTRHBuilder = cms.string('WithTrackAngle')

#replace with correct component in cloned version (replace with original TTRH producer)
#process.preFilterFirstStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.secPixelRecHits.CPE = cms.string('PixelCPEfromTrackAngle')
process.seclayertriplets.BPix.TTRHBuilder = cms.string('WithTrackAngle')
process.seclayertriplets.FPix.TTRHBuilder = cms.string('WithTrackAngle')
process.secMeasurementTracker.PixelCPE = cms.string('PixelCPEfromTrackAngle')
process.secWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.thPixelRecHits.CPE = cms.string('PixelCPEfromTrackAngle')
process.thlayertripletsa.BPix.TTRHBuilder = cms.string('WithTrackAngle')
process.thlayertripletsa.FPix.TTRHBuilder = cms.string('WithTrackAngle')
process.thMeasurementTracker.PixelCPE = cms.string('PixelCPEfromTrackAngle')
process.thWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')

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

