import FWCore.ParameterSet.Config as cms

process = cms.Process("Fullsim")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")
# replace with long barrel geometry
process.load("SLHCUpgradeSimulations.Geometry.longbarrel_cmsIdealGeometryXML_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")
process.SiPixelFakeGainOfflineESSource.file = 'SLHCUpgradeSimulations/Geometry/data/longbarrel/PixelSkimmedGeometry_empty.txt'
#process.SiPixelFakeGainOfflineESSource.file = 'SLHCUpgradeSimulations/Geometry/data/longbarrel/PixelSkimmedGeometry.txt'
process.SiPixelFakeLorentzAngleESSource.file = 'SLHCUpgradeSimulations/Geometry/data/longbarrel/PixelSkimmedGeometry.txt'

process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.destinations = cms.untracked.vstring("detailedInfo_fullLBmu50")
#process.MessageLogger.detailedInfo_strawb_mu50 = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG'))
#process.MessageLogger.debugModules= cms.untracked.vstring("PixelGeom")

# this config frament brings you the generator information
process.load("Configuration.StandardSequences.Generator_cff")

# this config frament brings you 3 steps of the detector simulation:
# -- vertex smearing (IR modeling)
# -- G4-based hit level detector simulation
# -- digitization (electronics readout modeling)
# it returns 2 sequences : 
# -- psim (vtx smearing + G4 sim)
# -- pdigi (digitization in all subsystems, i.e. tracker=pix+sistrips,
#           cal=ecal+ecal-0-suppression+hcal), muon=csc+dt+rpc)
#
process.load("Configuration.StandardSequences.Simulation_cff")

# please note the IMPORTANT: 
# in order to operate Digis, one needs to include Mixing module 
# (pileup modeling), at least in the 0-pileup mode
#
# There're 3 possible configurations of the Mixing module :
# no-pileup, low luminosity pileup, and high luminosity pileup
#
# they come, respectively, through the 3 config fragments below
#
# *each* config returns label "mix"; thus you canNOT have them
# all together in the same configuration, but only one !!!
#
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

#include "Configuration/StandardSequences/data/MixingLowLumiPileUp.cff" 
#include "Configuration/StandardSequences/data/MixingHighLumiPileUp.cff" 
process.load("Configuration.StandardSequences.L1Emulator_cff")

process.load("Configuration.StandardSequences.DigiToRaw_cff")

process.load("Configuration.StandardSequences.RawToDigi_cff")

#process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("SimTracker.Configuration.SimTracker_cff")
process.simSiPixelDigis.MissCalibrate = False
process.simSiPixelDigis.AddPixelInefficiency = -1
process.simSiPixelDigis.LorentzAngle_DB = False
process.simSiPixelDigis.killModules = False
process.simSiPixelDigis.NumPixelBarrel = cms.int32(14)
process.simSiPixelDigis.NumPixelEndcap = cms.int32(3)

process.siPixelClusters.src = 'simSiPixelDigis'
process.siPixelClusters.MissCalibrate = False
process.siStripZeroSuppression.RawDigiProducersList[0].RawDigiProducer = 'simSiStripDigis'
process.siStripZeroSuppression.RawDigiProducersList[1].RawDigiProducer = 'simSiStripDigis'
process.siStripZeroSuppression.RawDigiProducersList[2].RawDigiProducer = 'simSiStripDigis'
process.siStripClusters.DigiProducersList[0].DigiProducer= 'simSiStripDigis'

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
# replace FlatRandomPtGunSource.PGunParameters.PartID={13}
process.FlatRandomPtGunSource.PGunParameters.MinPt = 0.9
process.FlatRandomPtGunSource.PGunParameters.MaxPt = 50.0
process.FlatRandomPtGunSource.PGunParameters.MinEta = -2.4
process.FlatRandomPtGunSource.PGunParameters.MaxEta = 2.4
process.FlatRandomPtGunSource.AddAntiParticle = cms.untracked.bool(True)

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('/uscms_data/d2/cheung/slhc/testfullLB_muon_50GeV.root')
)

process.load("Validation.RecoTrack.cutsTPEffic_cfi")
process.load("Validation.RecoTrack.cutsTPFake_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.load("Validation.RecoTrack.MultiTrackValidator_cff")
#process.multiTrackValidator.label = ['generalTracks']
### if using simple (non-iterative) or old (as in 1_8_4) tracking
process.multiTrackValidator.label = ['ctfWithMaterialTracks']
process.multiTrackValidator.associators = ['TrackAssociatorByHits']
process.multiTrackValidator.UseAssociators = True
process.multiTrackValidator.outputFile = "validfullLB_muon_50GeV.root"
process.multiTrackValidator.nint = cms.int32(20)
process.multiTrackValidator.nintpT = cms.int32(25)
process.multiTrackValidator.maxpT = cms.double(50.0)

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
# restrict vertex fining in trackingtruthprod to smaller volume (note: these numbers in mm)
process.mergedtruth.volumeRadius = cms.double(100.0)
process.mergedtruth.volumeZ = cms.double(900.0)
process.mergedtruth.discardOutVolume = cms.bool(True)

process.cutsTPEffic.ptMin = cms.double(2.5)
process.cutsTPFake.ptMin = cms.double(2.0)
process.cutsTPFake.tip = cms.double(10.0)
process.cutsTPFake.lip = cms.double(90.0)
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
#next may not be needed
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.TTRHBuilder = cms.string('WithTrackAngle')

#next may not be needed
process.load("RecoTracker.SiTrackerMRHTools.SiTrackerMultiRecHitUpdator_cff")
process.siTrackerMultiRecHitUpdator.TTRHBuilder = cms.string('WithTrackAngle')

#replace with correct component in cloned version (replace with original TTRH producer)
process.preFilterFirstStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.secPixelRecHits.CPE = cms.string('PixelCPEfromTrackAngle')
process.seclayertriplets.BPix.TTRHBuilder = cms.string('WithTrackAngle')
process.seclayertriplets.FPix.TTRHBuilder = cms.string('WithTrackAngle')
process.secMeasurementTracker.PixelCPE = cms.string('PixelCPEfromTrackAngle')
process.secWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.thPixelRecHits.CPE = cms.string('PixelCPEfromTrackAngle')
process.thlayerpairs.BPix.TTRHBuilder = cms.string('WithTrackAngle')
process.thlayerpairs.FPix.TTRHBuilder = cms.string('WithTrackAngle')
process.thMeasurementTracker.PixelCPE = cms.string('PixelCPEfromTrackAngle')
process.thWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')

### produce an ntuple with pixel hits for analysis
process.ReadLocalMeasurement = cms.EDAnalyzer("StdHitNtuplizer",
   src = cms.InputTag("siPixelRecHits"),
   stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
   rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
   matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
   #trackProducer = cms.InputTag("generalTracks"),
   ### if using simple (non-iterative) or old (as in 1_8_4) tracking
   trackProducer = cms.InputTag("ctfWithMaterialTracks"),
   OutputFile = cms.string("stdgrechitfullLB_ntuple.root"),
   ### for using track hit association
   associatePixel = cms.bool(True),
   associateStrip = cms.bool(False),
   associateRecoTracks = cms.bool(False),
   ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
                         'g4SimHitsTrackerHitsPixelBarrelHighTof', 
                         'g4SimHitsTrackerHitsPixelEndcapLowTof', 
                         'g4SimHitsTrackerHitsPixelEndcapHighTof')
)

### modules to write output navigational information for tracking
#process.Tracer = cms.Service("Tracer",
#    indentation = cms.untracked.string('$$')
#)
#process.navigationSchoolAnalyzer = cms.EDAnalyzer("NavigationSchoolAnalyzer",
#    navigationSchoolName = cms.string('SimpleNavigationSchool')
#)

process.Timing =  cms.Service("Timing")

process.p0 = cms.Path(process.pgen)
process.p1 = cms.Path(process.psim)
process.p2 = cms.Path(process.pdigi)
process.p3 = cms.Path(process.L1Emulator)
#process.p4 = cms.Path(process.DigiToRaw)
#process.p5 = cms.Path(process.RawToDigi)
process.p5 = cms.Path(process.trackerlocalreco)
process.p6 = cms.Path(process.offlineBeamSpot+process.oldTracking_wtriplets)
#process.p6 = cms.Path(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks)
#process.p6 = cms.Path(process.reconstruction)
process.p7 = cms.Path(process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
#process.p7 = cms.Path(process.trackingParticles*process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
#process.p8 = cms.Path(process.writedet)
#process.p8 = cms.Path(process.navigationSchoolAnalyzer)
process.p8 = cms.Path(process.ReadLocalMeasurement)
process.outpath = cms.EndPath(process.FEVT)
#process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p3,process.p4,process.p5,process.p6,process.p7,process.outpath)
#process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p3,process.p5,process.outpath)
process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p3,process.p5,process.p6,process.p7)
