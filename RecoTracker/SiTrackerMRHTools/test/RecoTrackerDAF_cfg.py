import FWCore.ParameterSet.Config as cms

process = cms.Process("tracking")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi
process.DAFTrajectoryBuilder = RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi.CkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
process.DAFTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cff")

process.load("Validation.RecoTrack.MultiTrackValidator_cfi")

process.load("Validation.RecoTrack.cuts_cff")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/006A68E0-B13E-DD11-A989-000423D94534.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/04A2DCEE-B13E-DD11-A967-000423D99BF2.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/2810FA1A-B33E-DD11-8AAF-000423D991F0.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/2838E0F0-B03E-DD11-8F78-000423D6CA02.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/2A854552-B43E-DD11-8FA4-000423D9870C.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/30BCCFD3-B13E-DD11-8E6C-000423D991F0.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/428E41B0-B13E-DD11-8A47-000423D991F0.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/4EC7F0DC-B33E-DD11-8FFB-001617DBD230.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/5C498794-B23E-DD11-8421-0019DB29C614.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/68A096EA-B23E-DD11-82A9-001617DF785A.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/6AEDE036-B43E-DD11-85CA-000423D98EC8.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/72DB4F53-B53E-DD11-BC68-000423D999CA.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/8288191A-B33E-DD11-999C-000423D98868.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/869F7DA6-B13E-DD11-8D80-001617C3B5E4.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/9AE834A2-B53E-DD11-8544-000423DD2F34.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/9E4CD325-B33E-DD11-A99B-000423D98DB4.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/A03995CD-B43E-DD11-8D55-000423D98DB4.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/A2CBEE69-B73E-DD11-A9DA-000423D944FC.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/A6BED709-B53E-DD11-85E5-000423D9989E.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/AE6B5D48-B43E-DD11-9F0E-001617C3B710.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/AE7FAAB5-B33E-DD11-971A-001617E30CA4.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/B24B5F14-B23E-DD11-A6DB-001617E30D0A.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/B6D7658C-B23E-DD11-8666-001617DBD224.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/BA5CFE58-B53E-DD11-B8A5-000423D94C68.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/C29D2C61-B33E-DD11-B5C0-001617E30E2C.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/C834B799-B23E-DD11-94C4-001617E30D52.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/DA873060-B33E-DD11-B3B8-001617C3B5E4.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/E867D406-B23E-DD11-B662-000423D98EC8.root',
'/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853-IDEAL_V2-2nd/0000/EADD118B-B43E-DD11-9305-000423D9997E.root')
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(1)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tracks.root')
)

process.validation = cms.Sequence(process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.newSeedFromPairs*process.newSeedFromTriplets*process.newCombinedSeeds*process.DAFTrackCandidateMaker*process.ctfWithMaterialTracksDAF*process.validation)
process.DAFTrajectoryBuilder.ComponentName = 'DAFTrajectoryBuilder'
process.DAFTrajectoryBuilder.trajectoryFilterName = 'newTrajectoryFilter'
process.DAFTrackCandidateMaker.SeedProducer = 'newCombinedSeeds'
process.DAFTrackCandidateMaker.TrajectoryBuilder = 'DAFTrajectoryBuilder'
process.DAFTrackCandidateMaker.useHitsSplitting = False
process.DAFTrackCandidateMaker.doSeedingRegionRebuilding = True
process.DAFFittingSmoother.EstimateCut = -1
process.DAFFittingSmoother.MinNumberOfHits = 3
process.multiTrackValidator.out = 'validationPlots_DAF_ttbar.root'
process.multiTrackValidator.label = ['ctfWithMaterialTracksDAF']
process.multiTrackValidator.UseAssociators = True
process.TrackAssociatorByHits.UseSplitting = False


