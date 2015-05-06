#import FWCore.ParameterSet.Config as cms

process = cms.Process("TkVal")
process.load("FWCore.MessageService.MessageLogger_cfi")

### standard includes
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContentCosmics_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("RecoTracker.TrackProducer.TrackRefitterP5_cfi")


### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP_V1::All'
process.GlobalTag.globaltag = 'GLOBALTAG::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(NEVENT)
)
process.source = source

### validation-specific includes
#process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")
#process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cosmiccuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Validation.Configuration.postValidation_cff")

process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

## configuration of the cuts on CTF tracks
process.cutsRecoCTFTracksP5.minHit = cms.int32(NHITRECOTRKMIN)
process.cutsRecoCTFTracksP5.ptMin = cms.double(PTRECOTRKMIN)
process.cutsRecoCTFTracksP5.maxRapidity = cms.double(ETARECOTRKMAX)
process.cutsRecoCTFTracksP5.minRapidity = cms.double(ETARECOTRKMIN)
process.cutsRecoCTFTracksP5.lip = cms.double(LIPRECOTRKMAX)
process.cutsRecoCTFTracksP5.tip = cms.double(TIPRECOTRKMAX)
process.cutsRecoCTFTracksP5.maxChi2 = cms.double(CHISQRECOTRKMAX)

## configuration of the cuts on CosmicTF tracks
process.cutsRecoCosmicTFTracksP5.minHit = cms.int32(NHITRECOTRKMIN)
process.cutsRecoCosmicTFTracksP5.ptMin = cms.double(PTRECOTRKMIN)
process.cutsRecoCosmicTFTracksP5.maxRapidity = cms.double(ETARECOTRKMAX)
process.cutsRecoCosmicTFTracksP5.minRapidity = cms.double(ETARECOTRKMIN)
process.cutsRecoCosmicTFTracksP5.lip = cms.double(LIPRECOTRKMAX)
process.cutsRecoCosmicTFTracksP5.tip = cms.double(TIPRECOTRKMAX)
process.cutsRecoCosmicTFTracksP5.maxChi2 = cms.double(CHISQRECOTRKMAX)

## configuration of the cuts on RS tracks
process.cutsRecoRSTracksP5.minHit = cms.int32(NHITRECOTRKMIN)
process.cutsRecoRSTracksP5.ptMin = cms.double(PTRECOTRKMIN)
process.cutsRecoRSTracksP5.maxRapidity = cms.double(ETARECOTRKMAX)
process.cutsRecoRSTracksP5.minRapidity = cms.double(ETARECOTRKMIN)
process.cutsRecoRSTracksP5.lip = cms.double(LIPRECOTRKMAX)
process.cutsRecoRSTracksP5.tip = cms.double(TIPRECOTRKMAX)
process.cutsRecoRSTracksP5.maxChi2 = cms.double(CHISQRECOTRKMAX)

### configuration MultiTrackValidator ###
process.multiTrackValidator.outputFile = 'val.SAMPLE.root'
process.multiTrackValidator.UseAssociators = cms.bool(True)
process.multiTrackValidator.associators = ['quickTrackAssociatorByHits']
process.multiTrackValidator.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
process.multiTrackValidator.label = ['cutsRecoCTFTracksP5', 'cutsRecoCosmicTFTracksP5', 'cutsRecoRSTracksP5']
process.multiTrackValidator.useLogPt=cms.untracked.bool(False)
process.multiTrackValidator.min = cms.double(MINETA)
process.multiTrackValidator.max = cms.double(MAXETA)
process.multiTrackValidator.nint = cms.int32(NINTETA)
process.multiTrackValidator.minpT = cms.double(MINPT)
process.multiTrackValidator.maxpT = cms.double(MAXPT)
process.multiTrackValidator.nintpT = cms.int32(NINTPT)
process.multiTrackValidator.minDxy = cms.double(MINDXY) 
process.multiTrackValidator.maxDxy = cms.double(MAXDXY)
process.multiTrackValidator.nintDxy = cms.int32(NINTDXY)
process.multiTrackValidator.minDz = cms.double(MINDZ)
process.multiTrackValidator.maxDz = cms.double(MAXDZ)
process.multiTrackValidator.nintDz = cms.int32(NINTDZ)
process.multiTrackValidator.minPhi = cms.double(MINPHI)
process.multiTrackValidator.maxPhi = cms.double(MAXPHI)
process.multiTrackValidator.nintPhi = cms.int32(NINTPHI)
process.multiTrackValidator.minVertpos = cms.double(MINVERTPOS) 
process.multiTrackValidator.maxVertpos = cms.double(MAXVERTPOS)
process.multiTrackValidator.nintVertpos = cms.int32(NINTVERTPOS)
process.multiTrackValidator.minZpos = cms.double(MINZPOS) 
process.multiTrackValidator.maxZpos = cms.double(MAXZPOS)
process.multiTrackValidator.nintZpos = cms.int32(NINTZPOS)
process.multiTrackValidator.dxyRes_rangeMin = cms.double(DXYRESMIN) 
process.multiTrackValidator.dxyRes_rangeMax = cms.double(DXYRESMAX)
process.multiTrackValidator.dxyRes_nbin = cms.int32(100)
process.multiTrackValidator.dzRes_rangeMin = cms.double(DZRESMIN)
process.multiTrackValidator.dzRes_rangeMax = cms.double(DZRESMAX)
process.multiTrackValidator.dzRes_nbin = cms.int32(100)
process.multiTrackValidator.phiRes_rangeMin = cms.double(PHIRESMIN)
process.multiTrackValidator.phiRes_rangeMax = cms.double(PHIRESMAX)
process.multiTrackValidator.phiRes_nbin = cms.int32(100)
process.multiTrackValidator.ptRes_rangeMin = cms.double(PTRESMIN)
process.multiTrackValidator.ptRes_rangeMax = cms.double(PTRESMAX)
process.multiTrackValidator.ptRes_nbin = cms.int32(100)
process.multiTrackValidator.cotThetaRes_rangeMin = cms.double(COTTHETARESMIN)
process.multiTrackValidator.cotThetaRes_rangeMax = cms.double(COTTHETARESMAX)
process.multiTrackValidator.cotThetaRes_nbin = cms.int32(100)
process.multiTrackValidator.ptMinTP = cms.double(PTTPMIN)
process.multiTrackValidator.lipTP = cms.double(LIPTPMAX)
process.multiTrackValidator.tipTP = cms.double(TIPTPMAX)
process.multiTrackValidator.minRapidityTP = cms.double(ETATPMIN) 
process.multiTrackValidator.maxRapidityTP = cms.double(ETATPMAX)
process.multiTrackValidator.minHitTP = cms.int32(NHITTPMIN) 

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.re_tracking = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.tracksP5*
                                   process.cutsRecoCTFTracksP5*process.cutsRecoCosmicTFTracksP5*process.cutsRecoRSTracksP5*
                                   process.quickTrackAssociatorByHits*
                                   process.multiTrackValidator
                                   )

process.re_tracking_and_TP = cms.Sequence(process.trackingParticles*
                                   process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.tracksP5*
                                   process.cutsRecoCTFTracksP5*process.cutsRecoCosmicTFTracksP5*process.cutsRecoRSTracksP5*
                                   process.quickTrackAssociatorByHits*
                                   process.multiTrackValidator
                                   )

process.only_validation = cms.Sequence(process.cutsRecoCTFTracksP5*process.cutsRecoCosmicTFTracksP5*process.cutsRecoRSTracksP5*
                                       process.quickTrackAssociatorByHits*
                                       process.multiTrackValidator)
    
process.only_validation_and_TP = cms.Sequence(process.trackingParticles*
                                              process.cutsRecoCTFTracksP5*process.cutsRecoCosmicTFTracksP5*process.cutsRecoRSTracksP5*
                                              process.quickTrackAssociatorByHits*
                                              process.multiTrackValidator)

### customized versoin of the OutputModule
### it save the mininal information which is necessary to perform tracking validation (tracks, tracking particles, 
### digiSimLink,etc..)

process.customEventContent = cms.PSet(
     outputCommands = cms.untracked.vstring('drop *')
 )

process.customEventContent.outputCommands.extend(process.RecoTrackerRECO.outputCommands)
process.customEventContent.outputCommands.extend(process.BeamSpotRECO.outputCommands)
process.customEventContent.outputCommands.extend(process.SimGeneralFEVTDEBUG.outputCommands)
process.customEventContent.outputCommands.extend(process.RecoLocalTrackerRECO.outputCommands)
process.customEventContent.outputCommands.append('keep *_simSiStripDigis_*_*')
process.customEventContent.outputCommands.append('keep *_simSiPixelDigis_*_*')
process.customEventContent.outputCommands.append('drop SiStripDigiedmDetSetVector_simSiStripDigis_*_*')
process.customEventContent.outputCommands.append('drop PixelDigiedmDetSetVector_simSiPixelDigis_*_*')



process.OUTPUT = cms.OutputModule("PoolOutputModule",
                                  process.customEventContent,
                                  fileName = cms.untracked.string('output.SAMPLE.root')
                                  )


### final path and endPath
process.p = cms.Path(process.SEQUENCE)
#process.outpath = cms.EndPath(process.OUTPUT)


