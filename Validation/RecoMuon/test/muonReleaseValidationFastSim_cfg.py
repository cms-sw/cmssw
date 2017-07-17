#import FWCore.ParameterSet.Config as cms

FastSim=FASTSIM
onlyRecoMuons=ONLYRECOMUONS

process = cms.Process("PROCESSNAME")
process.load("FWCore.MessageService.MessageLogger_cfi")

### standard includes
process.load('Configuration.StandardSequences.GeometryPilot2_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
if (FastSim):
    process.load("FastSimulation.Configuration.mixNoPU_cfi")
else:
    process.load("SimGeneral.MixingModule.mixNoPU_cfi")

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GLOBALTAG::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(NEVENT)
)
process.source = source

### validation-specific includes
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoMuon.MuonTrackValidator_cff")
if (FastSim):
    process.load("FastSimulation.Validation.trackingParticlesFastSim_cfi")
else:
    process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

if (FastSim):
    process.load("Validation.RecoMuon.muonValidationFastSim_cff")
    process.load("Validation.RecoMuon.muonValidationHLTFastSim_cff")
else:
    process.load("Validation.RecoMuon.muonValidation_cff")
    process.load("Validation.RecoMuon.muonValidationHLT_cff")


process.endjob_step = cms.Path(process.endOfProcess)

#process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load('Configuration.StandardSequences.EDMtoMEAtJobEnd_cff')

process.load("Validation.Configuration.postValidation_cff")
process.load("HLTriggerOffline.Muon.HLTMuonPostVal_cff")
if (onlyRecoMuons):
    from Validation.Configuration.postValidation_cff import *
    postValidation.remove(postProcessorTrackSequence)
    postValidation_fastsim.remove(postProcessorTrackSequence)
    from HLTriggerOffline.Muon.HLTMuonPostVal_cff import *
    HLTMuonPostVal.remove(hltMuonPostProcessors)
    HLTMuonPostVal_FastSim.remove(hltMuonPostProcessors)

process.cutsRecoTracks.algorithm = ['ALGORITHM']
process.cutsRecoTracks.quality = ['QUALITY']

process.muonTrackValidator.associators = ['trackAssociatorByHits']

process.muonTrackValidator.label = ['TRACKS']
if (process.muonTrackValidator.label[0] == 'generalTracks'):
    process.muonTrackValidator.UseAssociators = cms.bool(True)
else:
    process.muonTrackValidator.UseAssociators = cms.bool(True)
######


process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

if (FastSim):
    process.recoMuonValidationSequence = cms.Sequence(process.trackAssociatorByHits
                                                      *process.muonTrackValidator
                                                      *process.recoMuonAssociationFastSim 
                                                      *process.recoMuonValidationFastSim
                                                      *process.recoMuonAssociationHLTFastSim_seq
                                                      *process.recoMuonValidationHLTFastSim_seq)
elif ('SAMPLE'=='RelValCosmics'):
    process.recoMuonValidationSequence = cms.Sequence(process.recoCosmicMuonValidation)
else:
    process.recoMuonValidationSequence = cms.Sequence(process.trackAssociatorByHits
                                                      *process.muonTrackValidator
                                                      *process.recoMuonValidation
                                                      *process.recoMuonValidationHLT_seq)

process.digi2track = cms.Sequence(process.siPixelDigis
                                  *process.SiStripRawToDigis
                                  *(process.trackerlocalreco
                                    +process.muonlocalreco)
                                  *(process.ckftracks
                                    *process.muonreco_plus_isolation)
                                  *process.cutsRecoTracks
                                  *process.recoMuonValidationSequence
                                  )
#redo also tracking particles
process.digi2track_and_TP = cms.Sequence(process.mix
                                         *process.trackingParticles
                                         *process.siPixelDigis
                                         *process.SiStripRawToDigis
                                         *(process.trackerlocalreco
                                           +process.muonlocalreco)
                                         *(process.ckftracks
                                           *process.muonreco_plus_isolation)
                                         *process.cutsRecoTracks
                                         *process.recoMuonValidationSequence
                                         )

process.re_tracking = cms.Sequence(process.siPixelRecHits
                                   *process.siStripMatchedRecHits
                                   *(process.ckftracks
                                     *process.muonreco_plus_isolation)
                                   *process.cutsRecoTracks
                                   *process.recoMuonValidationSequence
                                   )

process.re_tracking_and_TP = cms.Sequence(process.mix
                                          *process.trackingParticles
                                          *process.siPixelRecHits
                                          *process.siStripMatchedRecHits
                                          *(process.ckftracks
                                            *process.muonreco_plus_isolation)
                                          *process.cutsRecoTracks
                                          *process.recoMuonValidationSequence
                                          )

if (process.muonTrackValidator.label[0] == 'generalTracks'):

    process.only_validation = cms.Sequence(process.recoMuonValidationSequence)
else:
    process.only_validation = cms.Sequence(process.cutsRecoTracks
                                           *process.recoMuonValidationSequence
                                           )
    
if (process.muonTrackValidator.label[0] == 'generalTracks'):
    process.only_validation_and_TP = cms.Sequence(process.mix
                                                  *process.trackingParticles
                                                  *process.recoMuonValidationSequence
                                                  )
else:
    process.only_validation_and_TP = cms.Sequence(process.mix
                                                  *process.trackingParticles
                                                  *process.cutsRecoTracks
                                                  *process.recoMuonValidationSequence
                                                  )


### customized version of the OutputModule
### it saves the mininal information which is necessary to perform tracking validation (tracks, tracking particles, 
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
                                  fileName = cms.untracked.string('fullOutput.SAMPLE.root')
                                  )

process.VALOUTPUT = cms.OutputModule("PoolOutputModule",
                                     outputCommands = cms.untracked.vstring('drop *', "keep *_MEtoEDMConverter_*_*"),

                                     fileName = cms.untracked.string('output.SAMPLE.root')
)

ValidationSequence="SEQUENCE"

if ValidationSequence=="harvesting":
    process.DQMStore.collateHistograms = False

    process.dqmSaver.convention = 'Offline'

    process.dqmSaver.saveByRun = cms.untracked.int32(-1)
    process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
    process.dqmSaver.forceRunNumber = cms.untracked.int32(1)


    process.dqmSaver.workflow = "/GLOBALTAG/SAMPLE/Validation"
    process.DQMStore.verbose=3

    process.options = cms.untracked.PSet(
        fileMode = cms.untracked.string('FULLMERGE')
        )
    for filter in (getattr(process,f) for f in process.filters_()):
        if hasattr(filter,"outputFile"):
            filter.outputFile=""

if (FastSim):
    process.harvesting= cms.Sequence(
#        process.EDMtoMEConverter
        process.EDMtoME
        *process.postValidation_fastsim
        *process.HLTMuonPostVal_FastSim
        *process.dqmSaver)
else:
    process.harvesting= cms.Sequence(
#        process.EDMtoMEConverter
        process.EDMtoME
        *process.postValidation
        *process.HLTMuonPostVal
        *process.dqmSaver)


### final path and endPath
process.p = cms.Path(process.SEQUENCE)
if ValidationSequence!="harvesting":
    process.outpath = cms.EndPath(process.VALOUTPUT)

if ValidationSequence!="harvesting":
    process.schedule = cms.Schedule(
        process.p,
        process.endjob_step,
        process.outpath
        )
else:
    process.schedule = cms.Schedule(
        process.p
        )
