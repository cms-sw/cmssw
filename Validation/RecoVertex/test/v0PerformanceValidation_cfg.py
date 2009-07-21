process = cms.Process("VeeVal")
process.load("FWCore.MessageService.MessageLogger_cfi")

### standard includes
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP_V1::All'
process.GlobalTag.globaltag = 'GLOBALTAG::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(NEVENT)
)
process.source = source

### validation-specific includes
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoVertex.VertexValidation_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Validation.Configuration.postValidation_cff")

process.v0Validator.DQMRootFileName = ''

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

ValidationSequence="SEQUENCE"

if ValidationSequence == "only_validation":
    process.postProcessorV0.outputFileName = 'val.SAMPLE.root'

process.only_validation = cms.Sequence(process.trackingParticleRecoTrackAsssociation*process.v0Validator*process.postProcessorV0)


# Need to put in a PoolOutputModule at some point, for which I need to figure out
#  what the optimal event content would be

if ValidationSequence == "harvesting":
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
        if hasattr(filter, "outputFile"):
            filter.outputFile=""

process.harvesting = cms.Sequence(process.postValidation*process.EDMtoMEConverter*process.dqmSaver)

process.p = cms.Path(process.SEQUENCE)

process.schedule = cms.Schedule(process.p)
