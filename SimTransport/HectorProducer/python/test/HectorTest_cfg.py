import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')

process.load('SimTransport/HectorProducer/HectorTransport_cfi')
process.transport = cms.Path(process.LHCTransport)

process.load('Configuration/EventContent/EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('SinglePiPt10.cfi nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

# Input source

process.load('Configuration/Generator/MinBias_cfi')

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('HectorTest.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN'),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements

# Path and EndPath definitions
process.generation_step = cms.Path(process.ProductionFilterSequence*process.pgen)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.transport,process.out_step)

def customise(process):
    #Adding SimpleMemoryCheck service:
    process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                          ignoreTotal=cms.untracked.int32(1),
                                          oncePerEventMode=cms.untracked.bool(True))
    #Adding Timing service:
    process.Timing=cms.Service("Timing")
    
    #Add these 3 lines to put back the summary for timing information at the end of the logfile
    #(needed for TimeReport report)
    process.options = cms.untracked.PSet(
        wantSummary = cms.untracked.bool(True)

        )

    process.LHCTransport.Verbosity = cms.bool(True)
    process.source.pythiaHepMCVerbosity = cms.untracked.bool(True)
    process.source.maxEventsToPrint = cms.untracked.int32(-1)

    return(process)


# End of customisation function definition

process = customise(process)
