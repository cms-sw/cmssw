import FWCore.ParameterSet.Config as cms
def customise(process):

# user schedule: 

    del process.schedule[:]

    process.schedule.append(process.generation_step)
    process.schedule.append(process.simulation_step)
    process.schedule.append(process.digitisation_step)
    process.schedule.append(process.L1simulation_step)
    process.schedule.append(process.digi2raw_step)
    process.schedule.append(process.raw2digi_step)
    process.schedule.append(process.reconstruction_step)

    process.load("Validation/Configuration/noiseSimValid_cff")
    process.local_validation = cms.Path(process.globalhitsanalyze+process.globaldigisanalyze+process.noiseSimValid)
    process.schedule.append(process.local_validation)

    process.schedule.append(process.endjob_step)
    process.schedule.append(process.out_step)

# drop the plain root file outputs of all analyzers
# Note: all the validation "analyzers" are EDFilters!
    for filter in (getattr(process,f) for f in process.filters_()):
        if hasattr(filter,"outputFile"):
            filter.outputFile=""
        #Catch the problem with valid_HB.root that uses OutputFile instead of outputFile
        #if hasattr(filter,"OutputFile"):
        #    filter.OutputFile=""
        #In MultiTrackValidator there is an out root output file to be silenced too:
        #if hasattr(filter,"out"):
        #    filter.out=""
        #In SiPixelTrackingRecHitsValid there is a debugNtuple to be silenced too:
        #if hasattr(filter,"debugNtuple"):
        #    filter.debugNtuple=""
# In Tracker, CSC and DT validation, EDAnalyzers are used instead of EDFilters:
    for analyzer in (getattr(process,f) for f in process.analyzers_()):
        if hasattr(analyzer,"outputFile"):
            analyzer.outputFile=""
        #In MuonSimHitsValidAnalyzer there is a DT_outputFile to be silenced too:
        if hasattr(analyzer,"DT_outputFile"):
            analyzer.DT_outputFile=""

    #process.MessageLogger.categories=cms.untracked.vstring('DQMStore'
    #                                                       )
    #Configuring the standard output
    #process.MessageLogger.cout =  cms.untracked.PSet(
    #    noTimeStamps = cms.untracked.bool(True)
    #    ,threshold = cms.untracked.string('INFO')
    #    ,INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
    #    ,DQMStore = cms.untracked.PSet(limit = cms.untracked.int32(0))
    #    )
    return(process)
