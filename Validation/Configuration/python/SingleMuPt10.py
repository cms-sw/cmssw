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

    process.load("Validation/Configuration/trackerSimValid_cff")
    process.load("Validation/Configuration/muonSimValid_cff")
    process.local_validation = cms.Path((process.globalhitsanalyze+process.globaldigisanalyze+process.trackerSimValid+process.muonSimValid)*process.MEtoEDMConverter)
    process.schedule.append(process.local_validation)

    process.schedule.append(process.out_step)

# drop the plain root file outputs of all analyzers
# Note: all the validation "analyzers" are EDFilters!
    for filter in (getattr(process,f) for f in process.filters_()):
        print "Found analyzer (EDFilter) ",filter
        if hasattr(filter,"outputFile"):
            print "Silencing outputFile %s of %s analyzer"%(filter.outputFile, filter)
            filter.outputFile=""
        #Catch the problem with valid_HB.root that uses OutputFile instead of outputFile
        if hasattr(filter,"OutputFile"):
            print "Silencing OutputFile %s of %s analyzer"%(filter.OutputFile, filter)
            filter.OutputFile=""

    return(process)
