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

    process.load('Validation.GlobalDigis.globaldigis_analyze_cfi')
    process.load('Validation.GlobalRecHits.globalrechits_analyze_cfi')
    process.load('Validation.GlobalHits.globalhits_analyze_cfi')
    process.load("Validation/Configuration/noiseSimValid_cff")
    process.local_validation = cms.Path(process.globalhitsanalyze+process.globaldigisanalyze+process.noiseSimValid)
    process.schedule.append(process.local_validation)

    process.schedule.append(process.endjob_step)
    process.schedule.append(process.out_step)

    return(process)
