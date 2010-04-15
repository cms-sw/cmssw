def customise(process):

    #Renaming the process
    process.__dict__['_Process__name']='RestoringSeeds'

    #Optionally skipping the first events 
    #process.PoolSource.skipEvents=cms.untracked.uint32(3)

    #Drop on input everything but seeds to be restored
    process.source.inputCommands=cms.untracked.vstring('drop *',
         'keep RandomEngineStates_*_*_*')

    #Adding RandomNumberGeneratorService  
    process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string('randomEngineStateProducer')

    return(process)
