import FWCore.ParameterSet.Config as cms

def customise(process):
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'L1simulation_step'):
       process=customise_L1Emulator(process)
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)

    return process

def customise_Digi(process):
    process.RandomNumberGeneratorService.simMuonME0Digis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    )

    process.mix.mixObjects.mixSH.crossingFrames.append('MuonME0Hits')
    process.mix.mixObjects.mixSH.input.append(cms.InputTag("g4SimHits","MuonME0Hits"))
    process.mix.mixObjects.mixSH.subdets.append('MuonME0Hits')

    process.load('SimMuon.ME0Digitizer.muonME0Digis_cfi')
    process.muonDigi += process.simMuonME0Digis

    process=outputCustoms(process)
    return process

def customise_L1Emulator(process):
    return process

def customise_DigiToRaw(process):
    return process

def customise_RawToDigi(process):
    return process

def customise_Reco(process):
    return process

def customise_DQM(process):
    return process

def customise_harvesting(process):
    return (process)

def customise_Validation(process):
    return process

def outputCustoms(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonME0Digis_*_*')

    return process
