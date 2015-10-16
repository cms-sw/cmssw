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
        process=customise_RecoFull(process)
    if hasattr(process,'famosWithEverything'):
        process=customise_RecoFast(process)
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
    process.load('SimMuon.GEMDigitizer.muonME0DigisPreReco_cfi')
    process.muonDigi += process.simMuonME0Digis
    # process.load('SimCalorimetry.Configuration.SimCalorimetry_cff')
    # process.digitisation_step.remove(calDigi)
    process=outputCustoms(process)
    return process

def customise_L1Emulator(process):
    return process

def customise_DigiToRaw(process):
    return process

def customise_RawToDigi(process):
    return process

def customise_LocalReco(process):
    process.load('RecoLocalMuon.GEMRecHit.me0LocalReco_cff')
    process.muonlocalreco += process.me0LocalReco
    process=outputCustoms(process)
    return process

def customise_GlobalRecoInclude(process):
    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
    process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
    return process

def customise_GlobalRecoFast(process):
    customise_GlobalRecoInclude(process)
    # process.load('RecoMuon.MuonIdentification.me0MuonReco_cff')
    # process.reconstructionWithFamos += process.me0MuonReco
    return process

def customise_GlobalRecoFull(process):
    customise_GlobalRecoInclude(process)
    # process.load('RecoMuon.MuonIdentification.me0MuonReco_cff')
    # process.muonGlobalReco += process.me0MuonReco
    return process

def customise_RecoFast(process):
    process=customise_LocalReco(process)
    process=customise_GlobalRecoFast(process)
    process=outputCustoms(process)
    return process

def customise_RecoFull(process):
    process=customise_LocalReco(process)
    process=customise_GlobalRecoFull(process)
    process=outputCustoms(process)
    return process

def customise_Validation(process):
    process.load('Validation.Configuration.gemSimValid_cff')
    process.genvalid_all += process.me0SimValid
    return process

def customise_DQM(process):
    return process

def customise_harvesting(process):
    return process

def outputCustoms(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonME0Digis_*_*')
            getattr(process,b).outputCommands.append('keep *_me0RecHits_*_*')
            getattr(process,b).outputCommands.append('keep *_me0Segments_*_*')
            getattr(process,b).outputCommands.append('keep *_me0SegmentProducer_*_*')
            getattr(process,b).outputCommands.append('drop *_me0SegmentMatcher_*_*')
            getattr(process,b).outputCommands.append('drop *_me0MuonConverter_*_*')
            getattr(process,b).outputCommands.append('keep *_me0SegmentMatching_*_*')
            getattr(process,b).outputCommands.append('keep *_me0MuonConverting_*_*')
    return process
