import FWCore.ParameterSet.Config as cms

def customise2019(process):
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'L1simulation_step'):
        process=customise_L1Emulator2019(process,'pt0')
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

def customise2023(process):
    process = customise2019(process)
    if hasattr(process,'L1simulation_step'):
        process=customise_L1Emulator2023(process,'pt0')
    return process

def customise_Digi(process):
    process.RandomNumberGeneratorService.simMuonGEMDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    )

    process.mix.mixObjects.mixSH.crossingFrames.append('MuonGEMHits')
    process.mix.mixObjects.mixSH.input.append(cms.InputTag("g4SimHits","MuonGEMHits"))
    process.mix.mixObjects.mixSH.subdets.append('MuonGEMHits')

    process.load('SimMuon.GEMDigitizer.muonGEMDigi_cff')
    process.muonDigi += process.muonGEMDigi
    process=outputCustoms(process)
    return process

## TODO at migration to CMSSW 7X: make all params tracked!    
def customise_L1Emulator2019(process, ptdphi):
    process.simCscTriggerPrimitiveDigis.gemPadProducer =  cms.untracked.InputTag("simMuonGEMCSCPadDigis","")
    process.simCscTriggerPrimitiveDigis.clctSLHC.clctNplanesHitPattern = 3
    process.simCscTriggerPrimitiveDigis.clctSLHC.clctPidThreshPretrig = 2
    process.simCscTriggerPrimitiveDigis.clctParam07.clctPidThreshPretrig = 2
    ## GE1/1-ME1/1
    dphi_lct_pad98 = {
        'pt0'  : { 'odd' :  2.00000000 , 'even' :  2.00000000 },
        'pt05' : { 'odd' :  0.02203510 , 'even' :  0.00930056 },
        'pt06' : { 'odd' :  0.01825790 , 'even' :  0.00790009 },
        'pt10' : { 'odd' :  0.01066000 , 'even' :  0.00483286 },
        'pt15' : { 'odd' :  0.00722795 , 'even' :  0.00363230 },
        'pt20' : { 'odd' :  0.00562598 , 'even' :  0.00304879 },
        'pt30' : { 'odd' :  0.00416544 , 'even' :  0.00253782 },
        'pt40' : { 'odd' :  0.00342827 , 'even' :  0.00230833 }
    }
    tmb = process.simCscTriggerPrimitiveDigis.tmbSLHC
    tmb.me11ILT = cms.untracked.PSet(
        ## run the upgrade algorithm
        runME11ILT = cms.untracked.bool(True),

        ## run in debug mode
        debugLUTs = cms.untracked.bool(False),
        debugMatching = cms.untracked.bool(False),

        ## use old dataformat
        useOldLCTDataFormatALCTGEM = cms.untracked.bool(True),
        
        ## copad construction
        maxDeltaBXInCoPad = cms.untracked.int32(1),
        maxDeltaPadInCoPad = cms.untracked.int32(1),

        ## matching to pads in case LowQ CLCT
        maxDeltaBXPadEven = cms.untracked.int32(1),
        maxDeltaBXPadOdd = cms.untracked.int32(1),
        maxDeltaPadPadEven = cms.untracked.int32(2),
        maxDeltaPadPadOdd = cms.untracked.int32(3),

        ## matching to pads in case absent CLCT
        maxDeltaBXCoPadEven = cms.untracked.int32(0),
        maxDeltaBXCoPadOdd = cms.untracked.int32(0),
        maxDeltaPadCoPadEven = cms.untracked.int32(2),
        maxDeltaPadCoPadOdd = cms.untracked.int32(3),

        ## efficiency recovery switches
        dropLowQualityCLCTsNoGEMs_ME1a = cms.untracked.bool(False),
        dropLowQualityCLCTsNoGEMs_ME1b = cms.untracked.bool(True),
        buildLCTfromALCTandGEM_ME1a = cms.untracked.bool(True),
        buildLCTfromALCTandGEM_ME1b = cms.untracked.bool(True),
        doLCTGhostBustingWithGEMs = cms.untracked.bool(False),
        correctLCTtimingWithGEM = cms.untracked.bool(False),
        promoteALCTGEMpattern = cms.untracked.bool(True),
        promoteALCTGEMquality = cms.untracked.bool(True),
        
        ## rate reduction 
        doGemMatching = cms.untracked.bool(True),
        gemMatchDeltaEta = cms.untracked.double(0.08),
        gemMatchDeltaBX = cms.untracked.int32(1),
        gemMatchDeltaPhiOdd = cms.untracked.double(dphi_lct_pad98[ptdphi]['odd']),
        gemMatchDeltaPhiEven = cms.untracked.double(dphi_lct_pad98[ptdphi]['even']),
        gemClearNomatchLCTs = cms.untracked.bool(False),

        ## cross BX algorithm
        tmbCrossBxAlgorithm = cms.untracked.uint32(2),
        firstTwoLCTsInChamber = cms.untracked.bool(True),
    )
    return process

## TODO at migration to CMSSW 7X: make all params tracked!    
def customise_L1Emulator2023(process, ptdphi):
    process.simCscTriggerPrimitiveDigis.gemPadProducer =  cms.untracked.InputTag("simMuonGEMCSCPadDigis","")
    ## ME21 has its own SLHC processors
    process.simCscTriggerPrimitiveDigis.alctSLHCME21 = process.simCscTriggerPrimitiveDigis.alctSLHC.clone()
    process.simCscTriggerPrimitiveDigis.clctSLHCME21 = process.simCscTriggerPrimitiveDigis.clctSLHC.clone()
    process.simCscTriggerPrimitiveDigis.alctSLHCME21.alctNplanesHitPattern = 3
    process.simCscTriggerPrimitiveDigis.alctSLHCME21.runME21ILT = cms.untracked.bool(True)
    process.simCscTriggerPrimitiveDigis.clctSLHCME21.clctNplanesHitPattern = 3
    process.simCscTriggerPrimitiveDigis.clctSLHCME21.clctPidThreshPretrig = 2
    process.simCscTriggerPrimitiveDigis.clctParam07.clctPidThreshPretrig = 2
    tmb = process.simCscTriggerPrimitiveDigis.tmbSLHC
    ## GE2/1-ME2/1
    dphi_lct_pad98 = {
        'pt0'  : { 'odd' :  2.00000000 , 'even' :  2.00000000 },
        'pt05' : { 'odd' :  0.02203510 , 'even' :  0.00930056 },
        'pt06' : { 'odd' :  0.01825790 , 'even' :  0.00790009 },
        'pt10' : { 'odd' :  0.01066000 , 'even' :  0.00483286 },
        'pt15' : { 'odd' :  0.00722795 , 'even' :  0.00363230 },
        'pt20' : { 'odd' :  0.00562598 , 'even' :  0.00304879 },
        'pt30' : { 'odd' :  0.00416544 , 'even' :  0.00253782 },
        'pt40' : { 'odd' :  0.00342827 , 'even' :  0.00230833 }
    }
    tmb.me21ILT = cms.untracked.PSet(
        ## run the upgrade algorithm
        runME21ILT = cms.untracked.bool(True),

        ## run in debug mode
        debugLUTs = cms.untracked.bool(False),
        debugMatching = cms.untracked.bool(False),

        ## use old dataformat
        useOldLCTDataFormatALCTGEM = cms.untracked.bool(True),
        
        ## copad construction
        maxDeltaBXInCoPad = cms.untracked.int32(1),
        maxDeltaPadInCoPad = cms.untracked.int32(1),

        ## matching to pads in case LowQ CLCT
        maxDeltaBXPadEven = cms.untracked.int32(1),
        maxDeltaBXPadOdd = cms.untracked.int32(1),
        maxDeltaPadPadEven = cms.untracked.int32(2),
        maxDeltaPadPadOdd = cms.untracked.int32(3),

        ## matching to pads in case absent CLCT
        maxDeltaBXCoPadEven = cms.untracked.int32(0),
        maxDeltaBXCoPadOdd = cms.untracked.int32(0),
        maxDeltaPadCoPadEven = cms.untracked.int32(2),
        maxDeltaPadCoPadOdd = cms.untracked.int32(3),

        ## efficiency recovery switches
        dropLowQualityCLCTsNoGEMs = cms.untracked.bool(True),
        buildLCTfromALCTandGEM = cms.untracked.bool(True),
        doLCTGhostBustingWithGEMs = cms.untracked.bool(False),
        correctLCTtimingWithGEM = cms.untracked.bool(False),
        promoteALCTGEMpattern = cms.untracked.bool(True),
        promoteALCTGEMquality = cms.untracked.bool(True),

        ## rate reduction 
        doGemMatching = cms.untracked.bool(True),
        gemMatchDeltaEta = cms.untracked.double(0.08),
        gemMatchDeltaBX = cms.untracked.int32(1),
        gemMatchDeltaPhiOdd = cms.untracked.double(dphi_lct_pad98[ptdphi]['odd']),
        gemMatchDeltaPhiEven = cms.untracked.double(dphi_lct_pad98[ptdphi]['even']),
        gemClearNomatchLCTs = cms.untracked.bool(False),

        ## cross BX algorithm
        tmbCrossBxAlgorithm = cms.untracked.uint32(2),
        firstTwoLCTsInChamber = cms.untracked.bool(True),
    )
    return process

def customise_DigiToRaw(process):
    return process

def customise_RawToDigi(process):
    return process

def customise_Reco(process):
    process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
    process.muonlocalreco += process.gemRecHits
    process.standAloneMuons.STATrajBuilderParameters.EnableGEMMeasurement = cms.bool(True)
    process.standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableGEMMeasurement = cms.bool(True)
    process.refittedStandAloneMuons.STATrajBuilderParameters.EnableGEMMeasurement = cms.bool(True)
    process.refittedStandAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableGEMMeasurement = cms.bool(True)
    process=outputCustoms(process)
    return process

def customise_DQM(process):
    return process

def customise_Validation(process):
    process.load('Validation.Configuration.gemSimValid_cff')
    process.genvalid_all += process.gemSimValid

    process.load('Validation.RecoMuon.MuonTrackValidator_cfi')
    process.load('SimMuon.MCTruth.MuonAssociatorByHits_cfi')
    process.muonAssociatorByHitsCommonParameters.useGEMs = cms.bool(True)
    process.muonTrackValidator.useGEMs = cms.bool(True)
    return process

def customise_harvesting(process):
    process.load('Validation.Configuration.gemPostValidation_cff')
    process.genHarvesting += process.gemPostValidation
    return process

def outputCustoms(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonGEMDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonGEMCSCPadDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_gemRecHits_*_*')
    return process

    
