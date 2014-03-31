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

def customise_L1Emulator(process):
    process.simCscTriggerPrimitiveDigis.gemPadProducer =  cms.untracked.InputTag("simMuonGEMCSCPadDigis","")
    process.simCscTriggerPrimitiveDigis.clctSLHC.clctPidThreshPretrig = 2
    process.simCscTriggerPrimitiveDigis.clctParam07.clctPidThreshPretrig = 2
    tmb = process.simCscTriggerPrimitiveDigis.tmbSLHC
    tmb.gemMatchDeltaEta = cms.untracked.double(0.08)
    tmb.gemMatchDeltaBX = cms.untracked.int32(1)
    lct_store_gemdphi = True
    if lct_store_gemdphi:
        tmb.gemClearNomatchLCTs = cms.untracked.bool(False)
	tmb.gemMatchDeltaPhiOdd = cms.untracked.double(2.)
        tmb.gemMatchDeltaPhiEven = cms.untracked.double(2.)
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
    process=outputCustoms(process)
    return process

def customise_DQM(process):
    return process

def customise_Validation(process):
    process.load('Validation.Configuration.gemSimValid_cff')
    process.new_globalhitsanalyze = cms.Sequence(process.globalhitsanalyze*process.gemHitsValidation)
    process.new_globaldigisanalyze = cms.Sequence(process.globaldigisanalyze*process.gemDigiValidation)
    process.new_globalrechitsanalyze = cms.Sequence(process.globalrechitsanalyze*process.gemRecHitsValidation)
    process.prevalidation.remove(process.simHitTPAssocProducer)
    process.prevalidation.remove(process.tpSelection)
    process.prevalidation.remove(process.tpSelecForFakeRate)
    process.prevalidation.remove(process.tpSelecForEfficiency)
    process.prevalidation_step = cms.Path(process.prevalidation)
    process.validation.replace(process.globalhitsanalyze, process.new_globalhitsanalyze)
    process.validation.replace(process.globaldigisanalyze, process.new_globaldigisanalyze)
    process.validation.replace(process.globalrechitsanalyze, process.new_globalrechitsanalyze)

    process.validation.remove(process.pixRecHitsValid)
    process.validation.remove(process.stripRecHitsValid)
    process.validation.remove(process.trackingTruthValid)
    process.validation.remove(process.PixelTrackingRecHitsValid)
    process.validation.remove(process.tpClusterProducer)
    process.validation.remove(process.StripTrackingRecHitsValid)
    process.validation.remove(process.trackValidator)
    process.validation.remove(process.NoiseRatesValidation)
    process.validation.remove(process.RecHitsValidation)
    process.validation.remove(process.tpToTkMuonAssociation)
    process.validation.remove(process.tpToTkmuTrackAssociation)
    process.validation.remove(process.tpToStaMuonAssociation)
    process.validation.remove(process.tpToStaUpdMuonAssociation)
    process.validation.remove(process.tpToGlbMuonAssociation)
    process.validation.remove(process.trkProbeTrackVMuonAssoc)
    process.validation.remove(process.trkMuonTrackVTrackAssoc)
    process.validation.remove(process.staMuonTrackVMuonAssoc)
    process.validation.remove(process.staUpdMuonTrackVMuonAssoc)
    process.validation.remove(process.glbMuonTrackVMuonAssoc)  # mix mergedTruth
    process.validation.remove(process.recoMuonVMuAssoc_trk)  # mix mergedTrakTruth
    process.validation.remove(process.recoMuonVMuAssoc_sta)  # mix mergedTrakTruth
    process.validation.remove(process.recoMuonVMuAssoc_glb)  # mix mergedTrakTruth
    process.validation.remove(process.recoMuonVMuAssoc_tgt)  # mix mergedTrakTruth
    process.validation.remove(process.tpToTevFirstMuonAssociation)  # mix mergedTrakTruth
    process.validation.remove(process.tpToTevPickyMuonAssociation)  # mix mergedTrakTruth
    process.validation.remove(process.tpToTevDytMuonAssociation)  # mix mergedTrakTruth
    process.validation.remove(process.tevMuonFirstTrackVMuonAssoc)  # mix mergedTrakTruth
    process.validation.remove(process.tevMuonPickyTrackVMuonAssoc)  # mix mergedTrakTruth
    process.validation.remove(process.tevMuonDytTrackVMuonAssoc)  # mix mergedTrakTruth
    process.validation.remove(process.tpToStaSETMuonAssociation)  # mix mergedTrakTruth
    process.validation.remove(process.tpToStaSETUpdMuonAssociation)  # mix mergedTrakTruth
    process.validation.remove(process.tpToGlbSETMuonAssociation)  # mix mergedTrakTruth
    process.validation.remove(process.staSETMuonTrackVMuonAssoc)  # mix mergedTrakTruth
    process.validation.remove(process.staSETUpdMuonTrackVMuonAssoc)  # mix mergedTrakTruth
    process.validation.remove(process.glbSETMuonTrackVMuonAssoc)  # mix mergedTrakTruth
    process.validation.remove(process.tpToStaRefitMuonAssociation)  # mix mergedTrakTruth
    process.validation.remove(process.tpToStaRefitUpdMuonAssociation)  # mix mergedTrakTruth
    process.validation.remove(process.staRefitMuonTrackVMuonAssoc)  # mix mergedTrakTruth
    process.validation.remove(process.staRefitUpdMuonTrackVMuonAssoc)  # mix mergedTrakTruth
    process.validation.remove(process.trackingParticleRecoTrackAsssociation)  # mix mergedTrakTruth
    process.validation.remove(process.v0Validator)  # mix mergedTrakTruth
    process.validation.remove(process.l2MuonMuTrackV)  # mix mergedTrackTruth
    process.validation.remove(process.l2UpdMuonMuTrackV)  # mix mergedTrackTruth
    process.validation.remove(process.l3TkMuonMuTrackV)  # mix mergedTrackTruth
    process.validation.remove(process.l3MuonMuTrackV)  # mix mergedTrackTruth

    process.validation.remove(process.eleIsoDepositEcalFromHitsFull)  # missing eventsetup
    process.validation.remove(process.eleIsoDepositEcalFromHitsReduced)  # missing eventsetup
    process.validation.remove(process.eleIsoFromDepsEcalFromHitsByCrystalFull03)  # missing eventsetup
    process.validation.remove(process.eleIsoFromDepsEcalFromHitsByCrystalFull04)  # missing eventsetup
    process.validation.remove(process.eleIsoFromDepsEcalFromHitsByCrystalReduced03)  # missing eventsetup
    process.validation.remove(process.eleIsoFromDepsEcalFromHitsByCrystalReduced04)  # missing eventsetup
    process.validation.remove(process.eleIsoFromDepsHcalFromTowers03)  # missing eventsetup
    process.validation.remove(process.eleIsoFromDepsHcalFromTowers04)  # missing eventsetup

    process.validation.remove(process.photonValidation)  # tp_selection
    process.validation.remove(process.oldpfPhotonValidation)  # tp_selection
    process.validation.remove(process.pfPhotonValidation)  # tp_selection
    process.validation.remove(process.tkConversionValidation)  # tp_selection

    process.validation.remove(process.HLTSusyExoVal)  # missing L1GlobalTriggerObjectMapRecord

    process.validation.remove(process.hltHiggsValidator)  # seg fault

    process.validation_step = cms.EndPath( process.validation )
    process.schedule= cms.Schedule()
    for x in process.paths.items() :
       process.schedule += cms.Schedule(x[1])
    for x in process.endpaths.items() :
       process.schedule += cms.Schedule(x[1])
    process.load('Validation.RecoMuon.MuonTrackValidator_cfi')
    process.load('SimMuon.MCTruth.MuonAssociatorByHits_cfi')
    process.muonAssociatorByHitsCommonParameters.useGEMs = cms.bool(True)
    process.muonTrackValidator.useGEMs = cms.bool(True)
    return process


def customise_harvesting(process):
    process.load('Validation.Configuration.gemPostValidation_cff')
    process.postValidation += process.gemPostValidation
    process.postValidation.remove(process.photonPostprocessing)        # seg fault
    process.postValidation.remove(process.pfPhotonPostprocessing)      # seg fault
    process.postValidation.remove(process.oldpfPhotonPostprocessing)   # seg fault
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
