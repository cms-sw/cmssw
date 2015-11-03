import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.customiseForRunI import customiseForRunI

def customisePostLS1_Common(process):

    # deal with L1 Emulation separately
    from L1Trigger.L1TCommon.customsPostLS1 import customiseSimL1EmulatorForStage1
    process = customiseSimL1EmulatorForStage1(process)

    # deal with CSC separately
    from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_PostLS1
    process = customise_csc_PostLS1(process)

    # deal with FastSim separately
    from SLHCUpgradeSimulations.Configuration.fastSimCustoms import customise_fastSimPostLS1
    process = customise_fastSimPostLS1(process)

    # all the rest
    if hasattr(process,'g4SimHits'):
        process = customise_Sim(process)
    if hasattr(process,'DigiToRaw'):
        process = customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process = customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process = customise_Reco(process)
    if hasattr(process,'digitisation_step') or ( hasattr(process,'mix') and hasattr(process.mix,'digitizers')):
        process = customise_Digi_Common(process)
    if hasattr(process,'HLTSchedule'):
        process = customise_HLT(process)
    if hasattr(process,'L1simulation_step'):
        process = customise_L1Emulator(process)
    if hasattr(process,'dqmoffline_step'):
        process = customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process = customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process = customise_Validation(process)
    if hasattr(process,'datamixing_step'):
        process = customise_DataMix(process)

    return process


def customisePostLS1(process):

    # common customisation
    process = customisePostLS1_Common(process)

    # 25ns specific customisation
    if hasattr(process,'digitisation_step'):
        process = customise_Digi_25ns(process)
    if hasattr(process,'dqmoffline_step'):
        process = customise_DQM_25ns(process)

    return process


def customisePostLS1_lowPU(process):

    # common customisations
    process = customisePostLS1_Common(process)

    # 50ns specific customisation
    if hasattr(process,'digitisation_step'):
        process = customise_Digi_50ns(process)

    return process


def customisePostLS1_50ns(process):

    # common customisations
    process = customisePostLS1_Common(process)

    # 50ns specific customisation
    if hasattr(process,'digitisation_step'):
        process = customise_Digi_50ns(process)

    return process


def customisePostLS1_HI(process):

    # common customisation
    process = customisePostLS1_Common(process)

    # HI Specific additional customizations:
    from L1Trigger.L1TCommon.customsPostLS1 import customiseSimL1EmulatorForPostLS1_Additional_HI
    process = customiseSimL1EmulatorForPostLS1_Additional_HI(process)

    # HI L1Menu:
    #from L1Trigger.Configuration.customise_overwriteL1Menu import L1Menu_CollisionsHeavyIons2015_v0
    #process = L1Menu_CollisionsHeavyIons2015_v0(process)

    return process

def customisePostLS1_B0T(process):
    # 50ns only

    process=customisePostLS1_50ns(process)
    process=customiseForRunI(process)

    return process

def customisePostLS1_B0T_lowPU(process):
    # 50ns only

    process=customisePostLS1_lowPU(process)
    process=customiseForRunI(process)

    return process


def digiEventContent(process):
    #extend the event content

    alist=['RAWDEBUG','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b = a + 'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonCSCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonRPCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*')

    return process


def customise_DQM(process):
    #process.dqmoffline_step.remove(process.jetMETAnalyzer)
    # Turn off flag of gangedME11a
    process.l1tCsctf.gangedME11a = cms.untracked.bool(False)
    # Turn off "low bias voltage" region in HCAL noise filters
    if hasattr(process,'HBHENoiseFilterResultProducer'):
        process.HBHENoiseFilterResultProducer.IgnoreTS4TS5ifJetInLowBVRegion = cms.bool(False)
    return process


def customise_DQM_25ns(process):
    # Switch the default decision of the HCAL noise filter
    if hasattr(process,'HBHENoiseFilterResultProducer'):
        process.HBHENoiseFilterResultProducer.defaultDecision = cms.string("HBHENoiseFilterResultRun2Loose")
    return process


def customise_Validation(process):
    #process.validation_step.remove(process.PixelTrackingRecHitsValid)
    # We don't run the HLT
    #process.validation_step.remove(process.HLTSusyExoVal)
    #process.validation_step.remove(process.hltHiggsValidator)
    return process


def customise_Sim(process):
    # enable 2015 HF shower library
    process.g4SimHits.HFShowerLibrary.FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v3.root'
    return process

def customise_New_HCAL(process):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers'):
        if hasattr(process.mix.digitizers,'hcal'):
            process.mix.digitizers.hcal.minFCToDelay=5.
        if hasattr(process.mix.digitizers,'hcal') and hasattr(process.mix.digitizers.hcal,'hf1'):
            process.mix.digitizers.hcal.hf1.samplingFactor = cms.double(0.67)
        if hasattr(process.mix.digitizers,'hcal') and hasattr(process.mix.digitizers.hcal,'hf2'):
            process.mix.digitizers.hcal.hf2.samplingFactor = cms.double(0.67)

    if hasattr(process,'mixData'):
#        if hasattr(process.mixData.digitizers,'hcal'):
#            process.mixData.digitizers.hcal.minFCToDelay=5.
        if hasattr(process.mixData,'hf1'):
            process.mixData.hf1.samplingFactor = cms.double(0.67)
        if hasattr(process.mixData,'hf2'):
            process.mixData.hf2.samplingFactor = cms.double(0.67)

    if hasattr(process,'hltHbhereco'):
        process.hltHbhereco.timeSlewPars = cms.vdouble( 12.2999, -2.19142, 0, 12.2999, -2.19142, 0, 12.2999, -2.19142, 0 )
        process.hltHbhereco.respCorrM3   = cms.double( 0.95 )

    return process
        

def customise_Digi_Common(process):
    process = digiEventContent(process)
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers'):
        if hasattr(process.mix.digitizers,'hcal') and hasattr(process.mix.digitizers.hcal,'ho'):
            process.mix.digitizers.hcal.ho.photoelectronsToAnalog = cms.vdouble([4.0]*16)
            process.mix.digitizers.hcal.ho.siPMCode = cms.int32(1)
            process.mix.digitizers.hcal.ho.pixels = cms.int32(2500)
            process.mix.digitizers.hcal.ho.doSiPMSmearing = cms.bool(False)
        if hasattr(process.mix.digitizers,'hcal') and hasattr(process.mix.digitizers.hcal,'hf1'):
            process.mix.digitizers.hcal.hf1.samplingFactor = cms.double(0.60)
        if hasattr(process.mix.digitizers,'hcal') and hasattr(process.mix.digitizers.hcal,'hf2'):
            process.mix.digitizers.hcal.hf2.samplingFactor = cms.double(0.60)
    return process

def customise_DataMix(process):
    if hasattr(process,'mixData'):
        if hasattr(process.mixData,'ho'):
            process.mixData.ho.photoelectronsToAnalog = cms.vdouble([4.0]*16)
            process.mixData.ho.siPMCode = cms.int32(1)
            process.mixData.ho.pixels = cms.int32(2500)
            process.mixData.ho.doSiPMSmearing = cms.bool(False)
        if hasattr(process.mixData,'hf1'):
            process.mixData.hf1.samplingFactor = cms.double(0.60)
        if hasattr(process.mixData,'hf2'):
            process.mixData.hf2.samplingFactor = cms.double(0.60)
    return process


def customise_Digi_50ns(process):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers'):
        if hasattr(process.mix.digitizers,'pixel'):
            # pixel dynamic inefficency - 13TeV - 50ns case
            process.mix.digitizers.pixel.theInstLumiScaleFactor = cms.double(246.4)
            process.mix.digitizers.pixel.theLadderEfficiency_BPix1 = cms.vdouble(
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
            )
            process.mix.digitizers.pixel.theLadderEfficiency_BPix2 = cms.vdouble(
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
            )
            process.mix.digitizers.pixel.theLadderEfficiency_BPix3 = cms.vdouble(
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
            )
    if hasattr(process,'mixData'):
            # pixel dynamic inefficency - 13TeV - 50ns case
        process.mixData.theInstLumiScaleFactor = cms.double(246.4)
        process.mixData.theLadderEfficiency_BPix1 = cms.vdouble(
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
                0.979259,
                0.976677,
            )
        process.mixData.theLadderEfficiency_BPix2 = cms.vdouble(
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
                0.994321,
                0.993944,
            )
        process.mixData.theLadderEfficiency_BPix3 = cms.vdouble(
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
                0.996787,
                0.996945,
            )
    return process


def customise_Digi_25ns(process):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers'):
        if hasattr(process.mix.digitizers,'pixel'):
            # pixel dynamic inefficency - 13TeV - 25ns case
            process.mix.digitizers.pixel.theInstLumiScaleFactor = cms.double(364)
            process.mix.digitizers.pixel.theLadderEfficiency_BPix1 = cms.vdouble(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            )
            process.mix.digitizers.pixel.theLadderEfficiency_BPix2 = cms.vdouble(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            )
            process.mix.digitizers.pixel.theLadderEfficiency_BPix3 = cms.vdouble(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            )
            process.mix.digitizers.pixel.theModuleEfficiency_BPix1 = cms.vdouble(
                1,
                1,
                1,
                1,
            )
            process.mix.digitizers.pixel.theModuleEfficiency_BPix2 = cms.vdouble(
                1,
                1,
                1,
                1,
                )
            process.mix.digitizers.pixel.theModuleEfficiency_BPix3 = cms.vdouble(
                1,
                1,
                1,
                1,
            )
            process.mix.digitizers.pixel.thePUEfficiency_BPix1 = cms.vdouble(
                 1.00023,
                -3.18350e-06,
                 5.08503e-10,
                -6.79785e-14,
            )
            process.mix.digitizers.pixel.thePUEfficiency_BPix2 = cms.vdouble(
                 9.99974e-01,
                -8.91313e-07,
                 5.29196e-12,
                -2.28725e-15,
            )
            process.mix.digitizers.pixel.thePUEfficiency_BPix3 = cms.vdouble(
                 1.00005,
                -6.59249e-07,
                 2.75277e-11,
                -1.62683e-15,
            )
    if hasattr(process,'mixData'):
            # pixel dynamic inefficency - 13TeV - 25ns case
        process.mixData.theInstLumiScaleFactor = cms.double(364)
        process.mixData.theLadderEfficiency_BPix1 = cms.vdouble(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            )
        process.mixData.theLadderEfficiency_BPix2 = cms.vdouble(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            )
        process.mixData.theLadderEfficiency_BPix3 = cms.vdouble(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            )
        process.mixData.theModuleEfficiency_BPix1 = cms.vdouble(
                1,
                1,
                1,
                1,
            )
        process.mixData.theModuleEfficiency_BPix2 = cms.vdouble(
                1,
                1,
                1,
                1,
                )
        process.mixData.theModuleEfficiency_BPix3 = cms.vdouble(
                1,
                1,
                1,
                1,
            )
        process.mixData.thePUEfficiency_BPix1 = cms.vdouble(
                 1.00023,
                -3.18350e-06,
                 5.08503e-10,
                -6.79785e-14,
            )
        process.mixData.thePUEfficiency_BPix2 = cms.vdouble(
                 9.99974e-01,
                -8.91313e-07,
                 5.29196e-12,
                -2.28725e-15,
            )
        process.mixData.thePUEfficiency_BPix3 = cms.vdouble(
                 1.00005,
                -6.59249e-07,
                 2.75277e-11,
                -1.62683e-15,
            )
    return process


def customise_L1Emulator(process):
    return process


def customise_RawToDigi(process):
    process.RawToDigi.remove(process.gtEvmDigis)
    return process


def customise_DigiToRaw(process):
    return process


def customise_HLT(process):
    return process


def customise_Reco(process):
    #lowering HO threshold with SiPM
    if hasattr(process,'particleFlowRecHitHO'):
        for prod in process.particleFlowRecHitHO.producers:
            prod.qualityTests = cms.VPSet(
                cms.PSet(
                    name = cms.string("PFRecHitQTestThreshold"),
                    threshold = cms.double(0.05) # new threshold for SiPM HO
                    ),
                cms.PSet(
                    name = cms.string("PFRecHitQTestHCALChannel"),
                    maxSeverities      = cms.vint32(11),
                    cleaningThresholds = cms.vdouble(0.0),
                    flags              = cms.vstring('Standard')
                    )
                )

    #Lower Thresholds also for Clusters!!!    

        for p in process.particleFlowClusterHO.seedFinder.thresholdsByDetector:
            p.seedingThreshold = cms.double(0.08)

        for p in process.particleFlowClusterHO.initialClusteringStep.thresholdsByDetector:
            p.gatheringThreshold = cms.double(0.05)

        for p in process.particleFlowClusterHO.pfClusterBuilder.recHitEnergyNorms:
            p.recHitEnergyNorm = cms.double(0.05)

        process.particleFlowClusterHO.pfClusterBuilder.positionCalc.logWeightDenominator = cms.double(0.05)
        process.particleFlowClusterHO.pfClusterBuilder.allCellsPositionCalc.logWeightDenominator = cms.double(0.05)

        from RecoHI.Configuration.customise_PPwithHI import addHIIsolationProducer
        process = addHIIsolationProducer(process)

    return process


def customise_harvesting(process):
    #process.dqmHarvesting.remove(process.dataCertificationJetMET)
    #process.dqmHarvesting.remove(process.sipixelEDAClient)
    #process.dqmHarvesting.remove(process.sipixelCertification)
    return (process)        


def recoOutputCustoms(process):

    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b = a + 'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonCSCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonRPCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_rawDataCollector_*_*')
    return process
