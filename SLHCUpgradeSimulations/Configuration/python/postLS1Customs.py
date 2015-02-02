
import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_PostLS1,customise_csc_hlt
from L1Trigger.L1TCommon.customsPostLS1 import customiseSimL1EmulatorForPostLS1
from SLHCUpgradeSimulations.Configuration.fastSimCustoms import customise_fastSimPostLS1

def customisePostLS1(process):

    # deal with CSC separately:
    process = customise_csc_PostLS1(process)

    # deal with L1 Emulation separately:
    customiseSimL1EmulatorForPostLS1(process)

    # deal with FastSim separately:
    process = customise_fastSimPostLS1(process)

    # all the rest:
    if hasattr(process,'g4SimHits'):
        process=customise_Sim(process)
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'HLTSchedule'):
        process=customise_HLT(process)
    if hasattr(process,'L1simulation_step'):
        process=customise_L1Emulator(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)

    return process

def customiseRun2EraExtras(process):
    """
    This function should be used in addition to the "--era run2" cmsDriver
    option so that it can perform the last few changes that the era hasn't
    implemented yet.
    
    As functionality is added to the run2 era the corresponding line will
    be removed from this function until the whole function is removed.
    
    Currently does exactly the same as "customisePostLS1", since the run2
    era doesn't make any changes yet (coming in later pull requests).
    """
    # deal with CSC separately:
    # a simple sanity check first
    # list of (pathName, expected moduleName) tuples:
    paths_modules = [
        ('digitisation_step', 'simMuonCSCDigis'),
        ('L1simulation_step', 'simCscTriggerPrimitiveDigis'),
        ('L1simulation_step', 'simCsctfTrackDigis'),
        ('raw2digi_step', 'muonCSCDigis'),
        ('raw2digi_step', 'csctfDigis'),
        ('digi2raw_step', 'cscpacker'),
        ('digi2raw_step', 'csctfpacker'),
        ('reconstruction', 'csc2DRecHits'),
        ('dqmoffline_step', 'muonAnalyzer'),
        #('dqmHarvesting', ''),
        ('validation_step', 'relvalMuonBits')
    ]
    # verify:
    for path_name, module_name in paths_modules:
        if hasattr(process, path_name) and not hasattr(process, module_name):
            print "WARNING: module %s is not in %s path!!!" % (module_name, path_name)
            print "         This path has the following modules:"
            print "         ", getattr(process, path_name).moduleNames(),"\n"

    # L1 stub emulator upgrade algorithm
    if hasattr(process, 'simCscTriggerPrimitiveDigis'):
        from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigisPostLS1_cfi import cscTriggerPrimitiveDigisPostLS1
        process.simCscTriggerPrimitiveDigis = cscTriggerPrimitiveDigisPostLS1
        process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'simMuonCSCDigis', 'MuonCSCComparatorDigi')
        process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag( 'simMuonCSCDigis', 'MuonCSCWireDigi')

    # CSCTF that can deal with unganged ME1a
    if hasattr(process, 'simCsctfTrackDigis'):
        from L1Trigger.CSCTrackFinder.csctfTrackDigisUngangedME1a_cfi import csctfTrackDigisUngangedME1a
        process.simCsctfTrackDigis = csctfTrackDigisUngangedME1a
        process.simCsctfTrackDigis.DTproducer = cms.untracked.InputTag("simDtTriggerPrimitiveDigis")
        process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED")

    # deal with L1 Emulation separately:
    # replace the L1 menu from the global tag with one of the following alternatives
    # the menu will be read from an XML file instead of the global tag - must copy the file in luminosityDirectory
    luminosityDirectory = "startup"
    useXmlFile = 'L1Menu_Collisions2015_25ns_v2_L1T_Scales_20141121_Imp0_0x1030.xml'
    process.load('L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi')
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = luminosityDirectory
    process.l1GtTriggerMenuXml.DefXmlFile = useXmlFile

    process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff')
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml')

    #print "INFO:  Customising L1T emulator for 2015 run configuration"
    #print "INFO:  Customize the L1 menu"
    # the following line will break HLT if HLT menu is not updated with the corresponding menu
    #process=customiseL1Menu(process)
    #print "INFO:  loading RCT LUTs"
    #process.load("L1Trigger.L1TCalorimeter.caloStage1RCTLuts_cff")

    process.load("L1Trigger.L1TCommon.l1tDigiToRaw_cfi")
    process.load("L1Trigger.L1TCommon.l1tRawToDigi_cfi")
    process.load("L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi")

    process.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
    process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')

    if hasattr(process, 'simGtDigis'):
        process.simGtDigis.GmtInputTag = 'simGmtDigis'
        process.simGtDigis.GctInputTag = 'simCaloStage1LegacyFormatDigis'
        process.simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )
    if hasattr(process, 'gctDigiToRaw'):
        process.gctDigiToRaw.gctInputLabel = 'simCaloStage1LegacyFormatDigis'

    if hasattr(process, 'simGctDigis'):
        for sequence in process.sequences:
            #print "INFO:  checking sequence ", sequence
            #print "BEFORE:  ", getattr(process,sequence)
            getattr(process,sequence).replace(process.simGctDigis,process.L1TCaloStage1)
            #print "AFTER:  ", getattr(process,sequence)
        for path in process.paths:
            #print "INFO:  checking path ", path
            #print "BEFORE:  ", getattr(process,path)
            getattr(process,path).replace(process.simGctDigis,process.L1TCaloStage1)
            #print "AFTER:  ", getattr(process,path)

    if hasattr(process, 'DigiToRaw'):
        #print "INFO:  customizing DigiToRaw for Stage 1"
        #print process.DigiToRaw
        process.l1tDigiToRaw.InputLabel = cms.InputTag("simCaloStage1FinalDigis", "")
        process.l1tDigiToRaw.TauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "rlxTaus")
        process.l1tDigiToRaw.IsoTauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "isoTaus")
        process.l1tDigiToRaw.HFBitCountsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFBitCounts")
        process.l1tDigiToRaw.HFRingSumsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFRingSums")
        process.l1tDigiToRawSeq = cms.Sequence(process.gctDigiToRaw + process.l1tDigiToRaw);
        process.DigiToRaw.replace(process.gctDigiToRaw, process.l1tDigiToRawSeq)
        #print process.DigiToRaw
        if hasattr(process, 'rawDataCollector'):
            #print "INFO:  customizing rawDataCollector for Stage 1"
            process.rawDataCollector.RawCollectionList.append(cms.InputTag("l1tDigiToRaw"))
    if hasattr(process, 'RawToDigi'):
        #print "INFO:  customizing L1RawToDigi for Stage 1"
        #print process.RawToDigi
        process.L1RawToDigiSeq = cms.Sequence(process.gctDigis+process.caloStage1Digis+process.caloStage1LegacyFormatDigis)
        process.RawToDigi.replace(process.gctDigis, process.L1RawToDigiSeq)
        #print process.RawToDigi

    if hasattr(process, 'HLTL1UnpackerSequence'):
        #print "INFO: customizing HLTL1UnpackerSequence for Stage 1"
        #print process.HLTL1UnpackerSequence

        # extend sequence to add Layer 1 unpacking and conversion back to legacy format
        process.hltCaloStage1Digis = process.caloStage1Digis.clone()
        process.hltCaloStage1LegacyFormatDigis = process.caloStage1LegacyFormatDigis.clone()
        process.hltCaloStage1LegacyFormatDigis.InputCollection = cms.InputTag("hltCaloStage1Digis")
        process.hltCaloStage1LegacyFormatDigis.InputRlxTauCollection = cms.InputTag("hltCaloStage1Digis:rlxTaus")
        process.hltCaloStage1LegacyFormatDigis.InputIsoTauCollection = cms.InputTag("hltCaloStage1Digis:isoTaus")
        process.hltCaloStage1LegacyFormatDigis.InputHFSumsCollection = cms.InputTag("hltCaloStage1Digis:HFRingSums")
        process.hltCaloStage1LegacyFormatDigis.InputHFCountsCollection = cms.InputTag("hltCaloStage1Digis:HFBitCounts")
        #process.hltL1RawToDigiSeq = cms.Sequence(process.hltGctDigis+process.hltCaloStage1 + process.hltCaloStage1LegacyFormatDigis)
        process.hltL1RawToDigiSeq = cms.Sequence(process.hltCaloStage1Digis + process.hltCaloStage1LegacyFormatDigis)
        process.HLTL1UnpackerSequence.replace(process.hltGctDigis, process.hltL1RawToDigiSeq)

    alist=['hltL1GtObjectMap']
    for a in alist:
        #print "INFO: checking for", a, "in process."
        if hasattr(process,a):
            #print "INFO: customizing ", a, "to use new calo Stage 1 digis converted to legacy format"
            getattr(process, a).GctInputTag = cms.InputTag("hltCaloStage1LegacyFormatDigis")

    alist=['hltL1extraParticles']
    for a in alist:
        #print "INFO: checking for", a, "in process."
        if hasattr(process,a):
            #print "INFO:  customizing ", a, "to use new calo Stage 1 digis converted to legacy format"
            getattr(process, a).etTotalSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).nonIsolatedEmSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","nonIsoEm")
            getattr(process, a).etMissSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).htMissSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).forwardJetSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","forJets")
            getattr(process, a).centralJetSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","cenJets")
            getattr(process, a).tauJetSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","tauJets")
            getattr(process, a).isoTauJetSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","isoTauJets")
            getattr(process, a).isolatedEmSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","isoEm")
            getattr(process, a).etHadSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).hfRingEtSumsSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).hfRingBitCountsSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")

    blist=['l1extraParticles','recoL1extraParticles','dqmL1ExtraParticles']
    for b in blist:
        #print "INFO: checking for", b, "in process."
        if hasattr(process,b):
            #print "BEFORE:  ", getattr(process, b).centralJetSource
            if (getattr(process, b).centralJetSource == cms.InputTag("simGctDigis","cenJets")):
                getattr(process, b).etTotalSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).nonIsolatedEmSource = cms.InputTag("simCaloStage1LegacyFormatDigis","nonIsoEm")
                getattr(process, b).etMissSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).htMissSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).forwardJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","forJets")
                getattr(process, b).centralJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","cenJets")
                getattr(process, b).tauJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","tauJets")
                getattr(process, b).isoTauJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","isoTauJets")
                getattr(process, b).isolatedEmSource = cms.InputTag("simCaloStage1LegacyFormatDigis","isoEm")
                getattr(process, b).etHadSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).hfRingEtSumsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).hfRingBitCountsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
            else:
                #print "INFO:  customizing ", b, "to use new calo Stage 1 digis converted to legacy format"
                getattr(process, b).etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis")
                getattr(process, b).nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm")
                getattr(process, b).etMissSource = cms.InputTag("caloStage1LegacyFormatDigis")
                getattr(process, b).htMissSource = cms.InputTag("caloStage1LegacyFormatDigis")
                getattr(process, b).forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets")
                getattr(process, b).centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets")
                getattr(process, b).tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets")
                getattr(process, b).isoTauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets")
                getattr(process, b).isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm")
                getattr(process, b).etHadSource = cms.InputTag("caloStage1LegacyFormatDigis")
                getattr(process, b).hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis")
                getattr(process, b).hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis")
            #print "AFTER:  ", getattr(process, b).centralJetSource

    # deal with FastSim separately:
    if hasattr(process,'famosSimHits'):
        # enable 2015 HF shower library
        process.famosSimHits.Calorimetry.HFShowerLibrary.useShowerLibrary = True
    
        # change default parameters
        process.famosSimHits.ParticleFilter.pTMin  = 0.1
        process.famosSimHits.TrackerSimHits.pTmin  = 0.1
        process.famosSimHits.ParticleFilter.etaMax = 5.300
       
    from FastSimulation.PileUpProducer.PileUpFiles_cff import fileNames_13TeV
    process.genMixPileUpFiles = cms.PSet(fileNames = fileNames_13TeV)
    if hasattr(process,'famosPileUp'):
        if hasattr(process.famosPileUp,"PileUpSimulator"):
            process.famosPileUp.PileUpSimulator.fileNames = fileNames_13TeV

    # all the rest:
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HFShowerLibrary.FileName = 'SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v3.root'
    if hasattr(process,'reconstruction'):
        #lowering HO threshold with SiPM
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

    if hasattr(process,'digitisation_step'):
        alist=['RAWSIM','RAWDEBUG','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT','PREMIX','PREMIXRAW']
        for a in alist:
            b=a+'output'
            if hasattr(process,b):
                getattr(process,b).outputCommands.append('keep *_simMuonCSCDigis_*_*')
                getattr(process,b).outputCommands.append('keep *_simMuonRPCDigis_*_*')
                getattr(process,b).outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*')
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
            if hasattr(process.mix.digitizers,'pixel'):
                # DynamicInefficency - 13TeV - 50ns case
                if process.mix.bunchspace == 50:
                    process.mix.digitizers.pixel.theInstLumiScaleFactor = cms.double(246.4)
                    process.mix.digitizers.pixel.theLadderEfficiency_BPix1 = cms.vdouble( [0.979259,0.976677]*10 )
                    process.mix.digitizers.pixel.theLadderEfficiency_BPix2 = cms.vdouble( [0.994321,0.993944]*16 )
                    process.mix.digitizers.pixel.theLadderEfficiency_BPix3 = cms.vdouble( [0.996787,0.996945]*22 )
                # DynamicInefficency - 13TeV - 25ns case
                if process.mix.bunchspace == 25:
                    process.mix.digitizers.pixel.theInstLumiScaleFactor = cms.double(364)
                    process.mix.digitizers.pixel.theLadderEfficiency_BPix1 = cms.vdouble( [1]*20 )
                    process.mix.digitizers.pixel.theLadderEfficiency_BPix2 = cms.vdouble( [1]*32 )
                    process.mix.digitizers.pixel.theLadderEfficiency_BPix3 = cms.vdouble( [1]*44 )
                    process.mix.digitizers.pixel.theModuleEfficiency_BPix1 = cms.vdouble( 1, 1, 1, 1, )
                    process.mix.digitizers.pixel.theModuleEfficiency_BPix2 = cms.vdouble( 1, 1, 1, 1, )
                    process.mix.digitizers.pixel.theModuleEfficiency_BPix3 = cms.vdouble( 1, 1, 1, 1 )
                    process.mix.digitizers.pixel.thePUEfficiency_BPix1 = cms.vdouble( 1.00023, -3.18350e-06, 5.08503e-10, -6.79785e-14 )
                    process.mix.digitizers.pixel.thePUEfficiency_BPix2 = cms.vdouble( 9.99974e-01, -8.91313e-07, 5.29196e-12, -2.28725e-15 )
                    process.mix.digitizers.pixel.thePUEfficiency_BPix3 = cms.vdouble( 1.00005, -6.59249e-07, 2.75277e-11, -1.62683e-15 )
    if hasattr(process,'HLTSchedule'):
        process.hltCsc2DRecHits.readBadChannels = cms.bool(False)
        process.hltCsc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)
        if hasattr(process,"CSCIndexerESProducer"):
            process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
        if hasattr(process,"CSCChannelMapperESProducer"):
            process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")
    if hasattr(process,'dqmoffline_step'):
        process.l1tCsctf.gangedME11a = cms.untracked.bool(False)

    return process
    

def digiEventContent(process):
    #extend the event content

    alist=['RAWSIM','RAWDEBUG','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT','PREMIX','PREMIXRAW']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonCSCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonRPCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*')

    return process


def customise_DQM(process):
    #process.dqmoffline_step.remove(process.jetMETAnalyzer)
    # Turn off flag of gangedME11a
    process.l1tCsctf.gangedME11a = cms.untracked.bool(False)
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


def customise_Digi(process):
    process=digiEventContent(process)
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
        if hasattr(process.mix.digitizers,'pixel'):
            # DynamicInefficency - 13TeV - 50ns case
            if process.mix.bunchspace == 50:
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
            # DynamicInefficency - 13TeV - 25ns case
            if process.mix.bunchspace == 25:
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
    return process


def customise_L1Emulator(process):
    return process


def customise_RawToDigi(process):
    return process


def customise_DigiToRaw(process):
    return process


def customise_HLT(process):
    process=customise_csc_hlt(process)
    return process


def customise_Reco(process):
    #lowering HO threshold with SiPM
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

    return process


def customise_harvesting(process):
    #process.dqmHarvesting.remove(process.dataCertificationJetMET)
    #process.dqmHarvesting.remove(process.sipixelEDAClient)
    #process.dqmHarvesting.remove(process.sipixelCertification)
    return (process)        

def recoOutputCustoms(process):

    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonCSCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonRPCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_rawDataCollector_*_*')
    return process

