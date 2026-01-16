import FWCore.ParameterSet.Config as cms

# handle normal mixing or premixing
def getHcalDigitizer(process):
    if hasattr(process,'mixData'):
        return process.mixData
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        return process.mix.digitizers.hcal
    return None

def getHGCalDigitizer(process,section):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers'):
        if section == 'EE' and hasattr(process.mix.digitizers,'hgceeDigitizer'):
            return process.mix.digitizers.hgceeDigitizer
        elif section == 'FH' and hasattr(process.mix.digitizers,'hgchefrontDigitizer'):
            return process.mix.digitizers.hgchefrontDigitizer
        elif section == 'BH' and hasattr(process.mix.digitizers,'hgchebackDigitizer'):
            return process.mix.digitizers.hgchebackDigitizer
        elif section == 'HFNose' and hasattr(process.mix.digitizers,'hfnoseDigitizer'):
            return process.mix.digitizers.hfnoseDigitizer
    return None

# change assumptions about lumi rate
def setScenarioHLLHC(module,scenarioHLLHC):
    if scenarioHLLHC=="nominal":
        from CalibCalorimetry.HcalPlugins.HBHEDarkening_cff import _years_LHC, _years_HLLHC_nominal
        module.years = _years_LHC + _years_HLLHC_nominal
    elif scenarioHLLHC=="ultimate":
        from CalibCalorimetry.HcalPlugins.HBHEDarkening_cff import _years_LHC, _years_HLLHC_ultimate
        module.years = _years_LHC + _years_HLLHC_ultimate
    return module

# turnon = True enables default, False disables
# recalibration and darkening always together
def ageHB(process,turnon,scenarioHLLHC):
    if turnon:
        from CalibCalorimetry.HcalPlugins.HBHEDarkening_cff import HBDarkeningEP
        process.HBDarkeningEP = HBDarkeningEP
        process.HBDarkeningEP = setScenarioHLLHC(process.HBDarkeningEP,scenarioHLLHC)
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HBDarkening = cms.bool(turnon)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HBRecalibration = cms.bool(turnon)
    return process

def ageHE(process,turnon,scenarioHLLHC):
    if turnon:
        from CalibCalorimetry.HcalPlugins.HBHEDarkening_cff import HEDarkeningEP
        process.HEDarkeningEP = HEDarkeningEP
        process.HEDarkeningEP = setScenarioHLLHC(process.HEDarkeningEP,scenarioHLLHC)
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HEDarkening = cms.bool(turnon)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HERecalibration = cms.bool(turnon)
    return process

def ageHF(process,turnon):
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HFDarkening = cms.bool(turnon)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HFRecalibration = cms.bool(turnon)
    return process

def agedHFNose(process,algo=0):
    from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import HFNose_setEndOfLifeNoise
    process = HFNose_setEndOfLifeNoise(process,byDose=True,byDoseAlgo=algo)
    return process

def agedHGCal(process,algo=0):
    from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import HGCal_setEndOfLifeNoise
    process = HGCal_setEndOfLifeNoise(process,byDose=True,byDoseAlgo=algo)
    return process

def realisticHGCalStartup(process):
    from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import HGCal_setRealisticStartupNoise
    process = HGCal_setRealisticStartupNoise(process)
    return process

# needs lumi to set proper ZS thresholds (tbd)
def ageSiPM(process,turnon,lumi):
    process.es_hardcode.hbUpgrade.doRadiationDamage = turnon
    process.es_hardcode.heUpgrade.doRadiationDamage = turnon

    # todo: determine ZS threshold adjustments

    # adjust PF thresholds for increased noise
    # based on: https://baylor.box.com/s/w32ja75krcbxcycyifexu28dwlgrj7wg
    hcal_lumis = [300, 1000, 3000, 4500, 1e10]
    hcal_thresholds = {
        300: {
            "seed": [0.5, 0.625, 0.75, 0.75],
            "rec": [0.4, 0.5, 0.6, 0.6],
        },
        1000: {
            "seed": [1.0, 1.5, 1.5, 1.5],
            "rec": [0.8, 1.2, 1.2, 1.2],
        },
        3000: {
            "seed": [1.25, 2.5, 2.5, 2.5],
            "rec": [1.0, 2.0, 2.0, 2.0],
        },
        4500: {
            "seed": [1.5, 3.0, 3.0, 3.0],
            "rec": [1.25, 2.5, 2.5, 2.5],
        },
    }
    ctmodules = ['calotowermaker','caloTowerForTrk','caloTowerForTrkPreSplitting','towerMaker','towerMakerWithHO','hltPhase2TowerMakerForAll',' hltTowerMaker']
    for ilumi, hcal_lumi in enumerate(hcal_lumis[:-1]):
        if lumi >= hcal_lumi and lumi < hcal_lumis[ilumi+1]:
            pfclushbhemodules = ['particleFlowClusterHBHE','hltParticleFlowClusterHBHE']
            for pfhbhemod in pfclushbhemodules:
                if hasattr(process,pfhbhemod):
                    getattr(process,pfhbhemod).seedFinder.thresholdsByDetector[0].seedingThreshold              = hcal_thresholds[hcal_lumi]["seed"]
                    getattr(process,pfhbhemod).initialClusteringStep.thresholdsByDetector[0].gatheringThreshold = hcal_thresholds[hcal_lumi]["rec"]
                    getattr(process,pfhbhemod).pfClusterBuilder.recHitEnergyNorms[0].recHitEnergyNorm           = hcal_thresholds[hcal_lumi]["rec"]
                    getattr(process,pfhbhemod).pfClusterBuilder.positionCalc.logWeightDenominatorByDetector[0].logWeightDenominator = hcal_thresholds[hcal_lumi]["rec"]
                    getattr(process,pfhbhemod).pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[0].logWeightDenominator = hcal_thresholds[hcal_lumi]["rec"]
            pfclushcalmodules = ['particleFlowClusterHCAL','hltParticleFlowClusterHCAL']
            for pfhcalmod in  pfclushcalmodules:
                if hasattr(process,pfhcalmod):
                    getattr(process,pfhcalmod).pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[0].logWeightDenominator = hcal_thresholds[hcal_lumi]["rec"]
            pfrechithbhemodules = ['particleFlowRecHitHBHE', 'hltParticleFlowRecHitHBHE']
            for pfrechitmod in pfrechithbhemodules:
                if hasattr(process,pfrechitmod):
                    getattr(process,pfrechitmod).producers[0].qualityTests[0].cuts[0].threshold = hcal_thresholds[hcal_lumi]["rec"]
            for ctmod in ctmodules:
                if hasattr(process,ctmod):
                    getattr(process,ctmod).HBThreshold1 = hcal_thresholds[hcal_lumi]["rec"][0]
                    getattr(process,ctmod).HBThreshold2 = hcal_thresholds[hcal_lumi]["rec"][1]
                    getattr(process,ctmod).HBThreshold = hcal_thresholds[hcal_lumi]["rec"][-1]
            break

    return process

def ageHcal(process,lumi,instLumi,scenarioHLLHC):
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.DelivLuminosity = cms.double(float(lumi))  # integrated lumi in fb-1

    # these lines need to be further activated by turning on 'complete' aging for HF
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.InstLuminosity = cms.double(float(instLumi))
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(float(lumi))

    # recalibration and darkening always together
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.iLumi = cms.double(float(lumi))

    # functions to enable individual subdet aging
    process = ageHB(process,True,scenarioHLLHC)
    process = ageHE(process,True,scenarioHLLHC)
    process = ageHF(process,True)
    process = ageSiPM(process,True,lumi)

    return process

def turn_on_HB_aging(process):
    process = ageHB(process,True,"")
    return process

def turn_off_HB_aging(process):
    process = ageHB(process,False,"")
    return process

def turn_on_HE_aging(process):
    process = ageHE(process,True,"")
    return process

def turn_off_HE_aging(process):
    process = ageHE(process,False,"")
    return process

def turn_on_HF_aging(process):
    process = ageHF(process,True)
    return process

def turn_off_HF_aging(process):
    process = ageHF(process,False)
    return process

def turn_off_SiPM_aging(process):
    process = ageSiPM(process,False,0.0)
    return process

def hf_complete_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.HFDarkening = cms.untracked.bool(True)
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HFDarkening = cms.untracked.bool(False)
    return process

def ageEcal(process,lumi,instLumi):
    if hasattr(process,'g4SimHits'):
        #these lines need to be further activiated by tuning on 'complete' aging for ecal
        process.g4SimHits.ECalSD.InstLuminosity = cms.double(instLumi)
        process.g4SimHits.ECalSD.DelivLuminosity = cms.double(float(lumi))

   # available conditions
    ecal_lumis = [300,1000,3000,4500]
    ecal_conditions = [
        ['EcalIntercalibConstantsRcd','EcalIntercalibConstants_TL{:d}_upgrade_8deg_v2_mc'],
        ['EcalIntercalibConstantsMCRcd','EcalIntercalibConstantsMC_TL{:d}_upgrade_8deg_v2_mc'],
        ['EcalLaserAPDPNRatiosRcd','EcalLaserAPDPNRatios_TL{:d}_upgrade_8deg_mc'],
        ['EcalPedestalsRcd','EcalPedestals_TL{:d}_upgradeTIA_8deg_mc'],
        ['EcalTPGLinearizationConstRcd','EcalTPGLinearizationConst_TL{:d}_upgrade_8deg_mc'],
    ]

    # update PF thresholds, based on https://indico.cern.ch/event/653123/contributions/2659235/attachments/1491385/2318364/170711_upsg_ledovskoy.pdf
    ecal_thresholds = {
        300 : 0.103,
        1000 : 0.175,
        3000 : 0.435,
        4500 : 0.707,
    }
    ecal_seed_multiplier = 2.5

    # try to get conditions
    if int(lumi) in ecal_lumis:
        if not hasattr(process.GlobalTag,'toGet'):
            process.GlobalTag.toGet=cms.VPSet()
        for ecal_condition in ecal_conditions:
            process.GlobalTag.toGet.append(cms.PSet(
                record = cms.string(ecal_condition[0]),
                tag = cms.string(ecal_condition[1].format(int(lumi))),
                connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                )
            )
        pfclusecaluncorrmodules = ['particleFlowClusterECALUncorrected','hltParticleFlowClusterECALUncorrected','hltParticleFlowClusterECALUncorrectedL1Seeded',' hltParticleFlowClusterECALUncorrectedUnseeded']
        for pfclusecaluncorrmod in pfclusecaluncorrmodules:
            if hasattr(process,pfclusecaluncorrmod):
                _seeds = getattr(process,pfclusecaluncorrmod).seedFinder.thresholdsByDetector
                for iseed in range(0,len(_seeds)):
                    if _seeds[iseed].detector.value()=="ECAL_BARREL":
                        _seeds[iseed].seedingThreshold = cms.double(ecal_thresholds[int(lumi)]*ecal_seed_multiplier)
                _clusters = getattr(process,pfclusecaluncorrmod).initialClusteringStep.thresholdsByDetector
                for icluster in range(0,len(_clusters)):
                    if _clusters[icluster].detector.value()=="ECAL_BARREL":
                        _clusters[icluster].gatheringThreshold = cms.double(ecal_thresholds[int(lumi)])

    return process

def ecal_complete_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.ECalSD.AgeingWithSlopeLY = cms.untracked.bool(True)
    if hasattr(process,'ecal_digi_parameters'):
        process.ecal_digi_parameters.UseLCcorrection = cms.untracked.bool(False)
    return process

def ageMTD(process,lumi):

    mtd_lumis = [1000, 3000]
    mtd_parameters = {
        1000: {
            "light_output": 1444.,
            "dark_count_rate": 12.,
            "pulse_t2_threshold": 7.199,
            "pulse_e_threshold": 10.48,
            "sipm_gain": 417840.,
            "pulse_tbranch_a": [-2.2, 1.89e-8],
            "pulse_ebranch_a": [-1.1, 1.9e-8],
            "time_at_thr1rise": [3.1e5, -0.580],
            "time_at_thr2rise": [1.11e6, -0.631],
            "time_over_thr1": [1.3e9, 9.58676, -2.51351e-10, 5.98501e-18, -3.326821e-27, -7.61576e-10, 13.21],
            "slew_rate": [1.3e9, -0.8, 8.7e-9, 11.1],
            "pulse_q": [-43.5, 0.0793],
            "hit_time_res": "0.143789*pow(x,-1.09324)+0.0166063"
        },
        3000: {
            "light_output": 1004.,
            "dark_count_rate": 20.,
            "pulse_t2_threshold": 7.199,
            "pulse_e_threshold": 4.95,
            "sipm_gain": 330990.,
            "pulse_tbranch_a": [-2.2, 1.89e-8],
            "pulse_ebranch_a": [-1.1, 1.9e-8],
            "time_at_thr1rise": [3.1e5, -0.580],
            "time_at_thr2rise": [1.11e6, -0.631],
            "time_over_thr1": [1.3e9, 9.58676, -2.51351e-10, 5.98501e-18, -3.326821e-27, -7.61576e-10, 13.21],
            "slew_rate": [1.3e9, -0.8, 8.7e-9, 11.1],
            "pulse_q": [-34.3, 0.085],
            "hit_time_res": "0.2567*pow(x,-1.10973)+0.0165099"
        },
    }

    if int(lumi) in mtd_lumis:
        if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'fastTimingLayer'):
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.DeviceSimulation.LightOutput = cms.double(mtd_parameters[lumi]["light_output"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.LightOutput = cms.double(mtd_parameters[lumi]["light_output"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.PulseT2Threshold = cms.double(mtd_parameters[lumi]["pulse_t2_threshold"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.PulseEThrershold = cms.double(mtd_parameters[lumi]["pulse_e_threshold"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.DarkCountRate  = cms.double(mtd_parameters[lumi]["dark_count_rate"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.SiPMGain = cms.double(mtd_parameters[lumi]["sipm_gain"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.PulseTbranchAParam = cms.vdouble(mtd_parameters[lumi]["pulse_tbranch_a"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.PulseEbranchAParam = cms.vdouble(mtd_parameters[lumi]["pulse_ebranch_a"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.TimeAtThr1RiseParam = cms.vdouble(mtd_parameters[lumi]["time_at_thr1rise"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.TimeAtThr2RiseParam = cms.vdouble(mtd_parameters[lumi]["time_at_thr2rise"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.TimeOverThr1Param = cms.vdouble(mtd_parameters[lumi]["time_over_thr1"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.SlewRateParam = cms.vdouble(mtd_parameters[lumi]["slew_rate"])
            process.mix.digitizers.fastTimingLayer.barrelDigitizer.ElectronicsSimulation.PulseQParam = cms.vdouble(mtd_parameters[lumi]["pulse_q"])
        # --- This is for the workflows with premixing:
        if hasattr(process,'mixData') and hasattr(process.mixData,'workers') and hasattr(process.mixData.workers,'mtdBarrel'):
            process.mixData.workers.mtdBarrel.DeviceSimulation.LightOutput = cms.double(mtd_parameters[lumi]["light_output"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.LightOutput = cms.double(mtd_parameters[lumi]["light_output"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.PulseT2Threshold = cms.double(mtd_parameters[lumi]["pulse_t2_threshold"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.PulseEThrershold = cms.double(mtd_parameters[lumi]["pulse_e_threshold"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.DarkCountRate  = cms.double(mtd_parameters[lumi]["dark_count_rate"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.SiPMGain = cms.double(mtd_parameters[lumi]["sipm_gain"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.PulseTbranchAParam = cms.vdouble(mtd_parameters[lumi]["pulse_tbranch_a"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.PulseEbranchAParam = cms.vdouble(mtd_parameters[lumi]["pulse_ebranch_a"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.TimeAtThr1RiseParam = cms.vdouble(mtd_parameters[lumi]["time_at_thr1rise"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.TimeAtThr2RiseParam = cms.vdouble(mtd_parameters[lumi]["time_at_thr2rise"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.TimeOverThr1Param = cms.vdouble(mtd_parameters[lumi]["time_over_thr1"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.SlewRateParam = cms.vdouble(mtd_parameters[lumi]["slew_rate"])
            process.mixData.workers.mtdBarrel.ElectronicsSimulation.PulseQParam = cms.vdouble(mtd_parameters[lumi]["pulse_q"])
        # --- This is for the uncalibrated RecHit reconstruction:
        if hasattr(process,'mtdUncalibratedRecHits') and hasattr(process.mtdUncalibratedRecHits,'barrel'):
            process.mtdUncalibratedRecHits.barrel.npePerMeV = cms.double(mtd_parameters[lumi]["light_output"])
            process.mtdUncalibratedRecHits.barrel.npeToADC = cms.vdouble(mtd_parameters[lumi]["pulse_q"])
            process.mtdUncalibratedRecHits.barrel.timeResolutionInNs = cms.string(mtd_parameters[lumi]["hit_time_res"])
            process.mtdUncalibratedRecHits.barrel.timeWalkCorrection = cms.string(
                "{}/{}*pow({}/{}*(x-{}),{})".format( mtd_parameters[lumi]["time_at_thr1rise"][0],
                                                     0.020, # [ns], TDC LSB
                                                     mtd_parameters[lumi]["sipm_gain"],
                                                     mtd_parameters[lumi]["pulse_q"][1],
                                                     mtd_parameters[lumi]["pulse_q"][0],
                                                     mtd_parameters[lumi]["time_at_thr1rise"][1]
                )
            )

    return process

def customise_aging_300(process):
    process=ageHcal(process,300,5.0e34,"nominal")
    process=ageEcal(process,300,5.0e34)
    return process

def customise_aging_1000(process):
    process=ageHcal(process,1000,5.0e34,"nominal")
    process=turn_off_HE_aging(process) #avoid conflict between HGCal and Hcal in phase2 geom configuration
    process=ageEcal(process,1000,5.0e34)
    process=ageMTD(process,1000)
    return process

def customise_aging_3000(process):
    process=ageHcal(process,3000,5.0e34,"nominal")
    process=turn_off_HE_aging(process) #avoid conflict between HGCal and Hcal in phase2 geom configuration
    process=ageEcal(process,3000,5.0e34)
    process=ageMTD(process,3000)
    process=agedHGCal(process)
    process=agedHFNose(process)
    return process

def customise_aging_3000_ultimate(process):
    process=ageHcal(process,3000,7.5e34,"ultimate")
    process=turn_off_HE_aging(process) #avoid conflict between HGCal and Hcal in phase2 geom configuration
    process=ageEcal(process,3000,7.5e34)
    process=ageMTD(process,3000)
    process=agedHGCal(process)
    process=agedHFNose(process)
    return process

def customise_aging_4500_ultimate(process):
    process=ageHcal(process,4500,7.5e34,"ultimate")
    process=turn_off_HE_aging(process) #avoid conflict between HGCal and Hcal in phase2 geom configuration
    process=ageEcal(process,4500,7.5e34)
    process=agedHGCal(process)
    process=agedHFNose(process)
    return process
