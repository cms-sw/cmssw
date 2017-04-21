import FWCore.ParameterSet.Config as cms

def agePixel(process,lumi):
    prd=1.0
    if lumi>299:
        prd=1.0
    if lumi>499:
        prd=1.5
    if lumi>999:
        prd=0.
        
    # danger - watch for someone turning off pixel aging - if off - leave off
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'pixel') and not hasattr(process.mix.digitizers.pixel,'NoAging'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(float(prd*7.65))
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(16.5)
    return process    

# handle normal mixing or premixing
def getHcalDigitizer(process):
    if hasattr(process,'mixData'):
        return process.mixData
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        return process.mix.digitizers.hcal
    return None

# turnon = True enables default, False disables
# recalibration and darkening always together
def ageHB(process,turnon):
    if turnon:
        from CalibCalorimetry.HcalPlugins.HBHEDarkening_cff import HBDarkeningEP
        process.HBDarkeningEP = HBDarkeningEP
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HBDarkening = cms.bool(turnon)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HBRecalibration = cms.bool(turnon)
    return process

def ageHE(process,turnon):
    if turnon:
        from CalibCalorimetry.HcalPlugins.HBHEDarkening_cff import HEDarkeningEP
        process.HEDarkeningEP = HEDarkeningEP
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

# needs lumi to set proper ZS thresholds (tbd)
def ageSiPM(process,turnon,lumi):
    process.es_hardcode.hbUpgrade.doRadiationDamage = turnon
    process.es_hardcode.heUpgrade.doRadiationDamage = turnon

    # todo: determine ZS threshold adjustments

    return process

def ageHcal(process,lumi):
    instLumi=1.0e34
    if lumi>=1000:
        instLumi=5.0e34

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
    process = ageHB(process,True)
    process = ageHE(process,True)
    process = ageHF(process,True)
    process = ageSiPM(process,True,lumi)

    return process

def turn_on_HB_aging(process):
    process = ageHB(process,True)
    return process

def turn_off_HB_aging(process):
    process = ageHB(process,False)
    return process

def turn_on_HE_aging(process):
    process = ageHE(process,True)
    return process
    
def turn_off_HE_aging(process):
    process = ageHE(process,False)
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

def ageEcal(process,lumi):

    instLumi=1.0e34
    if lumi>=1000:
        instLumi=5.0e34
        
    if hasattr(process,'g4SimHits'):
        #these lines need to be further activiated by tuning on 'complete' aging for ecal 
        process.g4SimHits.ECalSD.InstLuminosity = cms.double(instLumi)
        process.g4SimHits.ECalSD.DelivLuminosity = cms.double(float(lumi))
    return process

def customise_aging_100(process):

    process=ageHcal(process,100)
    process=ageEcal(process,100)
    process=agePixel(process,100)
    return process

def customise_aging_200(process):

    process=ageHcal(process,200)
    process=ageEcal(process,200)
    process=agePixel(process,200)
    return process

def customise_aging_300(process):

    process=ageHcal(process,300)
    process=ageEcal(process,300)
    process=agePixel(process,300)
    return process

def customise_aging_400(process):

    process=ageHcal(process,400)
    process=ageEcal(process,400)
    process=agePixel(process,400)
    return process

def customise_aging_500(process):

    process=ageHcal(process,500)
    process=ageEcal(process,500)
    process=agePixel(process,500)
    return process

def customise_aging_600(process):

    process=ageHcal(process,600)
    process=ageEcal(process,600)
    process=agePixel(process,600)
    return process

def customise_aging_700(process):

    process=ageHcal(process,700)
    process=ageEcal(process,700)
    process=agePixel(process,700)
    return process

def customise_aging_1000(process):

    process=ageHcal(process,1000)
    process=ageEcal(process,1000)
    process=agePixel(process,1000)
    return process

def customise_aging_3000(process):

    process=ageHcal(process,3000)
    process=ageEcal(process,3000)
    process=agePixel(process,3000)
    return process

def customise_aging_ecalonly_300(process):

    process=ageEcal(process,300)
    return process

def customise_aging_ecalonly_1000(process):

    process=ageEcal(process,1000)
    return process

def customise_aging_ecalonly_3000(process):

    process=ageEcal(process,3000)
    return process

def customise_aging_newpixel_1000(process):

    process=ageEcal(process,1000)
    process=ageHcal(process,1000)
    return process

def customise_aging_newpixel_3000(process):

    process=ageEcal(process,3000)
    process=ageHcal(process,3000)
    return process

#no hcal 3000

def ecal_complete_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.ECalSD.AgeingWithSlopeLY = cms.untracked.bool(True)
    if hasattr(process,'ecal_digi_parameters'):    
        process.ecal_digi_parameters.UseLCcorrection = cms.untracked.bool(False)
    return process

def turn_off_Pixel_aging(process):

    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):    
        setattr(process.mix.digitizers.pixel,'NoAging',cms.double(1.))
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(0.)
    return process

def turn_on_Pixel_aging_1000(process):
    # just incase we want aging afterall
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):    
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.5)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    return process

def ecal_complete_aging_300(process):
    process=ecal_complete_aging(process)

    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalPedestalsRcd"),
                 tag = cms.string("EcalPedestals_TL300_IL1E34_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        ## laser D
        cms.PSet(record = cms.string("EcalLaserAPDPNRatiosRcd"),
                 tag = cms.string("EcalLaserAPDPNRatios_TL300_IL1E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        ## L1 trigger
        cms.PSet(record = cms.string("EcalTPGLinearizationConstRcd"),
                 tag = cms.string("EcalTPGLinearizationConst_TL300_IL1E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        cms.PSet(record = cms.string('EcalLaserAlphasRcd'),
                 tag = cms.string('EcalLaserAlphas_EB_sic1_btcp1_EE_sic1_btcp1'),
                 connect = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS')
                 ),
        #VPT aging
        cms.PSet(record = cms.string('EcalLinearCorrectionsRcd'),
                 tag = cms.string('EcalLinearCorrections_mc'),
                 connect = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS')
                 )
        )
                                    )
    return process

    
def ecal_complete_aging_1000(process):
    process=ecal_complete_aging(process)

    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalPedestalsRcd"),
                 tag = cms.string("EcalPedestals_TL1000_IL5E34_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        ## laser D
        cms.PSet(record = cms.string("EcalLaserAPDPNRatiosRcd"),
                 tag = cms.string("EcalLaserAPDPNRatios_TL1000_IL5E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        ## L1 trigger
        cms.PSet(record = cms.string("EcalTPGLinearizationConstRcd"),
                 tag = cms.string("EcalTPGLinearizationConst_TL1000_IL5E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        cms.PSet(record = cms.string('EcalLaserAlphasRcd'),
                 tag = cms.string('EcalLaserAlphas_EB_sic1_btcp1_EE_sic1_btcp1'),
                 connect = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS')
                 ),
        #VPT aging
        cms.PSet(record = cms.string('EcalLinearCorrectionsRcd'),
                 tag = cms.string('EcalLinearCorrections_mc'),
                 connect = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS')
                 )
        )
                                    )
    return process


def ecal_complete_aging_3000(process):
    process=ecal_complete_aging(process)

    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalPedestalsRcd"),
                 tag = cms.string("EcalPedestals_TL3000_IL5E34_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        ## laser D
        cms.PSet(record = cms.string("EcalLaserAPDPNRatiosRcd"),
                 tag = cms.string("EcalLaserAPDPNRatios_TL3000_IL5E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        ## L1 trigger
        cms.PSet(record = cms.string("EcalTPGLinearizationConstRcd"),
                 tag = cms.string("EcalTPGLinearizationConst_TL3000_IL5E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 ),
        cms.PSet(record = cms.string('EcalLaserAlphasRcd'),
                 tag = cms.string('EcalLaserAlphas_EB_sic1_btcp1_EE_sic1_btcp1'),
                 connect = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS')
                 ),
        #VPT aging
        cms.PSet(record = cms.string('EcalLinearCorrectionsRcd'),
                 tag = cms.string('EcalLinearCorrections_mc'),
                 connect = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS')
                 )
        )
                                    )

    return process

