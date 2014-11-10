import FWCore.ParameterSet.Config as cms

def agePixel(process,lumi):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'pixel') and not hasattr(process.mix.digitizers.pixel,'NoAging'):
        process.mix.digitizers.pixel.DoPixelAging = cms.bool(True)
        if lumi>299:#scenario updated with new values E. Migliore et al. 140807
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix1 = cms.double(0.4)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix2 = cms.double(0.3)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix3 = cms.double(0.0)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix4 = cms.double(0.0)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix1 = cms.double(0.36)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix2 = cms.double(0.36)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix3 = cms.double(0.36)
            
        if lumi>399:#scenario not updated 
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix1 = cms.double(1.2)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix2 = cms.double(0.43)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix3 = cms.double(0.26)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix4 = cms.double(0.09)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix1 = cms.double(0.51)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix2 = cms.double(0.51)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix3 = cms.double(0.51)
                
        if lumi>499:#scenario updated with new values E. Migliore et al. 140807
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix1 = cms.double(0.7)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix2 = cms.double(0.35)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix3 = cms.double(0.3)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix4 = cms.double(0.0)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix1 = cms.double(0.42)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix2 = cms.double(0.42)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix3 = cms.double(0.42)
            
        if lumi>599:#scenario not updated 
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix1 = cms.double(0.0)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix2 = cms.double(0.8)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix3 = cms.double(0.48)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix4 = cms.double(0.16)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix1 = cms.double(0.96)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix2 = cms.double(0.96)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix3 = cms.double(0.96)
           
        if lumi>699:#scenario not updated 
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix1 = cms.double(0.8)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix2 = cms.double(1.0)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix3 = cms.double(0.6)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix4 = cms.double(0.2)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix1 = cms.double(1.2)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix2 = cms.double(1.2)
        if lumi>799:#scenario not updated 
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix1 = cms.double(1.0)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix2 = cms.double(1.2)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix3 = cms.double(0.72)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix4 = cms.double(0.24)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix1 = cms.double(1.4)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix2 = cms.double(1.4)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix3 = cms.double(1.4)
        # new pixels!!    
        if lumi>801:
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix1 = cms.double(0.)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix2 = cms.double(0.)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix3 = cms.double(0.)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_BPix4 = cms.double(0.)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix1 = cms.double(0.)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix2 = cms.double(0.)
            process.mix.digitizers.pixel.thePixelPseudoRadDamage_FPix3 = cms.double(0.)

    return process    

#scenario = 1 enables default, 0 disables
def ageHB(process,scenario):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        process.mix.digitizers.hcal.HBDarkening = cms.uint32(scenario)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HBRecalibration = cms.uint32(scenario)
    return process

#scenario = 1 enables default, 0 disables
def ageHE(process,scenario):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        process.mix.digitizers.hcal.HEDarkening = cms.uint32(scenario)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HERecalibration = cms.uint32(scenario)
    return process

#turnon = True enables default, False disables
def ageHF(process,turnon):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        process.mix.digitizers.hcal.HFDarkening = cms.bool(turnon)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HFRecalibration = cms.bool(turnon)
    return process

#turnon = True enables default, False disables
#needs lumi to set proper ZS thresholds
def ageSipm(process,turnon,lumi):
    #recalibration and darkening always together
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.SipmAging = cms.bool(turnon)

    #change ZS thresholds for SiPMs
    if hasattr(process,'simHcalDigis'):
        if lumi<499 or not turnon: #values for 0 to 200
            process.simHcalDigis.HBlevel=cms.int32(16)
            process.simHcalDigis.HElevel=cms.int32(16)
        elif lumi<999: #values for 500
            process.simHcalDigis.HBlevel=cms.int32(90)
            process.simHcalDigis.HElevel=cms.int32(30)
        elif lumi<2999: #values for 1000
            process.simHcalDigis.HBlevel=cms.int32(140)
            process.simHcalDigis.HElevel=cms.int32(42)
        else: #values for 3000
            process.simHcalDigis.HBlevel=cms.int32(209)
            process.simHcalDigis.HElevel=cms.int32(76) 
    return process

def ageHcal(process,lumi):

    instLumi=1.0e34
    if lumi>=1000:
        instLumi=5.0e34

    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):  
        process.mix.digitizers.hcal.DelivLuminosity = cms.double(float(lumi))  # integrated lumi in fb-1

    #these lines need to be further activated by turning on 'complete' aging for HF 
    if hasattr(process,'g4SimHits'):  
        process.g4SimHits.HCalSD.InstLuminosity = cms.double(float(instLumi))
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(float(lumi))

    #recalibration and darkening always together
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.iLumi = cms.double(float(lumi))

    #functions to enable individual subdet aging
    process = ageHE(process,1)
    process = ageHF(process,True)
    process = ageSipm(process,True,lumi)
        
    return process

def turn_on_HB_aging_1(process):
    process = ageHB(process,1)
    return process

def turn_off_HB_aging(process):
    process = ageHB(process,0)
    return process

def turn_on_HE_aging_1(process):
    process = ageHE(process,1)
    return process

def turn_on_HE_aging_2(process):
    process = ageHE(process,2)
    return process

def turn_on_HE_aging_3(process):
    process = ageHE(process,3)
    return process
    
def turn_off_HE_aging(process):
    process = ageHE(process,0)
    return process
    
def turn_on_HF_aging(process):
    process = ageHF(process,True)
    return process
    
def turn_off_HF_aging(process):
    process = ageHF(process,False)
    return process

def turn_off_Sipm_aging(process):
    process = ageSipm(process,False,0)
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

def hf_complete_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.HFDarkening = cms.untracked.bool(True)
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):    
        process.mix.digitizers.hcal.HFDarkening = cms.untracked.bool(False)
    return process

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
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        ## laser D
        cms.PSet(record = cms.string("EcalLaserAPDPNRatiosRcd"),
                 tag = cms.string("EcalLaserAPDPNRatios_TL300_IL1E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        ## L1 trigger
        cms.PSet(record = cms.string("EcalTPGLinearizationConstRcd"),
                 tag = cms.string("EcalTPGLinearizationConst_TL300_IL1E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        cms.PSet(record = cms.string('EcalLaserAlphasRcd'),
                 tag = cms.string('EcalLaserAlphas_EB_sic1_btcp1_EE_sic1_btcp1'),
                 connect = cms.untracked.string('frontier://FrontierPrep/CMS_COND_ECAL')
                 ),
        #VPT aging
        cms.PSet(record = cms.string('EcalLinearCorrectionsRcd'),
                 tag = cms.string('EcalLinearCorrections_mc'),
                 connect = cms.untracked.string('frontier://FrontierPrep/CMS_COND_ECAL')
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
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        ## laser D
        cms.PSet(record = cms.string("EcalLaserAPDPNRatiosRcd"),
                 tag = cms.string("EcalLaserAPDPNRatios_TL1000_IL5E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        ## L1 trigger
        cms.PSet(record = cms.string("EcalTPGLinearizationConstRcd"),
                 tag = cms.string("EcalTPGLinearizationConst_TL1000_IL5E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        cms.PSet(record = cms.string('EcalLaserAlphasRcd'),
                 tag = cms.string('EcalLaserAlphas_EB_sic1_btcp1_EE_sic1_btcp1'),
                 connect = cms.untracked.string('frontier://FrontierPrep/CMS_COND_ECAL')
                 ),
        #VPT aging
        cms.PSet(record = cms.string('EcalLinearCorrectionsRcd'),
                 tag = cms.string('EcalLinearCorrections_mc'),
                 connect = cms.untracked.string('frontier://FrontierPrep/CMS_COND_ECAL')
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
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        ## laser D
        cms.PSet(record = cms.string("EcalLaserAPDPNRatiosRcd"),
                 tag = cms.string("EcalLaserAPDPNRatios_TL3000_IL5E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        ## L1 trigger
        cms.PSet(record = cms.string("EcalTPGLinearizationConstRcd"),
                 tag = cms.string("EcalTPGLinearizationConst_TL3000_IL5E34_v2_mc"),
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
                 ),
        cms.PSet(record = cms.string('EcalLaserAlphasRcd'),
                 tag = cms.string('EcalLaserAlphas_EB_sic1_btcp1_EE_sic1_btcp1'),
                 connect = cms.untracked.string('frontier://FrontierPrep/CMS_COND_ECAL')
                 ),
        #VPT aging
        cms.PSet(record = cms.string('EcalLinearCorrectionsRcd'),
                 tag = cms.string('EcalLinearCorrections_mc'),
                 connect = cms.untracked.string('frontier://FrontierPrep/CMS_COND_ECAL')
                 )
        )
                                    )

    return process

