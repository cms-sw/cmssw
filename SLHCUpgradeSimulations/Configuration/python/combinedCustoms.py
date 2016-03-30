import FWCore.ParameterSet.Config as cms
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_NoCrossing
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise as customisePhase1Tk
from SLHCUpgradeSimulations.Configuration.HCalCustoms import customise_HcalPhase1, customise_HcalPhase0

from SLHCUpgradeSimulations.Configuration.gemCustoms import customise2019 as customise_gem2019
from SLHCUpgradeSimulations.Configuration.gemCustoms import customise2023 as customise_gem2023
from SLHCUpgradeSimulations.Configuration.me0Customs import customise as customise_me0
from SLHCUpgradeSimulations.Configuration.rpcCustoms import customise as customise_rpc
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixRPCConditions

from SLHCUpgradeSimulations.Configuration.phase2TkTilted import customise as customiseTiltedTK

import SLHCUpgradeSimulations.Configuration.aging as aging

from Configuration.StandardSequences.Eras import eras

def cust_2017(process):
    # To allow simulatenous use of customisation and era while the era migration is in progress
    if not eras.run2_common.isChosen():
        process=customisePostLS1(process,displayDeprecationWarning=False)
    process=customisePhase1Tk(process)
    #process=customise_HcalPhase0(process)
    return process


def cust_2023sim(process):
    # To allow simulatenous use of customisation and era while the era migration is in progress
    return process

def cust_2023dev(process):
    # To allow simulatenous use of customisation and era while the era migration is in progress
    process=customiseTiltedTK(process)
    return process

def cust_2023LReco(process):
    # To allow simulatenous use of customisation and era while the era migration is in progress
    return process



######Below are the customized used for SLHC releases 
def cust_2019(process):
    process=customisePostLS1(process,displayDeprecationWarning=False)
    process=customisePhase1Tk(process)
    process=customise_HcalPhase1(process)
    return process

def cust_2019WithGem(process):
    process=cust_2019(process)
    process=customise_gem2019(process)
    return process

def cust_2023MuonOnly(process):
    process=customise_gem2023(process)
    process=customise_rpc(process)
    process=fixRPCConditions(process)
    return process

def noCrossing(process):
    process=customise_NoCrossing(process)
    return process

def cust_2023HGCal_common(process):   
    process = customise_rpc(process)
    process = fixRPCConditions(process)
    process = customise_HcalPhase1(process)
    process = customisePhase1Tk(process)    
    if hasattr(process,'L1simulation_step'):
        process.simEcalTriggerPrimitiveDigis.BarrelOnly = cms.bool(True)
    if hasattr(process,'digitisation_step'):
        if hasattr(process.mix.digitizers,'ecal'):
            process.mix.digitizers.ecal.doEE = cms.bool(False)
            process.mix.digitizers.ecal.doES = cms.bool(False)
        process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')
        process.mix.digitizers.hgceeDigitizer=process.hgceeDigitizer
        process.mix.digitizers.hgchebackDigitizer=process.hgchebackDigitizer
        process.mix.digitizers.hgchefrontDigitizer=process.hgchefrontDigitizer
        # update the HCAL Endcap for BH geom.
        newFactors = cms.vdouble(
            210.55, 197.93, 186.12, 189.64, 189.63,
            189.96, 190.03, 190.11, 190.18, 190.25,
            190.32, 190.40, 190.47, 190.54, 190.61,
            190.69, 190.83, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94)
        process.mix.digitizers.hcal.he.samplingFactors = newFactors
        process.mix.digitizers.hcal.he.photoelectronsToAnalog = cms.vdouble([10.]*len(newFactors))
        # Also need to tell the MixingModule to make the correct collections available from
        # the pileup, even if not creating CrossingFrames.
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgceeDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgchebackDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgchefrontDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.subdets.append( process.hgceeDigitizer.hitCollection.value() )
        process.mix.mixObjects.mixCH.subdets.append( process.hgchebackDigitizer.hitCollection.value() )
        process.mix.mixObjects.mixCH.subdets.append( process.hgchefrontDigitizer.hitCollection.value() )    
    return process

def cust_2023HGCal(process):    
    process = cust_2023HGCal_common(process)
    return process

def cust_2023HGCalMuon(process):    
    process = customise_me0(process)
    process = cust_2023HGCal_common(process)    
    return process

##### clone aging.py here 
def agePixel(process,lumi):
    process=process.agePixel(process,lumi)
    return process

def ageHcal(process,lumi):
    process=aging.ageHcal(process,lumi)
    return process

def ageEcal(process,lumi):
    process=aging.ageEcal(process,lumi)
    return process

def customise_aging_100(process):
    process=aging.customise_aging_100(process)
    return process

def customise_aging_200(process):
    process=aging.customise_aging_200(process)
    return process

def customise_aging_300(process):
    process=aging.customise_aging_300(process)
    return process

def customise_aging_400(process):
    process=aging.customise_aging_400(process)
    return process

def customise_aging_500(process):
    process=aging.customise_aging_500(process)
    return process

def customise_aging_600(process):
    process=aging.customise_aging_600(process)
    return process

def customise_aging_700(process):
    process=aging.customise_aging_700(process)
    return process


def customise_aging_1000(process):
    process=aging.customise_aging_1000(process)
    return process

def customise_aging_3000(process):
    process=aging.customise_aging_3000(process)
    return process

def customise_aging_ecalonly_300(process):
    process=aging.customise_aging_ecalonly_300(process)
    return process

def customise_aging_ecalonly_1000(process):
    process=aging.customise_aging_ecalonly_1000(process)
    return process

def customise_aging_ecalonly_3000(process):
    process=aging.customise_aging_ecalonly_3000(process)
    return process

def customise_aging_newpixel_1000(process):
    process=aging.customise_aging_newpixel_1000(process)
    return process

def customise_aging_newpixel_3000(process):
    process=aging.customise_aging_newpixel_3000(process)
    return process

def ecal_complete_aging(process):
    process=aging.ecal_complete_aging(process)
    return process

def turn_off_HE_aging(process):
    process=aging.turn_off_HE_aging(process)
    return process

def turn_off_HF_aging(process):
    process=aging.turn_off_HF_aging(process)
    return process

def turn_off_Pixel_aging(process):
    process=aging.turn_off_Pixel_aging(process)
    return process

def turn_on_Pixel_aging_1000(process):
    process=aging.turn_on_Pixel_aging_1000(process)
    return process

def hf_complete_aging(process):
    process=aging.hf_complete_aging(process)
    return process
    
def ecal_complete_aging_300(process):
    process=aging.ecal_complete_aging_300(process)
    return process

def ecal_complete_aging_1000(process):
    process=aging.ecal_complete_aging_1000(process)
    return process

def ecal_complete_aging_3000(process):
    process=aging.ecal_complete_aging_3000(process)
    return process

def fixEcalConditions_150(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalSRSettingsRcd"),
                 tag = cms.string("EcalSRSettings_TL150_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    return process

def fixEcalConditions_100(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalSRSettingsRcd"),
                 tag = cms.string("EcalSRSettings_TL100_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    return process

def fixEcalConditions_200(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalSRSettingsRcd"),
                 tag = cms.string("EcalSRSettings_TL200_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    return process

def fixEcalConditions_300(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalSRSettingsRcd"),
                 tag = cms.string("EcalSRSettings_TL300_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    return process

def fixEcalConditions_500(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalSRSettingsRcd"),
                 tag = cms.string("EcalSRSettings_TL500_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    return process

def fixEcalConditions_1000(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalSRSettingsRcd"),
                 tag = cms.string("EcalSRSettings_TL1000_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_3GeV_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    return process

def fixEcalConditions_3000(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalSRSettingsRcd"),
                 tag = cms.string("EcalSRSettings_TL3000_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_4GeV_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
        )
                                    )
    return process
