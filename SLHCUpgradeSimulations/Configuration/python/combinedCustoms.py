import FWCore.ParameterSet.Config as cms
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_NoCrossing
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise as customisePhase1Tk
from SLHCUpgradeSimulations.Configuration.HCalCustoms import customise_HcalPhase1, customise_HcalPhase0

from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixRPCConditions

from SLHCUpgradeSimulations.Configuration.phase2TkTilted import customise as customiseTiltedTK
from SLHCUpgradeSimulations.Configuration.phase2TkFlat import customise as customiseFlatTK

import SLHCUpgradeSimulations.Configuration.aging as aging


def cust_2023sim(process):
    # To allow simulatenous use of customisation and era while the era migration is in progress
    return process

def cust_2023tilted(process):
    # To allow simulatenous use of customisation and era while the era migration is in progress
    process=customiseTiltedTK(process)
    return process

def cust_2023LReco(process):
    # To allow simulatenous use of customisation and era while the era migration is in progress
    process=customiseFlatTK(process)
    return process

def cust_2023GReco(process):
    # To allow simulatenous use of customisation and era while the era migration is in progress
    process=customiseFlatTK(process)
    return process




######Below are the customized used for SLHC releases 
def cust_2019(process):
    process=customisePostLS1(process,displayDeprecationWarning=False)
    process=customisePhase1Tk(process)
    process=customise_HcalPhase1(process)
    return process

def cust_2023MuonOnly(process):
    process=fixRPCConditions(process)
    return process

def noCrossing(process):
    process=customise_NoCrossing(process)
    return process

def cust_2023HGCal_common(process):      
    return process

def cust_2023HGCal(process):    
    process = cust_2023HGCal_common(process)
    return process

def cust_2023HGCalMuon(process):    
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
