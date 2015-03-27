import FWCore.ParameterSet.Config as cms
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_NoCrossing
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise as customisePhase1Tk
from SLHCUpgradeSimulations.Configuration.HCalCustoms import customise_HcalPhase1, customise_HcalPhase0
# --> to delete from SLHCUpgradeSimulations.Configuration.gemCustoms import customise2019 as customise_gem2019
# --> to delete from SLHCUpgradeSimulations.Configuration.gemCustoms import customise2023 as customise_gem2023

import SLHCUpgradeSimulations.Configuration.aging as aging


def cust_2017(process):
    process=customisePostLS1(process)
    process=customisePhase1Tk(process)
    # process=customise_HcalPhase0(process)
    # process=fixRPCConditions(process)
    return process

def cust_2019(process):
    process=customisePostLS1(process)
    process=customisePhase1Tk(process)
    process=customise_HcalPhase1(process)
    # process=fixRPCConditions(process)
    return process

# --> to delete
# def cust_2023_MuonOnly(process):
#     process=customisePostLS1(process)
#     process=customisePhase1Tk(process)
#     process=customise_HcalPhase1(process)    ### will only work for Geometry Extended2023, but not for Extended2023Muon, because HcalPhase1 collides with ME0
#     process=customise_gem2019(process)       ### difference between gem2019 and gem2023 is that in the latter special L1 trigger configs need to be loaded
#     process=fixRPCConditions(process)
#     process=fixDTAlignmentConditions(process)
#     process=digiOnlyMuonDetectors(process)
#     process=recoOnlyMuonDetectors(process)
#     return process


def noCrossing(process):
    process=customise_NoCrossing(process)
    return process


def fixRPCConditions(process):
    process.simMuonRPCDigis.digiModel = cms.string('RPCSimAverageNoiseEffCls')
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("RPCStripNoisesRcd"),
                 tag = cms.string("RPC_testCondition_192Strips_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_RPC")
                 ),
        cms.PSet(record = cms.string("RPCClusterSizeRcd"),
                 tag = cms.string("RPCClusterSize_PhaseII_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_36X_RPC")
                 )
        )
    )
    return process

def fixDTAlignmentConditions(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
            cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"),
                     tag = cms.string("MuonDTAPEObjectsExtended_v0_mc"),
                     connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_ALIGN_000")
                 )
            )
    ),
    process.GlobalTag.toGet.extend( cms.VPSet(
            cms.PSet(record = cms.string("DTRecoUncertaintiesRcd"),
                     tag = cms.string("DTRecoUncertainties_True_v0"),
                     connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_DT_000")
                 )
            )
    ),
    return process

def fixCSCAlignmentConditions(process):
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(
            cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"),
                     tag = cms.string("MuonCSCAPEObjectsExtended_v0_mc"),
                     connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_ALIGN_000")
                 )
            )
    ),
#     process.GlobalTag.toGet.extend( cms.VPSet(
#             cms.PSet(record = cms.string("DTRecoUncertaintiesRcd"),
#                      tag = cms.string("DTRecoUncertainties_True_v0"),
#                      connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_DT_000")
#                  )
#             )
#     ),
    return process


# --> delete
# def digiOnlyMuonDetectors(process):
#     if hasattr(process,'digitisation_step'):
#         process.doAllDigi = cms.Sequence(process.muonDigi)
#         process.theDigitizersValid = cms.PSet()
#     if hasattr(process,'DigiToRaw'):
#         process.DigiToRaw = cms.Sequence(process.cscpacker*process.dtpacker*process.rpcpacker*process.rawDataCollector)
#     return process

# --> delete
# def recoOnlyMuonDetectors(process):
#     if hasattr(process,'RawToDigi'):
#         process.RawToDigi = cms.Sequence(process.muonCSCDigis+process.muonDTDigis+process.muonRPCDigis)
#     # if hasattr(process,'reconstruction'):
#     # to be defined
#     return process


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
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
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
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
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
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
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
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
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
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
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
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_3GeV_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
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
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
                 )
        )
                                    )
    process.GlobalTag.toGet.extend( cms.VPSet(
        cms.PSet(record = cms.string("EcalTPGLutIdMapRcd"),
                 tag = cms.string("EcalTPGLutIdMap_beamv5_4GeV_upgrade_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_ECAL")
                 )
        )
                                    )
    return process
