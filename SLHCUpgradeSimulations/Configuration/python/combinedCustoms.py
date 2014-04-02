import FWCore.ParameterSet.Config as cms
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import customise as customiseBE
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import customise as customiseBE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10D import customise as customiseBE5DPixel10D

from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import l1EventContent as customise_ev_BE
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import l1EventContent as customise_ev_BE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10D import l1EventContent as customise_ev_BE5DPixel10D

from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import customise as customiseLB6PS
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_4LPS_2L2S import customise as customiseLB4LPS_2L2S
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import l1EventContent as customise_ev_LB6PS
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_4LPS_2L2S import l1EventContent as customise_ev_LB4LPS_2L2S
from SLHCUpgradeSimulations.Configuration.phase1TkCustomsPixel10D import customise as customisePhase1TkPixel10D
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import customise as customiseTTI
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import l1EventContent_TTI as customise_ev_l1tracker
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import l1EventContent_TTI_forHLT

from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_NoCrossing
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise as customisePhase1Tk
from SLHCUpgradeSimulations.Configuration.HCalCustoms import customise_HcalPhase1, customise_HcalPhase0, customise_HcalPhase2
from SLHCUpgradeSimulations.Configuration.gemCustoms import customise as customise_gem
from SLHCUpgradeSimulations.Configuration.me0Customs import customise as customise_me0
from SLHCUpgradeSimulations.Configuration.rpcCustoms import customise as customise_rpc
from SLHCUpgradeSimulations.Configuration.fastsimCustoms import customiseDefault as fastCustomiseDefault
from SLHCUpgradeSimulations.Configuration.fastsimCustoms import customisePhase2 as fastCustomisePhase2
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_noPixelDataloss as cNoPixDataloss
from SLHCUpgradeSimulations.Configuration.customise_ecalTime import cust_ecalTime
import SLHCUpgradeSimulations.Configuration.aging as aging

def cust_phase1_Pixel10D(process):
    process=customisePostLS1(process)
    process=customisePhase1TkPixel10D(process)
    process=customise_HcalPhase1(process)
    return process 

def cust_phase2_BE5DPixel10D(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    return process

def cust_phase2_BE5D(process):
    process=customisePostLS1(process)
    process=customiseBE5D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5D(process)
    return process

def cust_phase2_BE(process):
    process=customisePostLS1(process)
    process=customiseBE(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE(process)
    return process

def cust_phase2_LB6PS(process): #obsolete
    process=customisePostLS1(process)
    process=customiseLB6PS(process)
    process=customise_ev_LB6PS(process)
    return process

def cust_phase2_LB4LPS_2L2S(process):#obsolete
    process=customisePostLS1(process)
    process=customiseLB4LPS_2L2S(process)
    process=customise_ev_LB4LPS_2L2S(process)
    return process

def cust_2017(process):
    process=customisePostLS1(process)
    process=customisePhase1Tk(process)
    process=customise_HcalPhase0(process)
#    process=fixRPCConditions(process)
    return process

def cust_2017EcalTime(process):
    process=cust_2017(process)
    process=cust_ecalTime(process)
    return process

def cust_2019(process):
    process=customisePostLS1(process)
    process=customisePhase1Tk(process)
    process=customise_HcalPhase1(process)
#    process=fixRPCConditions(process)
    return process

def cust_2019WithGem(process):
    process=cust_2019(process)
    process=customise_gem(process)
    return process

def cust_2023(process):
    process=customisePostLS1(process)
    process=customiseBE5D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5D(process)
    process=customise_gem(process)
    process=customise_rpc(process)
    return process

def cust_2023Pixel(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=customise_gem(process)
    return process

def cust_2023Pixel(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=customise_gem(process)
    return process

def cust_2023Muon(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=customise_gem(process)
    process=customise_rpc(process)
    process=customise_me0(process)
    return process

def cust_2023TTI(process):
    process=customisePostLS1(process)
    process=customiseTTI(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase0(process)
    process=customise_ev_l1tracker(process)
    return process

def cust_2023TTI_forHLT(process):
    process=customisePostLS1(process)
    process=customiseTTI(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase0(process)
    process=l1EventContent_TTI_forHLT(process)
    return process


def noCrossing(process):
    process=customise_NoCrossing(process)
    return process

##### clone aging.py here 
def agePixel(process,lumi):
    process=aging.agePixel(process,lumi)
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

def fastsimDefault(process):
    return fastCustomiseDefault(process)

def fastsimPhase2(process):
    return fastCustomisePhase2(process)

def bsStudyStep1(process):
    process.VtxSmeared.MaxZ = 11.0
    process.VtxSmeared.MinZ = -11.0
    return process

def bsStudyStep2(process):
    process.initialStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.02),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.7)
        )
    process.highPtTripletStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.02),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.7)
        )
    process.lowPtQuadStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.02),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.2)
        )
    process.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.015),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.35)
        )
    process.detachedQuadStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.5),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.3)
        )
    return process

def customise_noPixelDataloss(process):
    return cNoPixDataloss(process)

