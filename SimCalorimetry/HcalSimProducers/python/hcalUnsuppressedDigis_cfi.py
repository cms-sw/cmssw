import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *

# make a block so other modules, such as the data mixing module, can
# also run simulation

hcalSimBlock = cms.PSet(    
    hcalSimParameters,
    # whether cells with MC signal get noise added
    doNoise = cms.bool(True),
    killHE = cms.bool(False),
    HcalPreMixStage1 = cms.bool(False),
    HcalPreMixStage2 = cms.bool(False),
    # whether cells with no MC signal get an empty signal created
    # These empty signals can get noise via the doNoise flag
    doEmpty = cms.bool(True),
    doIonFeedback = cms.bool(True),
    doThermalNoise = cms.bool(True),
    doTimeSlew = cms.bool(True),
    doHFWindow = cms.bool(False),
    hitsProducer = cms.string('g4SimHits'),
    DelivLuminosity = cms.double(0),
    TestNumbering = cms.bool(False),
    HEDarkening = cms.bool(False),
    HFDarkening = cms.bool(False),
    minFCToDelay=cms.double(5.), # old TC model! set to 5 for the new one
    debugCaloSamples=cms.bool(False),
    ignoreGeantTime=cms.bool(False),
    # settings for SimHit test injection
    injectTestHits = cms.bool(False),
    # if no time is specified for injected hits, t = 0 will be used
    # (recommendation: enable "ignoreGeantTime" in that case to set t = tof)
    # otherwise, need 1 time value per energy value
    injectTestHitsEnergy = cms.vdouble(),
    injectTestHitsTime = cms.vdouble(),
    # format for cells: subdet, ieta, iphi, depth
    # multiple quadruplets can be specified
    # if instead only 1 value is given, 
    # it will be interpreted as an entire subdetector
    injectTestHitsCells = cms.vint32()
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify( hcalSimBlock, hitsProducer=cms.string('famosSimHits') )

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hcalSimBlock, TestNumbering = cms.bool(True) )

# remove HE processing for phase 2, completely put in HGCal land
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(hcalSimBlock, killHE = cms.bool(True) )
