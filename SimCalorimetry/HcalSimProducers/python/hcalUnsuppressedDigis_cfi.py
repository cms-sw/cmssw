import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *
from CondCore.DBCommon.CondDBSetup_cfi import *

# make a block so other modules, such as the data mixing module, can
# also run simulation

hcalSimBlock = cms.PSet(    
    hcalSimParameters,
    # whether cells with MC signal get noise added
    doNoise = cms.bool(True),
    # whether cells with no MC signal get an empty signal created
    # These empty signals can get noise via the doNoise flag
    doEmpty = cms.bool(True),
    doHPDNoise = cms.bool(False),
    doIonFeedback = cms.bool(True),
    doThermalNoise = cms.bool(True),
    HBTuningParameter = cms.double(0.65),
    HETuningParameter = cms.double(0.65),
    HFTuningParameter = cms.double(0.65),
    HOTuningParameter = cms.double(0.65),
    #HPDNoiseLibrary = cms.PSet(
    #   FileName = cms.FileInPath("SimCalorimetry/HcalSimAlgos/data/hpdNoiseLibrary.root"),
    #   HPDName = cms.untracked.string("HPD")
    #),
    doTimeSlew = cms.bool(True),
    doHFWindow = cms.bool(True),
    hitsProducer = cms.string('g4SimHits'),
    injectTestHits = cms.bool(False)
)

es_cholesky = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string("HcalCholeskyMatricesRcd"),
            tag = cms.string("TestCholesky")
        )),
#    connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_31X_HCAL'),
    connect = cms.string('sqlite_file:CondFormats/HcalObjects/data/cholesky_sql.db'),
    authenticationMethod = cms.untracked.uint32(0),
)


#es_cholesky = cms.ESSource('HcalTextCalibrations',
#    input = cms.VPSet(
#        cms.PSet(
#            object = cms.string('CholeskyMatrices'),
#            file = cms.FileInPath("CondFormats/HcalObjects/data/CholeskyMatrices.txt")
#        ),
#    ),
#    appendToDataLabel = cms.string('reference')
#)

simHcalUnsuppressedDigis = cms.EDProducer("HcalDigiProducer",
    hcalSimBlock
)



