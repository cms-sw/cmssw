import FWCore.ParameterSet.Config as cms
from CondCore.CondDB.CondDB_cfi import *

CondDB.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')

EcalTPGDoubleWeightRecords = cms.ESSource("PoolDBESSource",CondDB,
                                toGet = cms.VPSet(
                                            cms.PSet(record = cms.string("EcalTPGOddWeightGroupRcd"),
                                                    tag = cms.string("EcalTPGOddWeightGroup_mc")),
                                            cms.PSet(record = cms.string("EcalTPGOddWeightIdMapRcd"),
                                                    tag = cms.string("EcalTPGOddWeightIdMap_mc")),
                                            cms.PSet(record = cms.string("EcalTPGTPModeRcd"),
                                                    tag = cms.string("EcalTPGTPMode_mc")),
                                            )
                        )
