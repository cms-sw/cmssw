import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = 
cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseII/Tilted/EmptyPixelSkimmedGeometry.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS")
siPixelLorentzAngleESSource = cms.ESSource("PoolDBESSource",
                                           CondDB,
                                           toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelLorentzAngleRcd'),
                                                                      tag = cms.string("SiPixelLorentzAngle_phase2_T11_v0_mc")
                                                                  ),
                                                             cms.PSet(record = cms.string('SiPixelLorentzAngleSimRcd'),
                                                                      tag = cms.string("SiPixelSimLorentzAngle_phase2_T11_v0_mc")
                                                                  ),
                                                             cms.PSet(record = cms.string('SiPixelLorentzAngleRcd'),
                                                                      tag = cms.string("SiPixelLorentzAngle_phase2_forWidth_T11_v0_mc"),
                                                                      label = cms.untracked.string("forWidth")
                                                                  )
                                                             )
                                            )
es_prefer_lorentz = cms.ESPrefer("PoolDBESSource","siPixelLorentzAngleESSource")
