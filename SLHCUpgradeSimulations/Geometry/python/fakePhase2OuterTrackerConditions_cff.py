import FWCore.ParameterSet.Config as cms

##
## Fake Sim Outer Tracker Lorentz Angle
##

SiPhase2OTFakeLorentzAngleESSource = cms.ESSource('SiPhase2OuterTrackerFakeLorentzAngleESSource',
                                                  LAValue = cms.double(0.07),
                                                  recordName = cms.string("LorentzAngle")
                                                  )

es_prefer_fake_LA = cms.ESPrefer("SiPhase2OuterTrackerFakeLorentzAngleESSource","SiPhase2OTFakeLorentzAngleESSource")

##
## Fake Sim Outer Tracker Lorentz Angle
##

SiPhase2OTFakeSimLorentzAngleESSource = cms.ESSource('SiPhase2OuterTrackerFakeLorentzAngleESSource',
                                                     LAValue = cms.double(0.07),
                                                     recordName = cms.string("SimLorentzAngle")
                                                     )

es_prefer_fake_simLA = cms.ESPrefer("SiPhase2OuterTrackerFakeLorentzAngleESSource","SiPhase2OTFakeSimLorentzAngleESSource")

