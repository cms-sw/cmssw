import FWCore.ParameterSet.Config as cms

ctppsLHCInfoESSource = cms.ESSource("CTPPSLHCInfoESSource",
  beamEnergy = cms.double(6500),  # GeV
  xangle = cms.double(185)  # murad
)
