import FWCore.ParameterSet.Config as cms

ctppsLHCInfoESSource = cms.ESSource("CTPPSLHCInfoESSource",
  validityRange = cms.EventRange("270293:min - 290872:max"),
  beamEnergy = cms.double(6500),  # GeV
  xangle = cms.double(185)  # murad
)
