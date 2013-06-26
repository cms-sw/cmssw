import FWCore.ParameterSet.Config as cms

BeamSpotFakeConditions = cms.ESSource("BeamSpotFakeConditions",
    BeamType = cms.string('SimpleGaussian'),
    #FileInPath xmlCalibration = "RecoVertex/BeamSpotProducer/test/BeamSpotFakeConditions.xml"
    UseDummy = cms.bool(True)
)


