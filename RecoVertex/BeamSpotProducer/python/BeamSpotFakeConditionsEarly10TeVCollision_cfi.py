import FWCore.ParameterSet.Config as cms

BeamSpotFakeConditions = cms.ESSource("BeamSpotFakeConditions",
    BeamType = cms.string('Early10TeVCollision'),
    #FileInPath xmlCalibration = "RecoVertex/BeamSpotProducer/test/BeamSpotFakeConditions.xml"
    UseDummy = cms.bool(True)
)


# foo bar baz
# ebENzyAYFKe18
# tsfPCjABVDqjQ
