import FWCore.ParameterSet.Config as cms

from SimCTPPS.OpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2016PreTS2

# plotter for functions in sector 45
ctppsPlotOpticalFunctions_45 = cms.EDAnalyzer("OpticalFunctionsPlotter",
    opticsFile = cms.FileInPath("CondFormats/CTPPSOpticsObjects/data/2016_preTS2/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root"),
    opticsObjects = cms.vstring("ip5_to_station_150_h_1_lhcb2", "ip5_to_station_150_h_2_lhcb2"),
    beamConditions = lhcBeamConditions_2016PreTS2,
    minXi = cms.double(0.0),
    maxXi = cms.double(0.151),
    xiStep = cms.double(0.001),
)

# plotter for functions in sector 56
ctppsPlotOpticalFunctions_56 = ctppsPlotOpticalFunctions_45.clone(
    opticsFile = cms.FileInPath("CondFormats/CTPPSOpticsObjects/data/2016_preTS2/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root"),
    opticsObjects = cms.vstring("ip5_to_station_150_h_1_lhcb1", "ip5_to_station_150_h_2_lhcb1"),
)

