import FWCore.ParameterSet.Config as cms

# plotter for functions in sector 45
ctppsPlotOpticalFunctions_45 = cms.EDAnalyzer("OpticalFunctionsPlotter",
    opticsFile = cms.FileInPath("CondFormats/CTPPSOpticsObjects/data/2016_preTS2/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root"),

    opticsObjects = cms.vstring(
        "ip5_to_station_150_h_1_lhcb2",
        "ip5_to_station_150_h_2_lhcb2"
      ),

    # in m
    vtx0_y_45 = cms.double(300E-6),
    vtx0_y_56 = cms.double(200E-6),

    # in rad
    half_crossing_angle_45 = cms.double(+179.394E-6),
    half_crossing_angle_56 = cms.double(+191.541E-6),

)

# plotter for functions in sector 56
ctppsPlotOpticalFunctions_56 = ctppsPlotOpticalFunctions_45.clone(
    opticsFile = cms.FileInPath("CondFormats/CTPPSOpticsObjects/data/2016_preTS2/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root"),
    opticsObjects = cms.vstring("ip5_to_station_150_h_1_lhcb1", "ip5_to_station_150_h_2_lhcb1"),
)

