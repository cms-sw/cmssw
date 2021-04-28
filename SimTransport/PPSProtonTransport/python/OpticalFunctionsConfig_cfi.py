import FWCore.ParameterSet.Config as cms

_baseOpticalFunctionsParameters = cms.PSet(
          TransportMethod = cms.string('OpticalFunctions'),
          ApplyZShift = cms.bool(True),
          lhcInfoLabel = cms.string(""),
          opticsLabel = cms.string(""),
          produceHitsRelativeToBeam = cms.bool(True),
          useEmpiricalApertures = cms.bool(True)
)

_config_2016_preTS2 = cms.PSet(
  opticalFunctionConfig = cms.PSet(
           es_source = cms.PSet(
                   validityRange = cms.EventRange("0:min - 999999:max"),
                   opticalFunctions = cms.VPSet(
                           cms.PSet( xangle = cms.double(185), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2016_preTS2/version2/185urad.root") )
                   )
          ),
          defaultCrossingAngle = cms.double(185.0)
  ),
  optics_parameters = cms.PSet(
          empiricalAperture45_xi0_int = cms.double(0.111),
          empiricalAperture45_xi0_slp = cms.double(0.000E+00),
          empiricalAperture45_a_int = cms.double(127.0),
          empiricalAperture45_a_slp = cms.double(-0.000),
          empiricalAperture56_xi0_int = cms.double(0.138),
          empiricalAperture56_xi0_slp = cms.double(0.000E+00),
          empiricalAperture56_a_int = cms.double(191.6),
          empiricalAperture56_a_slp = cms.double(-0.000),
  )
)

_config_2016_postTS2 = cms.PSet(
  opticalFunctionConfig = cms.PSet(
          es_source = cms.PSet(
                  validityRange = cms.EventRange("0:min - 999999:max"),
                  opticalFunctions = cms.VPSet(
                          cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2016_postTS2/version2/140urad.root") )
                  )
          ),
          defaultCrossingAngle = cms.double(140.0)
  ),
  optics_parameters = cms.PSet(
          empiricalAperture45_xi0_int = cms.double(0.104),
          empiricalAperture45_xi0_slp = cms.double(0.000E+00),
          empiricalAperture45_a_int = cms.double(116.4),
          empiricalAperture45_a_slp = cms.double(-0.000),
          empiricalAperture56_xi0_int = cms.double(0.110),
          empiricalAperture56_xi0_slp = cms.double(0.),
          empiricalAperture56_a_int = cms.double(150.0),
          empiricalAperture56_a_slp = cms.double(0.),
  )
)

_config_2017_preTS2 = cms.PSet(
  opticalFunctionConfig = cms.PSet(
          es_source = cms.PSet(
                  validityRange = cms.EventRange("0:min - 999999:max"),
                  opticalFunctions = cms.VPSet(
                          cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version4/120urad.root") ),
                          cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version4/130urad.root") ),
                          cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version4/140urad.root") )
                  )
          ),
          defaultCrossingAngle = cms.double(140.0)
  ),
  optics_parameters = cms.PSet(
          empiricalAperture45_xi0_int = cms.double(0.066),
          empiricalAperture45_xi0_slp = cms.double(3.536E-04),
          empiricalAperture45_a_int = cms.double(47.7),
          empiricalAperture45_a_slp = cms.double(0.447),
          empiricalAperture56_xi0_int = cms.double(0.062),
          empiricalAperture56_xi0_slp = cms.double(5.956E-04),
          empiricalAperture56_a_int = cms.double(-31.9),
          empiricalAperture56_a_slp = cms.double(1.323),
  )
)

_config_2017_postTS2 = cms.PSet(
  opticalFunctionConfig = cms.PSet(
          es_source = cms.PSet(
                  validityRange = cms.EventRange("0:min - 999999:max"),
                  opticalFunctions = cms.VPSet(
                          cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version4/120urad.root") ),
                          cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version4/130urad.root") ),
                          cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version4/140urad.root") )
                  )
           ),
          defaultCrossingAngle = cms.double(140.0)
  ),
  optics_parameters = cms.PSet(
          empiricalAperture45_xi0_int = cms.double(0.073),
          empiricalAperture45_xi0_slp = cms.double(4.107E-04),
          empiricalAperture45_a_int = cms.double(39.0),
          empiricalAperture45_a_slp = cms.double(0.768),
          empiricalAperture56_xi0_int = cms.double(0.067),
          empiricalAperture56_xi0_slp = cms.double(6.868E-04),
          empiricalAperture56_a_int = cms.double(-50.2),
          empiricalAperture56_a_slp = cms.double(1.740),
  )
)

_config_2018 = cms.PSet(
  opticalFunctionConfig = cms.PSet(
          es_source = cms.PSet(
                  validityRange = cms.EventRange("0:min - 999999:max"),
                  opticalFunctions = cms.VPSet(
                          cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version4/120urad.root") ),
                          cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version4/130urad.root") ),
                          cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version4/140urad.root") )
                  )
          ),
          defaultCrossingAngle = cms.double(140.0)
  ),
  optics_parameters = cms.PSet(
          empiricalAperture45_xi0_int = cms.double(0.079),
          empiricalAperture45_xi0_slp = cms.double(4.211E-04),
          empiricalAperture45_a_int = cms.double(42.8),
          empiricalAperture45_a_slp = cms.double(0.669),
          empiricalAperture56_xi0_int = cms.double(0.074),
          empiricalAperture56_xi0_slp = cms.double(6.604E-04),
          empiricalAperture56_a_int = cms.double(-22.7),
          empiricalAperture56_a_slp = cms.double(1.600),
  )
)

_config_2021 = cms.PSet(
  opticalFunctionConfig = cms.PSet(
          es_source = cms.PSet(
                  validityRange = cms.EventRange("0:min - 999999:max"),
                  opticalFunctions = cms.VPSet(
                          cms.PSet( xangle = cms.double(110.444), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2021/version_pre3/110.444urad.root") ),
                          cms.PSet( xangle = cms.double(184.017), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2021/version_pre3/184.017urad.root") )
                  )
          ),
          defaultCrossingAngle = cms.double(0.0)
  ),
  optics_parameters = cms.PSet(
         empiricalAperture45_xi0_int = cms.double(0.079),
          empiricalAperture45_xi0_slp = cms.double(4.211E-04),
          empiricalAperture45_a_int = cms.double(42.8),
          empiricalAperture45_a_slp = cms.double(0.669),
          empiricalAperture56_xi0_int = cms.double(0.074),
          empiricalAperture56_xi0_slp = cms.double(6.604E-04),
          empiricalAperture56_a_int = cms.double(-22.7),
          empiricalAperture56_a_slp = cms.double(1.600)
  )
)

_opticalfunctionsTransportSetup_2016_preTS2 =cms.PSet(
                opticalFunctionConfig = _config_2016_preTS2.opticalFunctionConfig,
                optics_parameters =  cms.PSet(_baseOpticalFunctionsParameters,_config_2016_preTS2.optics_parameters)
)

_opticalfunctionsTransportSetup_2016_postTS2 =cms.PSet(
                opticalFunctionConfig = _config_2016_postTS2.opticalFunctionConfig,
                optics_parameters = cms.PSet(_baseOpticalFunctionsParameters,_config_2016_postTS2.optics_parameters)
)

_opticalfunctionsTransportSetup_2017_preTS2 =cms.PSet(
                opticalFunctionConfig = _config_2017_preTS2.opticalFunctionConfig,
                optics_parameters = cms.PSet(_baseOpticalFunctionsParameters,_config_2017_preTS2.optics_parameters)
)

_opticalfunctionsTransportSetup_2017_postTS2 =cms.PSet(
                opticalFunctionConfig = _config_2017_postTS2.opticalFunctionConfig,
                optics_parameters = cms.PSet(_baseOpticalFunctionsParameters,_config_2017_postTS2.optics_parameters)
)

opticalfunctionsTransportSetup_2018 =cms.PSet(
                _baseOpticalFunctionsParameters,
                _config_2018.opticalFunctionConfig,
                _config_2018.optics_parameters
)

opticalfunctionsTransportSetup_2021 =cms.PSet(
                _baseOpticalFunctionsParameters,
                _config_2021.opticalFunctionConfig,
                _config_2021.optics_parameters
)

# Default setup
opticalfunctionsTransportSetup_2016 = _opticalfunctionsTransportSetup_2016_preTS2.clone()
opticalfunctionsTransportSetup_2017 = _opticalfunctionsTransportSetup_2017_preTS2.clone()
