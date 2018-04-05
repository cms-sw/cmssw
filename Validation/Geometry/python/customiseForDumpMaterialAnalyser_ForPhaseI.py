def customiseForMaterialAnalyser_ForPhaseI(process):
  """Extend eta range of the generator to better cover the material map"""
  process.generator.PGunParameters.MaxEta = 3.0
  process.generator.PGunParameters.MinEta = -3.0
  return process
