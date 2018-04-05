def customiseForMaterialAnalyser_ForPhaseII(process):
  """Extend eta range of the generator to better cover the material map"""
  process.generator.PGunParameters.MaxEta = 4.5
  process.generator.PGunParameters.MinEta = -4.5
  return process
