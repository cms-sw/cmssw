def customiseForMaterialAnalyser(process):
  process.load("Validation.Geometry.trackingRecoMaterialAnalyzer_cfi")
  if getattr(process, 'schedule'):
    process.schedule.append(process.materialDumper_step)
  return process
