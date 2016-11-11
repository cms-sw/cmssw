import FWCore.ParameterSet.Config as cms

simCastorDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('CastorDataFramesSorted'))
    )
)
simEcalUnsuppressedDigis = cms.EDAlias()
#    mix = cms.VPSet(
#      cms.PSet(type = cms.string('EBDigiCollection')),
#      cms.PSet(type = cms.string('EEDigiCollection')),
#      cms.PSet(type = cms.string('ESDigiCollection'))
#    )
#)
simHcalUnsuppressedDigis = cms.EDAlias()
#    mix = cms.VPSet(
#      cms.PSet(type = cms.string('HBHEDataFramesSorted')),
#      cms.PSet(type = cms.string('HFDataFramesSorted')),
#      cms.PSet(type = cms.string('HODataFramesSorted')),
#      cms.PSet(type = cms.string('ZDCDataFramesSorted'))
#    )
#)
simSiPixelDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('PixelDigiedmDetSetVector')),
      cms.PSet(type = cms.string('PixelDigiSimLinkedmDetSetVector'))
    )
)
simSiStripDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('SiStripDigiedmDetSetVector')),
      cms.PSet(type = cms.string('SiStripRawDigiedmDetSetVector')),
      cms.PSet(type = cms.string('StripDigiSimLinkedmDetSetVector'))
    )
)
#mergedtruth = cms.EDAlias(
#    mix = cms.VPSet(
#      cms.PSet(type = cms.string('TrackingParticles')),
#      cms.PSet(type = cms.string('TrackingVertexs'))
#    )
#)

# no castor,pixel,strip digis in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(simCastorDigis, mix = None)
fastSim.toModify(simSiPixelDigis, mix = None)
fastSim.toModify(simSiStripDigis, mix = None)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(simCastorDigis, mix = None)
