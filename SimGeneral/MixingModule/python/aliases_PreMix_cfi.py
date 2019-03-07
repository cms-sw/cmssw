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
simHGCalUnsuppressedDigis = cms.EDAlias()
simHFNoseUnsuppressedDigis = cms.EDAlias()
_pixelCommon = cms.VPSet(
    cms.PSet(type = cms.string('PixelDigiedmDetSetVector')),
    cms.PSet(type = cms.string('PixelDigiSimLinkedmDetSetVector'))
)
simSiPixelDigis = cms.EDAlias(
    mix = _pixelCommon + [cms.PSet(type = cms.string('PixelFEDChanneledmNewDetSetVector'))]
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

genPUProtons = cms.EDAlias(
    mixData = cms.VPSet(
        cms.PSet( type = cms.string('recoGenParticles') )
    )
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(simCastorDigis, mix = None)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(simSiPixelDigis, mix = _pixelCommon) 

# no castor,pixel,strip digis in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(simCastorDigis, mix = None)
fastSim.toModify(simSiPixelDigis, mix = None)
fastSim.toModify(simSiStripDigis, mix = None)
