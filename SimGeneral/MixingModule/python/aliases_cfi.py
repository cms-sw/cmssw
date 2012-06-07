import FWCore.ParameterSet.Config as cms

simCastorDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('CastorDataFramesSorted'))
    )
)
simEcalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('EBDigiCollection')),
      cms.PSet(type = cms.string('EEDigiCollection')),
      cms.PSet(type = cms.string('ESDigiCollection'))
    )
)
simHcalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('HBHEDataFramesSorted')),
      cms.PSet(type = cms.string('HFDataFramesSorted')),
      cms.PSet(type = cms.string('HODataFramesSorted')),
      cms.PSet(type = cms.string('ZDCDataFramesSorted'))
    )
)
simSiPixelDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('PixelDigiedmDetSetVector')),
      cms.PSet(type = cms.string('PixelDigiSimLinkedmDetSetVector'))
    )
)
simSiStripDigis = cms.EDAlias(
    simSiStripDigiSimLink = cms.VPSet(
      cms.PSet(type = cms.string('StripDigiSimLinkedmDetSetVector'))
    ),
    mix = cms.VPSet(
      cms.PSet(type = cms.string('SiStripDigiedmDetSetVector')),
      cms.PSet(type = cms.string('SiStripRawDigiedmDetSetVector'))
    )
)
mergedtruth = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('TrackingParticles')),
      cms.PSet(type = cms.string('TrackingVertexs'))
    )
)
