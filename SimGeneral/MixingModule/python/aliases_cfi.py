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

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toModify(simEcalUnsuppressedDigis,
                           mix = cms.VPSet(
                               cms.PSet(type = cms.string('EBDigiCollectionPh2'))
                           )
)


simHcalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('HBHEDataFramesSorted')),
      cms.PSet(type = cms.string('HFDataFramesSorted')),
      cms.PSet(type = cms.string('HODataFramesSorted')),
      cms.PSet(type = cms.string('ZDCDataFramesSorted')),
      cms.PSet(type = cms.string('QIE10DataFrameHcalDataFrameContainer')),
      cms.PSet(type = cms.string('QIE11DataFrameHcalDataFrameContainer'))
    )
)
_pixelCommon = cms.VPSet(
    cms.PSet(type = cms.string('PixelDigiedmDetSetVector')),
    cms.PSet(type = cms.string('PixelDigiSimLinkedmDetSetVector'))
)
simSiPixelDigis = cms.EDAlias(
    mix = _pixelCommon
) 
simSiStripDigis = cms.EDAlias(
    mix = cms.VPSet(
      cms.PSet(type = cms.string('SiStripDigiedmDetSetVector')),
      cms.PSet(type = cms.string('SiStripRawDigiedmDetSetVector')),
      cms.PSet(type = cms.string('StripDigiSimLinkedmDetSetVector'))
    )
)
simHGCalUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
        cms.PSet(
            type = cms.string("DetIdHGCSampleHGCDataFramesSorted"),
            fromProductInstance = cms.string("HGCDigisEE"),
            toProductInstance = cms.string("EE"),
        ),
        cms.PSet(
            type = cms.string("DetIdHGCSampleHGCDataFramesSorted"),
            fromProductInstance = cms.string("HGCDigisHEfront"),
            toProductInstance = cms.string("HEfront"),
        ),
        cms.PSet(
            type = cms.string("DetIdHGCSampleHGCDataFramesSorted"),
            fromProductInstance = cms.string("HGCDigisHEback"),
            toProductInstance = cms.string("HEback"),
        ),
    )
)
simHFNoseUnsuppressedDigis = cms.EDAlias(
    mix = cms.VPSet(
        cms.PSet(
            type = cms.string("DetIdHGCSampleHGCDataFramesSorted"),
            fromProductInstance = cms.string("HFNoseDigis"),
            toProductInstance = cms.string("HFNose"),
        ),
    )
)

simAPVsaturation = cms.EDAlias(
    mix = cms.VPSet(
        cms.PSet(type = cms.string('bool'))
    )
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(simCastorDigis, mix = None)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
(~phase2_hgcal).toModify(simHGCalUnsuppressedDigis, mix = None)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
(premix_stage1 & phase2_hgcal).toModify(simHGCalUnsuppressedDigis,
    mix = {
        0 : dict(type = "PHGCSimAccumulator"),
        1 : dict(type = "PHGCSimAccumulator"),
        2 : dict(type = "PHGCSimAccumulator"),
    }
)

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
(~phase2_hfnose).toModify(simHFNoseUnsuppressedDigis, mix = None)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(simSiPixelDigis, mix = _pixelCommon + [cms.PSet(type = cms.string('PixelFEDChanneledmNewDetSetVector'))])

# no castor,pixel,strip digis in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(simCastorDigis, mix = None)
fastSim.toModify(simSiPixelDigis, mix = None)
fastSim.toModify(simSiStripDigis, mix = None)
fastSim.toModify(simAPVsaturation, mix = None)
