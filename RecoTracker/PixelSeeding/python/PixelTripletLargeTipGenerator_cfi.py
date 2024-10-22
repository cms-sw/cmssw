import FWCore.ParameterSet.Config as cms

# moving to the block.  Will delete the PSet once transition is done
PixelTripletLargeTipGenerator = cms.PSet(
    maxElement = cms.uint32(100000),
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    ComponentName = cms.string('PixelTripletLargeTipGenerator'),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3),
    extraHitRPhitolerance = cms.double(0.032),
    extraHitRZtolerance = cms.double(0.037)
)
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
for e in [peripheralPbPb, pp_on_XeXe_2017, pp_on_AA]:
    e.toModify(PixelTripletLargeTipGenerator, maxElement = 1000000)
