from RecoTracker.TkSeedGenerator.multiHitFromChi2EDProducerDefault_cfi import multiHitFromChi2EDProducerDefault as _multiHitFromChi2EDProducerDefault
multiHitFromChi2EDProducer = _multiHitFromChi2EDProducerDefault.clone()

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
for e in [peripheralPbPb, pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify(multiHitFromChi2EDProducer, maxElement = 1000000)
