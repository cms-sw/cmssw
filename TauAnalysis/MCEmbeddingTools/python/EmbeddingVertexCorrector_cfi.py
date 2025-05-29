import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from SimGeneral.MixingModule.digitizers_cfi import theDigitizers
from IOMC.EventVertexGenerators.VtxSmearedRealistic_cfi import VtxSmeared
VtxCorrectedToInput = cms.EDProducer(
    "EmbeddingVertexCorrector", src=cms.InputTag("generator", "unsmeared")
)
(run2_common | run3_common).toReplaceWith(VtxSmeared, VtxCorrectedToInput)

(run2_common & ~run3_common).toModify(
    theDigitizers, castor={"doNoise": cms.bool(False)}
)
(run2_common | run3_common).toModify(
    theDigitizers,
    ecal={"doESNoise": cms.bool(False), "doENoise": cms.bool(False)},
    hcal={
        "doNoise": cms.bool(False),
        "doThermalNoise": cms.bool(False),
        "doHPDNoise": cms.bool(False),
    },
    pixel={"AddNoisyPixels": cms.bool(False), "AddNoise": cms.bool(False)},
    strip={"Noise": cms.bool(False)},
)
