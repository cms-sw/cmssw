import FWCore.ParameterSet.Config as cms
import RecoVertex.BeamSpotProducer.beamSpotOnlineProducer_cfi as _mod
onlineBeamSpotProducer = _mod.beamSpotOnlineProducer.clone(
                                        src = 'scalersRawToDigi',
                                        setSigmaZ = -1, #negative value disables it.
                                        gtEvmLabel = 'gtEvmDigis'
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(onlineBeamSpotProducer, useTransientRecord = True)
