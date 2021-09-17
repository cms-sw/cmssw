import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import *

#scalers = cms.EDProducer('ScalersRawToDigi')
BeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer")

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(onlineBeamSpotProducer, useTransientRecord = True)

onlineBeamSpot = cms.Sequence( onlineBeamSpotProducer )

