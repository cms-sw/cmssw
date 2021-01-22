import FWCore.ParameterSet.Config as cms

EcalLiteDTUPedestalsRcd =  cms.ESSource("EmptyESSource",
                                recordName = cms.string("EcalLiteDTUPedestalsRcd"),
                                firstValid = cms.vuint32(1),
                                iovIsRunNotTime = cms.bool(True)
                                )

EcalLiteDTUPedestals = cms.ESProducer(
    "EcalLiteDTUPedestalsESProducer",
    ComponentName = cms.string('EcalLiteDTUPedestalProducer'),
    MeanPedestalsGain10 = cms.double(12),
    RMSPedestalsGain10  = cms.double(2.5),
    MeanPedestalsGain1  = cms.double(12.),
    RMSPedestalsGain1   = cms.double(2.)
)
