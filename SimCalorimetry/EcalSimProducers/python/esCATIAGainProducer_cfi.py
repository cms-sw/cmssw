import FWCore.ParameterSet.Config as cms

EcalCATIAGainRatiosRcd =  cms.ESSource("EmptyESSource",
                                recordName = cms.string("EcalCATIAGainRatiosRcd"),
                                firstValid = cms.vuint32(1),
                                iovIsRunNotTime = cms.bool(True)
                                )

EcalCATIAGainRatios = cms.ESProducer("EcalCATIAGainRatiosESProducer",
                                     ComponentName = cms.string('EcalCatiaGainProducer'),
                                     CATIAGainRatio = cms.double(10.))

# foo bar baz
# iQVCfDlt5U1Z6
# eX6lWMqpCRLjN
