

import FWCore.ParameterSet.Config as cms

L1CaloTriggerSetupSource = cms.ESSource("EmptyESSource",
                                        recordName = cms.string('L1CaloTriggerSetupRcd'),
                                        firstValid = cms.vuint32(1),
                                        iovIsRunNotTime = cms.bool(True)
                                        )

L1CaloTriggerSetup = cms.ESProducer("L1CaloTriggerSetupProducer",
                                    InputXMLFile = cms.FileInPath('SLHCUpgradeSimulations/L1Trigger/data/setup40.xml')
                                    )

