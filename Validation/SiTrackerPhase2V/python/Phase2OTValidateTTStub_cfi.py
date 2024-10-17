import FWCore.ParameterSet.Config as cms
import math

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
Phase2OTValidateTTStub = DQMEDAnalyzer('Phase2OTValidateTTStub',
    TopFolderName = cms.string('TrackerPhase2OTStubV'),
    TTStubs = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),
    trackingParticleToken = cms.InputTag("mix", "MergedTrackTruth"),
    MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
    MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"),
    TP_minNStub = cms.int32(4),
    TP_minNLayersStub = cms.int32(4),
    TP_minPt = cms.double(2.0),
    TP_maxEta = cms.double(2.4),
    TP_maxVtxZ = cms.double(15.0)
)

# Apply premix stage2 modifications if necessary
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(Phase2OTValidateTTStub, trackingParticleToken = "mixData:MergedTrackTruth")