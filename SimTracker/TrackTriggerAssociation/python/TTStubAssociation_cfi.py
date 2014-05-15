import FWCore.ParameterSet.Config as cms

TTStubAssociatorFromPixelDigis = cms.EDProducer("TTStubAssociator_PixelDigi_",
    TTStubs = cms.VInputTag( cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
                             cms.InputTag("TTStubsFromPixelDigis", "StubRejected"),
    ),
    TTClusterTruth = cms.VInputTag( cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                    cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"),
    ) # NOTE: the two vectors of input tags must be of the same size and in the correct order
      # as the producer would run on a specific pair and return a big error message
      # if the vectors are uncorrectly dimensioned
      # so "StubAccepted" needs the "ClusterAccepted" MC truth, and "StubRejected" the "ClusterInclusive" MC truth
)

