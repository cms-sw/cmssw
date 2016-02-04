import FWCore.ParameterSet.Config as cms

inclusiveVertexFinder  = cms.EDProducer("InclusiveVertexFinder",
       beamSpot = cms.InputTag("offlineBeamSpot"),
       primaryVertices = cms.InputTag("offlinePrimaryVertices"),
       tracks = cms.InputTag("generalTracks"),
       minHits = cms.uint32(7),
       minPt = cms.double(0.8),
       seedMin3DIPSignificance = cms.double(1.5),
       seedMin3DIPValue = cms.double(0.005),
       clusterMaxDistance = cms.double(0.05), #500um
       clusterMaxSignificance = cms.double(3.0), #3 sigma
       clusterScale = cms.double(10), #10x the IP
       clusterMinAngleCosine = cms.double(0.0), # only forward decays
       vertexMinAngleCosine = cms.double(0.98), # scalar prod direction of tracks and flight dir
       vertexMinDLen2DSig = cms.double(2.5), #2.5 sigma
       vertexMinDLenSig = cms.double(0.5), #0.5 sigma

       vertexReco = cms.PSet(
               finder = cms.string('avr'),
               primcut = cms.double(1.0),
               seccut = cms.double(3),
               smoothing = cms.bool(True)
       )


)


