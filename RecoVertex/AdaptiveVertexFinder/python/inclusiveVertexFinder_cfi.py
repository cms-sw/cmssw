import FWCore.ParameterSet.Config as cms

inclusiveVertexFinder  = cms.EDProducer("InclusiveVertexFinder",
       beamSpot = cms.InputTag("offlineBeamSpot"),
       primaryVertices = cms.InputTag("offlinePrimaryVertices"),
       tracks = cms.InputTag("generalTracks"),
       minHits = cms.uint32(8),
       maximumLongitudinalImpactParameter = cms.double(0.3),
       minPt = cms.double(0.8),
       maxNTracks = cms.uint32(30),

       clusterizer = cms.PSet(
           seedMin3DIPSignificance = cms.double(1.2),
           seedMin3DIPValue = cms.double(0.005),
           clusterMaxDistance = cms.double(0.05), #500um
           clusterMaxSignificance = cms.double(4.5), #4.5 sigma
           clusterScale = cms.double(1), 
           clusterMinAngleCosine = cms.double(0.5), # only forward decays
       ),

       vertexMinAngleCosine = cms.double(0.95), # scalar prod direction of tracks and flight dir
       vertexMinDLen2DSig = cms.double(2.5), #2.5 sigma
       vertexMinDLenSig = cms.double(0.5), #0.5 sigma
       vertexReco = cms.PSet(
               finder = cms.string('avr'),
               primcut = cms.double(1.0),
               seccut = cms.double(3),
               smoothing = cms.bool(True)
       )


)


