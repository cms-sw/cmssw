import FWCore.ParameterSet.Config as cms

inclusiveVertexFinder  = cms.EDProducer("InclusiveVertexFinder",
       beamSpot = cms.InputTag("offlineBeamSpot"),
       primaryVertices = cms.InputTag("offlinePrimaryVertices"),
       tracks = cms.InputTag("generalTracks"),
       minHits = cms.uint32(8),
       maximumLongitudinalImpactParameter = cms.double(0.3),
       maximumTimeSignificance = cms.double(3.0),
       minPt = cms.double(0.8),
       maxNTracks = cms.uint32(30),

       clusterizer = cms.PSet(
           seedMax3DIPSignificance = cms.double(9999.),
           seedMax3DIPValue = cms.double(9999.),
           seedMin3DIPSignificance = cms.double(1.2),
           seedMin3DIPValue = cms.double(0.005),
           clusterMaxDistance = cms.double(0.05), #500um
           clusterMaxSignificance = cms.double(4.5), #4.5 sigma
           distanceRatio = cms.double(20), # was cluster scale = 1 / density factor =0.05 
           clusterMinAngleCosine = cms.double(0.5), # only forward decays
           maxTimeSignificance = cms.double(3.5) #3.5 sigma, since the time cut is track-to-track
       ),

       vertexMinAngleCosine = cms.double(0.95), # scalar prod direction of tracks and flight dir
       vertexMinDLen2DSig = cms.double(2.5), #2.5 sigma
       vertexMinDLenSig = cms.double(0.5), #0.5 sigma
       fitterSigmacut =  cms.double(3),
       fitterTini = cms.double(256),
       fitterRatio = cms.double(0.25),
       useDirectVertexFitter = cms.bool(True),
       useVertexReco  = cms.bool(True),
       vertexReco = cms.PSet(
               finder = cms.string('avr'),
               primcut = cms.double(1.0),
               seccut = cms.double(3),
               smoothing = cms.bool(True)
       )


)


