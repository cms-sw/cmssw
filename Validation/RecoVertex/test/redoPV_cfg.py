import FWCore.ParameterSet.Config as cms

# example of redoing the Primary vertex reconstruction before analyzing

process = cms.Process("redoPV")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V25B_356ReReco-v1/0004/0E72CE54-F43B-DF11-A06F-0026189438BD.root')
)


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.GlobalTag.globaltag= "START3X_V25B::All"
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#process.load("Configuration.EventContent.EventContent_cff")


process.load("RecoVertex.Configuration.RecoVertex_cff")


from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesWithBS_cfi import *

# new squence (if needed)
offlinePrimaryVerticesDA=offlinePrimaryVertices.clone()
process.vertexreco = cms.Sequence(offlinePrimaryVertices*offlinePrimaryVerticesWithBS*offlinePrimaryVerticesDA)


# new parameters (if needed)


# track selection, common for all producers here (doesn't have to be)
TkFilterParameters=cms.PSet(
    minPt = cms.double(0.0),                   # direct pt cut
    maxD0Significance = cms.double(5.0),       # impact parameter significance
    maxNormalizedChi2 = cms.double(5.0),       # loose cut on track chi**2
    minPixelLayersWithHits = cms.int32(2),     # two or more pixel layers
    minSiliconLayersWithHits = cms.int32(5),   # five or more tracker layers (includes pixels)
    trackQuality = cms.string("any")           # track quality not used
    )


# offlinePrimaryVertices   gap clustering with unconstrained fit
process.offlinePrimaryVertices.verbose = cms.untracked.bool(False)            
process.offlinePrimaryVertices.TkFilterParameters=TkFilterParameters
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("generalTracks")
process.offlinePrimaryVertices.TkClusParameters=cms.PSet(algorithm=cms.string('gap'),
                                                         TkGapClusParameters=cms.PSet(zSeparation = cms.double(0.2))) # 2 mm separation
process.offlinePrimaryVertices.vertexCollections[0].minNdof  = cms.double(0.0)     # no contraint: ndof = 2 * sum(weights) - 3
process.offlinePrimaryVertices.vertexCollections[1].minNdof  = cms.double(2.0)     # w/constraint: ndof = 2 * sum(weights)


# Test
process.offlinePrimaryVerticesDA.verbose = cms.untracked.bool(False)
process.offlinePrimaryVerticesDA.TkFilterParameters=TkFilterParameters
process.offlinePrimaryVerticesDA.TrackLabel = cms.InputTag("generalTracks")
process.offlinePrimaryVerticesDA.TkClusParameters=cms.PSet(
    algorithm=cms.string('DA'),
    TkDAClusParameters = cms.PSet(
      coolingFactor = cms.double(0.8),  #  slow annealing
      Tmin = cms.double(4.0),           #  freezeout temperature
      vertexSize = cms.double(0.05)     #  ~ resolution / sqrt(Tmin)
    )
    
process.offlinePrimaryVerticesDA.vertexCollections =  cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               )]
)




# the analyzer 
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoVertex.PrimaryVertexAnalyzer4PU_cfi") 
process.vertexAnalysis.vertexCollections=cms.untracked.vstring(["offlinePrimaryVertices","offlinePrimaryVerticesDA"]) 
 

process.p = cms.Path(process.vertexreco*process.vertexAnalysis)

