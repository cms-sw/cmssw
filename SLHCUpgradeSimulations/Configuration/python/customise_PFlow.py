import FWCore.ParameterSet.Config as cms

def customise_extendedTrackerBarrel( process ) :
    if hasattr(process,'reconstruction'):
        for link in process.particleFlowBlock.linkDefinitions:
            if hasattr(link,'trackerEtaBoundary') : link.trackerEtaBoundary = cms.double(3.0)
        for importer in process.particleFlowBlock.elementImporters :
            if importer.source.value()=="particleFlowClusterHFEM" : importer.importerName = cms.string("ClusterImporterForForwardTracker")
            if importer.source.value()=="particleFlowClusterHFHAD" : importer.importerName = cms.string("ClusterImporterForForwardTracker")
    return process

def customise_use3DHCalClusters( process ) :
    if hasattr(process,'reconstruction'):
        # I'm not sure if removing entries while iterating affects the iteration, so I'll be safe
        # and find everything I need to delete first, then loop again to remove them.
        if hasattr(process,'pfClusteringHCAL'):
            process.pfClusteringHCAL=  cms.Sequence( process.particleFlowRecHitHBHEHO      +
                                                     process.particleFlowRecHitHF          +
                                                     process.particleFlowClusterHCALSemi3D +
                                                     process.particleFlowClusterHF           )
        thingsToDelete=[]
        for importer in process.particleFlowBlock.elementImporters :
            if importer.source.value()=="particleFlowClusterHCAL" : thingsToDelete.append( importer )
            if importer.source.value()=="particleFlowClusterHO" : thingsToDelete.append( importer )
            if importer.source.value()=="particleFlowClusterHFEM" : thingsToDelete.append( importer )
            if importer.source.value()=="particleFlowClusterHFHAD" : thingsToDelete.append( importer )
        for importer in thingsToDelete :
            process.particleFlowBlock.elementImporters.remove( importer )

        # Now that I've removed the old entries, I can add the new ones
        process.particleFlowBlock.elementImporters.append( cms.PSet( importerName = cms.string("GenericClusterImporter"), 
                                                                     source = cms.InputTag("particleFlowClusterHCALSemi3D") ) )
        process.particleFlowBlock.elementImporters.append( cms.PSet( importerName = cms.string("GenericClusterImporter"), 
                                                                     source = cms.InputTag("particleFlowClusterHF") ) )

        if hasattr(process,'tcMetWithPFclusters'):
            process.tcMetWithPFclusters.PFClustersHCAL = cms.InputTag("particleFlowClusterHCALSemi3D")
            process.tcMetWithPFclusters.PFClustersHFEM = cms.InputTag("particleFlowClusterHF")
            process.tcMetWithPFclusters.PFClustersHFHAD = cms.InputTag("particleFlowClusterHF")
    return process

