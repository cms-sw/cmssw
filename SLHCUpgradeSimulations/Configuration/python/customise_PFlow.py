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

def customise_phase1ElectronHOverE( process ) :
    if hasattr( process, "ecalDrivenElectronSeeds" ) :
        process.ecalDrivenElectronSeeds.SeedConfiguration.hOverEMethodBarrel = 0 # 0 = cone #1 = single tower #2 = towersBehindCluster #3 = clusters (max is 4) 
        process.ecalDrivenElectronSeeds.SeedConfiguration.hOverEMethodEndcap = 1 # 0 = cone #1 = single tower #2 = towersBehindCluster #3 = clusters (max is 4)
        process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEBarrel = 0.15 
        process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEEndcaps = 0.1 
        process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEOuterEndcaps = 0.2
    return process

def customise_shashlikElectronHOverE( process ) :
    if hasattr( process, "ecalDrivenElectronSeeds" ) :
        process.ecalDrivenElectronSeeds.SeedConfiguration.hOverEMethodBarrel = 0 # 0 = cone #1 = single tower #2 = towersBehindCluster #3 = clusters (max is 4) 
        process.ecalDrivenElectronSeeds.SeedConfiguration.hOverEMethodEndcap = 3 # 0 = cone #1 = single tower #2 = towersBehindCluster #3 = clusters (max is 4)
        process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEBarrel = 0.15 
        process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEEndcaps = 0.5 
        process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEOuterEndcaps = 1.0
    if hasattr(process,'ecalDrivenGsfElectrons'):
        process.ecalDrivenGsfElectrons.hOverEMethodEndcap = cms.int32(3)
        process.ecalDrivenGsfElectrons.maxHOverEEndcaps = cms.double(99999.)
    if hasattr(process,'gsfElectrons'):
        process.gsfElectrons.hOverEMethodEndcap = cms.int32(3)
        process.gsfElectrons.maxHOverEEndcaps = cms.double(99999.)
    return process

def customise_HGCalElectronHOverE( process ) :
    if hasattr( process, "ecalDrivenElectronSeeds" ) :
        process.ecalDrivenElectronSeeds.SeedConfiguration.hOverEMethodBarrel = 0 # 0 = cone #1 = single tower #2 = towersBehindCluster #3 = clusters (max is 4) 
        process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEBarrel = 0.15 
    return process
