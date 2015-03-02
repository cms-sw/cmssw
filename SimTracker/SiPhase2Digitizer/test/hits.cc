#include "SimTracker/SiPhase2Digitizer/test/hits.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

ValHitsCollection ValHitsBuilder(const TrackerGeometry* tkGeom, edm::DetSetVector< PixelClusterSimLink >* clusterLinks) {

    ValHitsCollection hits;

    edm::DetSetVector< PixelClusterSimLink >::const_iterator DSViter;
    edm::DetSet< PixelClusterSimLink >::const_iterator clusterLink;

    for (DSViter = clusterLinks->begin(); DSViter != clusterLinks->end(); ++DSViter) {

        // Get the geometry of the tracker
        DetId detId(DSViter->detId());
        const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(detId);
        const PixelGeomDetUnit* pixDet = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
        if (!pixDet) assert(0);

        // Loop over cluster links
        for (clusterLink = DSViter->data.begin(); clusterLink != DSViter->data.end(); ++clusterLink) {

            PixelClusterSimLink link = *clusterLink;

            // Get the cluster
            edm::Ref< edmNew::DetSetVector< SiPixelCluster >, SiPixelCluster > const& cluster = link.getCluster();

            // Create the Hit
            ValHit newHit;

            // Set the cluster position
            MeasurementPoint mp(cluster->x(), cluster->y());
            newHit.localPos = geomDetUnit->topology().localPosition(mp);
            newHit.globalPos = geomDetUnit->surface().toGlobal(newHit.localPos);

            // Set the error
            newHit.xx = newHit.xy = newHit.yy = -1;

            // Add the simTracks and the reference
            newHit.simTracks = link.getSimTracks();
            newHit.cluster = link.getCluster();

            // Add the Hit
            hits[DSViter->detId()].push_back(newHit);
        }
    }

    return hits;
}

ValHitsCollection ValHitsBuilder(const TrackerGeometry* tkGeom, edm::DetSetVector< PixelClusterSimLink >* clusterLinks, edmNew::DetSetVector< SiPixelRecHit >* recHits) {

    ValHitsCollection hits;

    edmNew::DetSetVector< SiPixelRecHit >::const_iterator DSViter;
    edmNew::DetSet< SiPixelRecHit >::const_iterator rechHitIter;

    edm::DetSet< PixelClusterSimLink >::const_iterator clusterLink;

    for (DSViter = recHits->begin(); DSViter != recHits->end(); ++DSViter) {

        // Get the geometry of the tracker
        DetId detId(DSViter->detId());
        const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(detId);

        // Get the cluster simlinks DetSet
        edm::DetSet< PixelClusterSimLink > clusters = (*clusterLinks)[DSViter->detId()];

        // Loop over recHits
        for (rechHitIter = DSViter->begin(); rechHitIter != DSViter->end(); ++rechHitIter) {

            // Get the cluster
            edm::Ref< edmNew::DetSetVector< SiPixelCluster >, SiPixelCluster > const& cluster = rechHitIter->cluster();

            // Create the Hit
            ValHit newHit;

            // Set the recHit position
            newHit.localPos = rechHitIter->localPosition();

            MeasurementPoint mp(rechHitIter->localPosition().x(), rechHitIter->localPosition().y());
            newHit.globalPos = geomDetUnit->surface().toGlobal(geomDetUnit->topology().localPosition(mp));

            // Set the Error
            newHit.xx = rechHitIter->localPositionError().xx();
            newHit.xy = rechHitIter->localPositionError().xy();
            newHit.yy = rechHitIter->localPositionError().yy();

            bool clusterFound = false;

            // Loop over the clusters
            for (clusterLink = clusters.begin(); clusterLink != clusters.end(); ++clusterLink) {
                PixelClusterSimLink link = *clusterLink;

                edm::Ref< edmNew::DetSetVector< SiPixelCluster >, SiPixelCluster > const& clusterFromLink = link.getCluster();

                // Compare the clusters
                if (cluster == clusterFromLink) {

                    // Add the simTracks and the reference
                    newHit.simTracks = link.getSimTracks();
                    newHit.cluster = link.getCluster();

                    clusterFound = true;
                    break;
                }
            }

            // Add the Hit
            if (clusterFound) hits[DSViter->detId()].push_back(newHit);

        }
    }

    return hits;
}
