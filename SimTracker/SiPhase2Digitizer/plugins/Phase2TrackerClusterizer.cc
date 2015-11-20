#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerClusterizer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>
#include <iostream>

namespace cms {

    /*
     * Initialise the producer
     */ 

    Phase2TrackerClusterizer::Phase2TrackerClusterizer(edm::ParameterSet const& conf) :
        conf_(conf),
        src_(conf.getParameter< edm::InputTag >("src")),
        maxClusterSize_(conf.getParameter< unsigned int >("maxClusterSize")),
        maxNumberClusters_(conf.getParameter< unsigned int >("maxNumberClusters")) {
            // Objects that will be produced
            produces< Phase2TrackerCluster1DCollectionNew >(); 
            // 
            clusterizer_  = new Phase2TrackerClusterizerAlgorithm(maxClusterSize_, maxNumberClusters_);
        }

    Phase2TrackerClusterizer::~Phase2TrackerClusterizer() {
        delete clusterizer_;
    }

    /*
     * Clusterize the events
     */

    void Phase2TrackerClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {

        // Get the Digis
        edm::Handle< edm::DetSetVector< Phase2TrackerDigi > > digis;
        event.getByLabel(src_, digis);
        //event.getByLabel("mix", "Tracker", digis); 
        
        // Get the geometry
        edm::ESHandle< TrackerGeometry > geomHandle;
        eventSetup.get< TrackerDigiGeometryRecord >().get(geomHandle);
        const TrackerGeometry* tkGeom(&(*geomHandle)); 

        // Global container for the clusters of each modules
        std::auto_ptr< Phase2TrackerCluster1DCollectionNew > outputClusters(new Phase2TrackerCluster1DCollectionNew());

        // Go over all the modules
        for (edm::DetSetVector< Phase2TrackerDigi >::const_iterator DSViter = digis->begin(); DSViter != digis->end(); ++DSViter) {

            DetId detId(DSViter->detId());

            // Geometry
            const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
            const PixelGeomDetUnit* pixDet = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
            if (!pixDet) assert(0);

            // Container for the clusters that will be produced for this modules
            edmNew::DetSetVector< Phase2TrackerCluster1D >::FastFiller clusters(*outputClusters, DSViter->detId());

            // Setup the clusterizer algorithm for this detector (see ClusterizerAlgorithm for more details)
            clusterizer_->setup(pixDet);

            // Pass the list of Digis to the main algorithm
            // This function will store the clusters in the previously created container
            clusterizer_->clusterizeDetUnit(*DSViter, clusters);

            if (clusters.empty()) clusters.abort();
        }

        // Add the data to the output
        event.put(outputClusters);
    }
}

using cms::Phase2TrackerClusterizer;
DEFINE_FWK_MODULE(Phase2TrackerClusterizer);
