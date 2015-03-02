#include "SimTracker/SiPhase2Digitizer/plugins/SiPhase2Clusterizer.h"
#include "SimTracker/SiPhase2Digitizer/interface/ClusterizerAlgorithm.h"
//#include "SimTracker/SiPhase2Digitizer/interface/PixelClusterSimLink.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

namespace cms {

    SimTrackerSiPhase2Clusterizer::SimTrackerSiPhase2Clusterizer(edm::ParameterSet const& conf) :
        conf_(conf),
        src_(conf.getParameter< edm::InputTag >("src")),
        maxClusterSize_(conf.getParameter< int >("maxClusterSize")),
        maxNumberClusters_(conf.getParameter< int >("maxNumberClusters")),
        generateClusterSimLink_(conf.getParameter< bool >("clusterSimLink")) {

        // Objects that will be produced
        produces< SiPixelClusterCollectionNew >(); 

        // Optionally produce simlinks
        //if (generateClusterSimLink_) produces< edm::DetSetVector<PixelClusterSimLink> >();

        // Set the algorithm to use
        clusterizer_ = new ClusterizerAlgorithm(conf, maxClusterSize_, maxNumberClusters_);
    }

    SimTrackerSiPhase2Clusterizer::~SimTrackerSiPhase2Clusterizer() { }

    void SimTrackerSiPhase2Clusterizer::beginJob(edm::Run const& run, edm::EventSetup const& eventSetup) {
        edm::LogInfo("SimTrackerSiPhase2Clusterizer") << "[SiPhase2Clusterizer::beginJob]";
    }

    void SimTrackerSiPhase2Clusterizer::produce(edm::Event & e, const edm::EventSetup & eventSetup) {

        // Get the Digis
        edm::Handle< edm::DetSetVector< PixelDigi > >  digis;
        e.getByLabel(src_, digis);

        // Get the simlinks for the Digis
        edm::Handle< edm::DetSetVector< PixelDigiSimLink > > pixelSimLinks;
        e.getByLabel(src_, pixelSimLinks);

        // Get the geometry
        edm::ESHandle< TrackerGeometry > geomHandle;
        eventSetup.get< TrackerDigiGeometryRecord >().get(geomHandle);
        const TrackerGeometry* tkGeom = &(*geomHandle); 

        edm::ESHandle<TrackerTopology> tTopoHandle;
        eventSetup.get<IdealGeometryRecord>().get(tTopoHandle);
        const TrackerTopology* tTopo = tTopoHandle.product();

        // Global container for the clusters of each modules
        std::auto_ptr< SiPixelClusterCollectionNew > outputClusters(new SiPixelClusterCollectionNew());

        // Go over all the modules
        for (edm::DetSetVector< PixelDigi >::const_iterator DSViter = digis->begin(); DSViter != digis->end(); ++DSViter) {

            if (!isOuterTracker(DSViter->detId(), tTopo)) continue;

            // Geometry & detID
            DetId detId(DSViter->detId());
            const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(detId);
            const PixelGeomDetUnit* pixDet = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
            if (!pixDet) assert(0);

            // Container for the clusters that will be produced for this modules
            edmNew::DetSetVector<SiPixelCluster>::FastFiller clusters(*outputClusters, DSViter->detId());

            // Setup the clusterizer algorithm for this detector (see ClusterizerAlgorithm for more details)
            clusterizer_->setup(pixDet);

            // Pass the list of Digis to the main algorithm
            // This function will store the clusters in the previously created container
            clusterizer_->clusterizeDetUnit(*DSViter, pixelSimLinks, clusters);

            if (clusters.empty()) clusters.abort();
        }

        // Add the data to the output
        edm::OrphanHandle< edmNew::DetSetVector<SiPixelCluster> > clusterCollection = e.put(outputClusters);

        // Do the same operations for the SimLinks if we have to generate them
        /*if (generateClusterSimLink_) {
            std::vector< edm::DetSet< PixelClusterSimLink > > linksByDet;
            clusterizer_->makeLinks(clusterCollection, linksByDet);
            std::auto_ptr< edm::DetSetVector< PixelClusterSimLink > > outputLinks(new edm::DetSetVector< PixelClusterSimLink >(linksByDet));
            e.put(outputLinks);
        }*/
    }

    bool SimTrackerSiPhase2Clusterizer::isOuterTracker(unsigned int detid, const TrackerTopology* topo) {
        DetId theDetId(detid);
        if (theDetId.det() == DetId::Tracker) {
            if (theDetId.subdetId() == PixelSubdetector::PixelBarrel) return (topo->pxbLayer(detid) >= 5);
            else if (theDetId.subdetId() == PixelSubdetector::PixelEndcap) return (topo->pxfDisk(detid) >= 11);
            else return false;
        }
        return false;
    }
}

using cms::SimTrackerSiPhase2Clusterizer;
DEFINE_FWK_MODULE(SimTrackerSiPhase2Clusterizer);

