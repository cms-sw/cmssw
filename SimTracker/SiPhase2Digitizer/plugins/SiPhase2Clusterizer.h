#ifndef SimTracker_SiPhase2Digitizer_SimTrackerSiPhase2Clusterizer_h
#define SimTracker_SiPhase2Digitizer_SimTrackerSiPhase2Clusterizer_h

#include "SimTracker/SiPhase2Digitizer/interface/ClusterizerAlgorithm.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace cms {

    class SimTrackerSiPhase2Clusterizer : public edm::EDProducer {

        public:
            explicit SimTrackerSiPhase2Clusterizer(const edm::ParameterSet & conf);
            virtual ~SimTrackerSiPhase2Clusterizer();
    	    void setupClusterizer();
    	    void beginJob(edm::Run const& run, edm::EventSetup const& eventSetup);
    	    virtual void produce(edm::Event & e, const edm::EventSetup & eventSetup);
            bool isOuterTracker(unsigned int detid, const TrackerTopology* topo);

  	private:
    	    edm::ParameterSet conf_;
    	    ClusterizerAlgorithm* clusterizer_;
            edm::InputTag src_;
            int maxClusterSize_;
            int maxNumberClusters_;
            bool generateClusterSimLink_;
    };
}

#endif
