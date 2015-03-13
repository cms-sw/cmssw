#ifndef SimTracker_SiPhase2Digitizer_Phase2TrackerClusterizer_h
#define SimTracker_SiPhase2Digitizer_Phase2TrackerClusterizer_h

#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerClusterizerAlgorithm.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace cms {

    class Phase2TrackerClusterizer : public edm::EDProducer {

        public:

            explicit Phase2TrackerClusterizer(const edm::ParameterSet& conf);
            virtual ~Phase2TrackerClusterizer();
            virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);

        private:

            bool isOuterTracker(const DetId& detid, const TrackerTopology* topo);
            
            edm::ParameterSet conf_;
            Phase2TrackerClusterizerAlgorithm* clusterizer_;
            edm::InputTag src_;
            unsigned int maxClusterSize_;
            unsigned int maxNumberClusters_;

    };
}

#endif
