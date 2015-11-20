#ifndef SimTracker_SiPhase2Digitizer_Phase2TrackerClusterizerAlgorithm_h
#define SimTracker_SiPhase2Digitizer_Phase2TrackerClusterizerAlgorithm_h

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerClusterizerArray.h"

class Phase2TrackerClusterizerAlgorithm {

    public:

        Phase2TrackerClusterizerAlgorithm(unsigned int, unsigned int);
        void setup(const PixelGeomDetUnit*);
        void clusterizeDetUnit(const edm::DetSet< Phase2TrackerDigi >&, edmNew::DetSetVector< Phase2TrackerCluster1D >::FastFiller&);

    private:

        void fillMatrix(edm::DetSet< Phase2TrackerDigi >::const_iterator, edm::DetSet< Phase2TrackerDigi >::const_iterator);
        void clearMatrix(edm::DetSet< Phase2TrackerDigi >::const_iterator, edm::DetSet< Phase2TrackerDigi >::const_iterator);

        Phase2TrackerClusterizerArray matrix_;
        unsigned int maxClusterSize_;
        unsigned int maxNumberClusters_;
        unsigned int nrows_;
        unsigned int ncols_;

};

#endif
