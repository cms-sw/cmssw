#ifndef SimTracker_SiPhase2Digitizer_ClusterizerAlgorithm_h
#define SimTracker_SiPhase2Digitizer_ClusterizerAlgorithm_h

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

#include "SimTracker/SiPhase2Digitizer/interface/PixelClusterHitArray.h"
//#include "SimTracker/SiPhase2Digitizer/interface/PixelClusterSimLink.h"

#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <map>

class PixelGeomDetUnit;

class PixelClusterSimLink;

class ClusterizerAlgorithm {

public:
    typedef edm::DetSet<PixelDigi>::const_iterator DigiIterator;

    ClusterizerAlgorithm(edm::ParameterSet const& conf, int maxClusterSize, int maxNumberClusters);
    void setup(const PixelGeomDetUnit* pixDet);
    void clusterizeDetUnit(const edm::DetSet<PixelDigi> & pixelDigis, const edm::Handle< edm::DetSetVector< PixelDigiSimLink > > & pixelSimLinks, edmNew::DetSetVector<SiPixelCluster>::FastFiller & clusters) ;
  //  void makeLinks(edm::OrphanHandle< edmNew::DetSetVector<SiPixelCluster> > & clusters, std::vector<edm::DetSet<PixelClusterSimLink> > & linksByDet);

    unsigned int getSimTrackId(const edm::Handle< edm::DetSetVector< PixelDigiSimLink > > & pixelSimLinks, int channel);

private:
    void copy_to_buffer(DigiIterator begin, DigiIterator end);
    void clear_buffer(DigiIterator begin, DigiIterator end);

public:
    edm::ParameterSet conf_;
    PixelClusterHitArray hits;
    int maxClusterSize_;
    int maxNumberClusters_;
    int nrows_;
    int ncols_;
    unsigned int rawDetId_;

    std::map< SiPixelCluster, std::vector< unsigned int > > tmpSimLinks;

    struct AccretionCluster {
        static constexpr unsigned short MAXSIZE = 256;
        unsigned short adc[256];
        unsigned short x[256];
        unsigned short y[256];
        unsigned short xmin = 16000;
        unsigned short xmax = 0;
        unsigned short ymin = 16000;
        unsigned short ymax = 0;
        unsigned int isize = 0;
        unsigned int curr = 0;
        unsigned short top() const { return curr; }
        void pop() { ++curr; }
        bool empty() { return curr == isize; }
        bool add(SiPixelCluster::PixelPos const & p, unsigned short const iadc) {
            if (isize == MAXSIZE) return false;
            xmin = std::min(xmin, (unsigned short) p.row());
            xmax = std::max(xmax, (unsigned short) p.row());
            ymin = std::min(ymin, (unsigned short) p.col());
            ymax = std::max(ymax, (unsigned short) p.col());
            adc[isize] = iadc;
            x[isize] = p.row();
            y[isize++] = p.col();
            return true;
        }
        unsigned short size() { return isize; }
        unsigned short xsize() { return xmax - xmin + 1; }
        unsigned short ysize() { return ymax - ymin + 1; }
    };

};

#endif
