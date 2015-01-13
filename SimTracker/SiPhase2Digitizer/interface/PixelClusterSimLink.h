#ifndef SimTracker_SiPhase2Digitizer_PixelClusterSimLinks_h
#define SimTracker_SiPhase2Digitizer_PixelClusterSimLinks_h

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>
#include <algorithm>

class PixelClusterSimLink {

public:
    PixelClusterSimLink() { };
    void setSimTracks(std::vector< unsigned int > simTrack) {
        simTracks_ = simTrack;
        std::sort(simTracks_.begin(), simTracks_.end() );
        simTracks_.erase(std::unique(simTracks_.begin(), simTracks_.end()), simTracks_.end());
    };
    void setCluster(edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster>  cluster) { cluster_ = cluster; };

    std::vector< unsigned int > getSimTracks() { return simTracks_; };
    edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster>  getCluster() { return cluster_; };

    inline bool operator< ( const PixelClusterSimLink& other ) const { return true; }

private:
    edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster>  cluster_;
    std::vector< unsigned int > simTracks_;

};

#endif
