#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <map>
#include <string>

namespace {
namespace {
 	CrossingFrame<PSimHit> dummy1;
 	CrossingFrame<PCaloHit> dummy2;
 	CrossingFrame<SimTrack> dummy3;
 	CrossingFrame<SimVertex> dummy4;
        std::vector<int> dummy14;
        std::vector<PSimHit> dummy10;
        std::vector<PCaloHit> dummy11;
        std::vector<SimTrack> dummy12;
        std::vector<SimVertex> dummy13;
	//        std::map<std::string,edm::PCaloHitContainer> dummy3;
	//        std::map<std::string,edm::PSimHitContainer> dummy4;
	//        std::map<std::string,std::vector<edm::PCaloHitContainer> > dummy5;
	//        std::map<std::string,std::vector<edm::PSimHitContainer> > dummy6;

        edm::Wrapper<CrossingFrame<PSimHit> > dummy20;
        edm::Wrapper<CrossingFrame<PCaloHit> > dummy21;
        edm::Wrapper<CrossingFrame<SimTrack> > dummy22;
        edm::Wrapper<CrossingFrame<SimVertex> > dummy23;
}
}
