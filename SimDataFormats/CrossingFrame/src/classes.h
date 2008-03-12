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
 	CrossingFrame dummy1;
        std::vector<edm::PSimHitContainer> dummy10;
        std::vector<edm::PCaloHitContainer> dummy11;
        std::vector<edm::SimTrackContainer> dummy2;
        std::vector<edm::SimVertexContainer> dummy22;
        std::map<std::string,edm::PCaloHitContainer> dummy3;
        std::map<std::string,edm::PSimHitContainer> dummy4;
        std::map<std::string,std::vector<edm::PCaloHitContainer> > dummy5;
        std::map<std::string,std::vector<edm::PSimHitContainer> > dummy6;

        edm::Wrapper<CrossingFrame> dummy20;
}
}
