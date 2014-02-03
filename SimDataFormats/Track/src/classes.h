#include "SimDataFormats/Track/interface/CoreSimTrack.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>
 
namespace {
  struct dictionary {
    SimTrack dummy22;
    edm::SimTrackContainer dummy222;
    std::vector<const SimTrack*> dummyvcp;
    edm::Wrapper<edm::SimTrackContainer> dummy22222;
    SimTrackRef r1;
    SimTrackRefVector rv1;
    SimTrackRefProd rp1;
  };
}
