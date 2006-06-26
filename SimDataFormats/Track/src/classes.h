#include "SimDataFormats/Track/interface/CoreSimTrack.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <CLHEP/Vector/LorentzVector.h>

#include <vector>
 
namespace {
  namespace {
    CLHEP::HepLorentzVector dummy1;
    CLHEP::Hep3Vector dummy2;
    SimTrack dummy22;
    edm::SimTrackContainer dummy222;
    edm::Wrapper<edm::SimTrackContainer> dummy22222;
    SimTrackRef r1;
    SimTrackRefVector rv1;
    SimTrackRefProd rp1;
  }
}
