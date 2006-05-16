#include "SimDataFormats/Track/interface/CoreSimTrack.h"
#include "SimDataFormats/Track/interface/EmbdSimTrack.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <CLHEP/Vector/LorentzVector.h>

#include <vector>
 
namespace {
  namespace {
    CLHEP::HepLorentzVector dummy1;
    CLHEP::Hep3Vector dummy2;
    EmbdSimTrack dummy22;
    edm::EmbdSimTrackContainer dummy222;
    edm::Wrapper<edm::EmbdSimTrackContainer> dummy22222;
    EmbdSimTrackRef r1;
    EmbdSimTrackRefVector rv1;
    EmbdSimTrackRefProd rp1;
  }
}
