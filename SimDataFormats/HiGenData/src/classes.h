#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {
    edm::GenHIEvent dummy0;
    edm::Wrapper<edm::GenHIEvent> dummy1;
    edm::helpers::Key<edm::RefProd<reco::GenParticleCollection> > dumdumy;
  };
}

