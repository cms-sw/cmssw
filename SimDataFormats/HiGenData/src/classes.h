#include "SimDataFormats/HiGenData/interface/SubEvent.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/HiGenData/interface/SubEventMap.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {
    std::vector<edm::SubEvent> dummy2;
    edm::GenHIEvent dummy0;
    edm::SubEventMap dummy3;
    edm::Wrapper<edm::GenHIEvent> dummy1;
    edm::Wrapper<edm::SubEventMap> dummy4;
    edm::helpers::Key<edm::RefProd<reco::GenParticleCollection> > dumdumy;
  };
}

