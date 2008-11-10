#include "SimDataFormats/HiGenData/interface/SubEvent.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {
    std::vector<edm::SubEvent> dummy2;
    edm::GenHIEvent dummy0;
    edm::Wrapper<edm::GenHIEvent> dummy1;
  };
}

