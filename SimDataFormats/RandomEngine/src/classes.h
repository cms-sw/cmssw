
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace { 
  struct dictionary {
    edm::Wrapper<std::vector<RandomEngineState> > dummy1;
  };
}
