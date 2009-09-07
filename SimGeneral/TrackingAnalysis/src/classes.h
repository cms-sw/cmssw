#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <boost/cstdint.hpp> 
#include <vector>
#include <map>

namespace {
  struct dictionary {
    std::vector<EncodedTruthId> dummy0;
    std::map<EncodedTruthId, unsigned int> dummy1;
    edm::Wrapper<std::map<EncodedTruthId, unsigned int> > dummy2;
  };
}
