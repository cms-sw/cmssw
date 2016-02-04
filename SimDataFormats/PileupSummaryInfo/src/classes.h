#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace { 
  struct dictionary {
    PileupSummaryInfo dummy0;
    std::vector<PileupSummaryInfo> dummy1;
    edm::Wrapper<PileupSummaryInfo> dummy2;
    edm::Wrapper<std::vector<PileupSummaryInfo> > dummy3;
  };
}


