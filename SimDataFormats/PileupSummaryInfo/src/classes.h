#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupVertexContent.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace SimDataFormats_PileupSummaryInfo {
  struct dictionary {
    PileupSummaryInfo dummy0;
    std::vector<PileupSummaryInfo> dummy1;
    edm::Wrapper<PileupSummaryInfo> dummy2;
    edm::Wrapper<std::vector<PileupSummaryInfo> > dummy3;
    PileupMixingContent dummy4;
    std::vector<PileupMixingContent> dummy5;
    edm::Wrapper<PileupMixingContent> dummy6;
    edm::Wrapper<std::vector<PileupMixingContent> > dummy7;
    PileupVertexContent dummy8;
    std::vector<PileupVertexContent> dummy9;
    edm::Wrapper<PileupVertexContent> dummy10;
    edm::Wrapper<std::vector<PileupVertexContent> > dummy11;
  };
}


