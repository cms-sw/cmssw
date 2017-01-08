#include "VHbbAnalysis/Heppy/interface/FastSoftActivity.h"
#include "VHbbAnalysis/Heppy/interface/ColorFlow.h"
#include "VHbbAnalysis/Heppy/interface/SignedImpactParameter.h"

#include <vector>

namespace {
  struct heppy_dictionaryvhbb {
    heppy::FastSoftActivity  fs_;
    heppy::ColorFlow cl_;
  };
}

namespace {
    struct dictionary {
        SignedImpactParameter sipc;
    };
}


