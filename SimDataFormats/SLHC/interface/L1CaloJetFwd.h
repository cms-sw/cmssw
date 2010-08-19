#ifndef L1CaloJet_Fwd
#define L1CaloJet_Fwd
#include "DataFormats/Common/interface/Ref.h"

namespace l1slhc {
class L1CaloJet;
}

namespace l1slhc {
  typedef std::vector<L1CaloJet> L1CaloJetCollection;
  typedef edm::Ref<L1CaloJetCollection> L1CaloJetRef;
}

#endif
