#ifndef CaloAnalysis_MtdSimTracksterFwd_h
#define CaloAnalysis_MtdSimTracksterFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

namespace io_v1 {
  class MtdSimTrackster;
  std::ostream &operator<<(std::ostream &s, MtdSimTrackster const &tp);
}  // namespace io_v1
using MtdSimTrackster = io_v1::MtdSimTrackster;

typedef std::vector<MtdSimTrackster> MtdSimTracksterCollection;
typedef edm::Ref<MtdSimTracksterCollection> MtdSimTracksterRef;
typedef edm::RefVector<MtdSimTracksterCollection> MtdSimTracksterRefVector;
typedef edm::RefProd<MtdSimTracksterCollection> MtdSimTracksterRefProd;
typedef edm::RefVector<MtdSimTracksterCollection> MtdSimTracksterContainer;

#endif
