#ifndef CaloAnalysis_MtdSimTracksterFwd_h
#define CaloAnalysis_MtdSimTracksterFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

class MtdSimTrackster;
typedef std::vector<MtdSimTrackster> MtdSimTracksterCollection;
typedef edm::Ref<MtdSimTracksterCollection> MtdSimTracksterRef;
typedef edm::RefVector<MtdSimTracksterCollection> MtdSimTracksterRefVector;
typedef edm::RefProd<MtdSimTracksterCollection> MtdSimTracksterRefProd;
typedef edm::RefVector<MtdSimTracksterCollection> MtdSimTracksterContainer;

std::ostream &operator<<(std::ostream &s, MtdSimTrackster const &tp);

#endif
