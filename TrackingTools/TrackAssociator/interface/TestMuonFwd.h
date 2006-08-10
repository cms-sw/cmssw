#ifndef TrackAssociator_TestMuonFwd_h
#define TrackAssociator_TestMuonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class TestMuon;
  /// collection of TestMuon objects
  typedef std::vector<TestMuon> TestMuonCollection;
  /// presistent reference to a TestMuon
  typedef edm::Ref<TestMuonCollection> TestMuonRef;
  /// references to TAMuon collection
  typedef edm::RefProd<TestMuonCollection> TestMuonRefProd;
  /// vector of references to TAMuon objects all in the same collection
  typedef edm::RefVector<TestMuonCollection> TestMuonRefVector;
  /// iterator over a vector of references to TestMuon objects all in the same collection
  typedef TestMuonRefVector::iterator testmuon_iterator;
}

#endif
