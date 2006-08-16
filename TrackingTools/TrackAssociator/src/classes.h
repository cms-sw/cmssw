#include "DataFormats/Common/interface/Wrapper.h"
#include "TrackingTools/TrackAssociator/interface/TestMuon.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfoCollection.h"
#include "TrackingTools/TrackAssociator/interface/MuonSegmentMatchCollection.h"

#include <vector>

namespace {
  namespace {
    std::vector<reco::TestMuon> v1;
    std::vector<math::XYZPoint> vp1;
    std::vector<math::XYZVector> vv1;
    edm::Wrapper<std::vector<reco::TestMuon> > c1;
    edm::Ref<std::vector<reco::TestMuon> > r1;
    edm::RefProd<std::vector<reco::TestMuon> > rp1;
    edm::RefVector<std::vector<reco::TestMuon> > rv1;
     
    TrackDetMatchInfo tdi1;
    edm::Wrapper<TrackDetMatchInfoCollection> trackDetMatchInfoCollectionWrapper;
    edm::Wrapper<MuonSegmentMatchCollection> muonSegmentMatchCollectionWrapper;
    reco::TestMuon::MuonMatch mm1;
    std::vector<reco::TestMuon::MuonMatch> vmm1;
    reco::TestMuon::MuonIsolation mi1;
    reco::TestMuon::MuonEnergy me1;
    // std::vector<reco::AssociatedMuonSegment> p1;
  }
}
