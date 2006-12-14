#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include <vector>

namespace {
  namespace {
    std::vector<Trajectory> kk;
    edm::Wrapper< std::vector<Trajectory> > trajCollWrapper;
    
    TrajTrackAssociationCollection ttam;
    edm::Wrapper<TrajTrackAssociationCollection> wttam;
    TrajTrackAssociation vttam;
    TrajTrackAssociationRef rttam;
    TrajTrackAssociationRefProd rpttam;
    TrajTrackAssociationRefVector rvttam;
  
  }
}
