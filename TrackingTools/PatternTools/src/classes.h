#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"
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

    std::vector<MomentumConstraint> j2;
    edm::Wrapper<std::vector<MomentumConstraint> > j3;
  
    TrackMomConstraintAssociationCollection i1;
    edm::Wrapper<TrackMomConstraintAssociationCollection> i2;
    TrackMomConstraintAssociation i3;
    TrackMomConstraintAssociationRef i4;
    TrackMomConstraintAssociationRefProd i5;
    TrackMomConstraintAssociationRefVector i6;
  
    std::vector<VertexConstraint> jj2;
    edm::Wrapper<std::vector<VertexConstraint> > jj3;
  
    TrackVtxConstraintAssociationCollection ii1;
    edm::Wrapper<TrackVtxConstraintAssociationCollection> ii2;
    TrackVtxConstraintAssociation ii3;
    TrackVtxConstraintAssociationRef ii4;
    TrackVtxConstraintAssociationRefProd ii5;
    TrackVtxConstraintAssociationRefVector ii6;
  
  }
}
