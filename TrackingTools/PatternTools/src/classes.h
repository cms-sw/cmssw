#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "DataFormats/CLHEP/interface/Migration.h" 
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "boost/intrusive_ptr.hpp" 
#include "DataFormats/GeometryVector/interface/LocalVector.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 
#include "DataFormats/Common/interface/OwnVector.h" 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"
#include <vector>

namespace {
  struct dictionary {
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
  
  };
}
