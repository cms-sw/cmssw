#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
// #include "Rtypes.h" 
// #include "Math/Cartesian3D.h" 
// #include "Math/Polar3D.h" 
// #include "Math/CylindricalEta3D.h" 
// #include <boost/cstdint.hpp> 
// #include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "TrackingTools/PatternTools/interface/Trajectory.h"
// #include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
// #include "DataFormats/GeometrySurface/interface/Surface.h" 
// #include "DataFormats/CLHEP/interface/Migration.h" 
// #include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
// #include "boost/intrusive_ptr.hpp" 
// #include "TrackingTools/DetLayers/interface/DetLayer.h" 
// #include "DataFormats/GeometryVector/interface/LocalVector.h" 
// #include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 
// #include "DataFormats/Common/interface/OwnVector.h" 
// #include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"
// #include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"
#include "TrackingTools/GsfTracking/interface/GsfTrackConstraintAssociation.h"
#include <vector>

namespace {
  struct dictionary {

    TrajGsfTrackAssociationCollection ttam;
    edm::Wrapper<TrajGsfTrackAssociationCollection> wttam;
    TrajGsfTrackAssociation vttam;
    TrajGsfTrackAssociationRef rttam;
    TrajGsfTrackAssociationRefProd rpttam;
    TrajGsfTrackAssociationRefVector rvttam;


    GsfTrackMomConstraintAssociationCollection i1;
    edm::Wrapper<GsfTrackMomConstraintAssociationCollection> i2;
    GsfTrackMomConstraintAssociation i3;
    GsfTrackMomConstraintAssociationRef i4;
    GsfTrackMomConstraintAssociationRefProd i5;
    GsfTrackMomConstraintAssociationRefVector i6;
  
    GsfTrackVtxConstraintAssociationCollection ii1;
    edm::Wrapper<GsfTrackVtxConstraintAssociationCollection> ii2;
    GsfTrackVtxConstraintAssociation ii3;
    GsfTrackVtxConstraintAssociationRef ii4;
    GsfTrackVtxConstraintAssociationRefProd ii5;
    GsfTrackVtxConstraintAssociationRefVector ii6;
  
  };
}
