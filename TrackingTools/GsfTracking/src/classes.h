#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"
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
