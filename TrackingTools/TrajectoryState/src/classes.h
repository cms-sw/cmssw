#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Rtypes.h"
#include "Math/Cartesian3D.h"
#include "Math/Polar3D.h"
#include "Math/CylindricalEta3D.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/CLHEP/interface/Migration.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include <vector>

typedef TrajectoryStateOnSurface TrackParamConstraint;
typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection, std::vector<TrajectoryStateOnSurface> > >
    TrackParamConstraintAssociationCollection;
typedef TrackParamConstraintAssociationCollection::value_type TrackParamConstraintAssociation;
typedef edm::Ref<TrackParamConstraintAssociationCollection> TrackParamConstraintAssociationRef;
typedef edm::RefProd<TrackParamConstraintAssociationCollection> TrackParamConstraintAssociationRefProd;
typedef edm::RefVector<TrackParamConstraintAssociationCollection> TrackParamConstraintAssociationRefVector;

namespace TrackingTools_TrajectoryState {
  struct dictionary {
    TrajectoryStateOnSurface dummytsos;
    std::vector<TrajectoryStateOnSurface> jjj2;
    edm::Wrapper<std::vector<TrajectoryStateOnSurface> > jjj3;

    TrackParamConstraintAssociationCollection iii1;
    edm::Wrapper<TrackParamConstraintAssociationCollection> iii2;
    TrackParamConstraintAssociation iii3;
    TrackParamConstraintAssociationRef iii4;
    TrackParamConstraintAssociationRefProd iii5;
    TrackParamConstraintAssociationRefVector iii6;
  };
}  // namespace TrackingTools_TrajectoryState
