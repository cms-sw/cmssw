#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Rtypes.h"
#include "Math/Cartesian3D.h"
#include "Math/Polar3D.h"
#include "Math/CylindricalEta3D.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajAnnealing.h"
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
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"
#include <vector>

namespace TrackingTools_PatternTools {
  struct dictionary {
    std::vector<Trajectory> kk;
    edm::Wrapper<std::vector<Trajectory> > trajCollWrapper;

    std::vector<TrajAnnealing> ta;
    edm::Wrapper<std::vector<TrajAnnealing> > trajAnnCollaction;

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

    edm::helpers::KeyVal<edm::RefProd<std::vector<Trajectory> >, edm::RefProd<std::vector<Trajectory> > > x1;
    edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>, std::vector<Trajectory>, unsigned int> > x2;
    edm::Wrapper<edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>, std::vector<Trajectory>, unsigned int> > >
        x3;
    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::Track> >, edm::RefProd<std::vector<Trajectory> > > x4;
    edm::AssociationMap<edm::OneToOne<std::vector<reco::Track>, std::vector<Trajectory>, unsigned int> > x5;
    edm::Wrapper<edm::AssociationMap<edm::OneToOne<std::vector<reco::Track>, std::vector<Trajectory>, unsigned int> > >
        x6;
    edm::helpers::KeyVal<edm::RefProd<std::vector<Trajectory> >, edm::RefProd<std::vector<TrajectorySeed> > > x7;
    edm::AssociationMap<edm::OneToMany<std::vector<Trajectory>, std::vector<TrajectorySeed>, unsigned int> > x8;
    edm::Wrapper<
        edm::AssociationMap<edm::OneToMany<std::vector<Trajectory>, std::vector<TrajectorySeed>, unsigned int> > >
        x9;
    edm::helpers::KeyVal<edm::RefProd<std::vector<TrackCandidate> >, edm::RefProd<std::vector<Trajectory> > > x10;
    edm::AssociationMap<edm::OneToOne<std::vector<TrackCandidate>, std::vector<Trajectory>, unsigned int> > x11;
    edm::Wrapper<edm::AssociationMap<edm::OneToOne<std::vector<TrackCandidate>, std::vector<Trajectory>, unsigned int> > >
        x12;
  };
}  // namespace TrackingTools_PatternTools
