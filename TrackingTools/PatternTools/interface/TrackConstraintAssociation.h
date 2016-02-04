#ifndef TrackingTools_PatternTools_TrackConstraintAssociation_h
#define TrackingTools_PatternTools_TrackConstraintAssociation_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

//typedef std::pair<double,double> MomentumConstraint;
struct MomentumConstraint {
  MomentumConstraint(const double & f, const double & s) :momentum(f),error(s){}
  MomentumConstraint() : momentum(0), error(0) {}
  
  double momentum;
  double error;
};

typedef std::pair<GlobalPoint,GlobalError> VertexConstraint;

typedef TrajectoryStateOnSurface TrackParamConstraint;

typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection,std::vector<MomentumConstraint> > > TrackMomConstraintAssociationCollection;
typedef TrackMomConstraintAssociationCollection::value_type TrackMomConstraintAssociation;
typedef edm::Ref<TrackMomConstraintAssociationCollection> TrackMomConstraintAssociationRef;
typedef edm::RefProd<TrackMomConstraintAssociationCollection> TrackMomConstraintAssociationRefProd;
typedef edm::RefVector<TrackMomConstraintAssociationCollection> TrackMomConstraintAssociationRefVector;

typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection,std::vector<VertexConstraint> > > TrackVtxConstraintAssociationCollection;
typedef TrackVtxConstraintAssociationCollection::value_type TrackVtxConstraintAssociation;
typedef edm::Ref<TrackVtxConstraintAssociationCollection> TrackVtxConstraintAssociationRef;
typedef edm::RefProd<TrackVtxConstraintAssociationCollection> TrackVtxConstraintAssociationRefProd;
typedef edm::RefVector<TrackVtxConstraintAssociationCollection> TrackVtxConstraintAssociationRefVector;

typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection,std::vector<TrackParamConstraint> > > TrackParamConstraintAssociationCollection;
typedef TrackParamConstraintAssociationCollection::value_type TrackParamConstraintAssociation;
typedef edm::Ref<TrackParamConstraintAssociationCollection> TrackParamConstraintAssociationRef;
typedef edm::RefProd<TrackParamConstraintAssociationCollection> TrackParamConstraintAssociationRefProd;
typedef edm::RefVector<TrackParamConstraintAssociationCollection> TrackParamConstraintAssociationRefVector;

#endif
