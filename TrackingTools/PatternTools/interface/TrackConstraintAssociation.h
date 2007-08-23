#ifndef TrackingTools_PatternTools_TrackConstraintAssociation_h
#define TrackingTools_PatternTools_TrackConstraintAssociation_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
typedef std::pair<double,double> MomentumConstraint;
typedef std::pair<GlobalPoint,GlobalError> VertexConstraint;

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

#endif
