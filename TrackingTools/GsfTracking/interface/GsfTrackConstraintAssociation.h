#ifndef GsfTrackingTools_PatternTools_GsfTrackConstraintAssociation_h
#define GsfTrackingTools_PatternTools_GsfTrackConstraintAssociation_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

typedef edm::AssociationMap<edm::OneToOne<reco::GsfTrackCollection,std::vector<MomentumConstraint> > > 
GsfTrackMomConstraintAssociationCollection;
typedef GsfTrackMomConstraintAssociationCollection::value_type GsfTrackMomConstraintAssociation;
typedef edm::Ref<GsfTrackMomConstraintAssociationCollection> GsfTrackMomConstraintAssociationRef;
typedef edm::RefProd<GsfTrackMomConstraintAssociationCollection> GsfTrackMomConstraintAssociationRefProd;
typedef edm::RefVector<GsfTrackMomConstraintAssociationCollection> 
GsfTrackMomConstraintAssociationRefVector;

typedef edm::AssociationMap<edm::OneToOne<reco::GsfTrackCollection,std::vector<VertexConstraint> > > 
GsfTrackVtxConstraintAssociationCollection;
typedef GsfTrackVtxConstraintAssociationCollection::value_type GsfTrackVtxConstraintAssociation;
typedef edm::Ref<GsfTrackVtxConstraintAssociationCollection> GsfTrackVtxConstraintAssociationRef;
typedef edm::RefProd<GsfTrackVtxConstraintAssociationCollection> GsfTrackVtxConstraintAssociationRefProd;
typedef edm::RefVector<GsfTrackVtxConstraintAssociationCollection> 
GsfTrackVtxConstraintAssociationRefVector;

#endif
