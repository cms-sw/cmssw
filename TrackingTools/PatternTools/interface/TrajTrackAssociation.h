#ifndef TrackingTools_PatternTools_TrajTrackAssociation_h
#define TrackingTools_PatternTools_TrajTrackAssociation_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

typedef edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>,
                                          reco::TrackCollection,unsigned short> > TrajTrackAssociationCollection;
typedef TrajTrackAssociationCollection::value_type TrajTrackAssociation;

// reference to an object in a collection of TrajTrack objects
typedef edm::Ref<TrajTrackAssociationCollection> TrajTrackAssociationRef;

/// reference to a collection of TrajTrack objects
typedef edm::RefProd<TrajTrackAssociationCollection> TrajTrackAssociationRefProd;

/// vector of references to objects in the same colletion of TrajTrack objects
typedef edm::RefVector<TrajTrackAssociationCollection> TrajTrackAssociationRefVector;

#endif
