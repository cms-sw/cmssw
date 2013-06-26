#ifndef TrackingTools_GsfTracking_TrajGsfTrackAssociation_h
#define TrackingTools_GsfTracking_TrajGsfTrackAssociation_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

typedef edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>,
                                          reco::GsfTrackCollection,unsigned short> > TrajGsfTrackAssociationCollection;
typedef TrajGsfTrackAssociationCollection::value_type TrajGsfTrackAssociation;

// reference to an object in a collection of TrajGsfTrack objects
typedef edm::Ref<TrajGsfTrackAssociationCollection> TrajGsfTrackAssociationRef;

/// reference to a collection of TrajGsfTrack objects
typedef edm::RefProd<TrajGsfTrackAssociationCollection> TrajGsfTrackAssociationRefProd;

/// vector of references to objects in the same colletion of TrajGsfTrack objects
typedef edm::RefVector<TrajGsfTrackAssociationCollection> TrajGsfTrackAssociationRefVector;

#endif
