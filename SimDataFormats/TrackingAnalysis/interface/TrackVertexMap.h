#ifndef SimDataFormats_TrackVertexMap_h
#define SimDataFormats_TrackVertexMap_h

#include "DataFormats/Common/interface/AssociationMap.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"


typedef edm::AssociationMap<edm::OneToMany<TrackingParticleCollection,TrackingVertexCollection> > 
        TrackVertexAssociationCollection;

typedef TrackVertexAssociationCollection::value_type     TrackVertexAssociation;       
typedef edm::Ref<TrackVertexAssociationCollection>       TrackVertexAssociationRef;       
typedef edm::RefProd<TrackVertexAssociationCollection>   TrackVertexAssociationRefProd;       
typedef edm::RefVector<TrackVertexAssociationCollection> TrackVertexAssociationRefVector;       
           
typedef edm::AssociationMap<edm::OneToMany<TrackingVertexCollection,TrackingParticleCollection> > 
        VertexTrackAssociationCollection;

typedef VertexTrackAssociationCollection::value_type     VertexTrackAssociation;       
typedef edm::Ref<VertexTrackAssociationCollection>       VertexTrackAssociationRef;       
typedef edm::RefProd<VertexTrackAssociationCollection>   VertexTrackAssociationRefProd;       
typedef edm::RefVector<VertexTrackAssociationCollection> VertexTrackAssociationRefVector;       
            
/*class TrackVertexMap {

 public:
   void addVertexToTrack(TrackingParticleRef &t, TrackingVertexRef &v); 
//   TrackingVertexContainer::const_iterator verticesForTrack_begin();
//   TrackingVertexContainer::const_iterator verticesForTrack_end();
 private:
         
   TrackVertexAssociationCollection trackVertexMap_;  

};*/

#endif
