#ifndef VertexAssociatorByTracks_h
#define VertexAssociatorByTracks_h

#include "SimTracker/VertexAssociation/interface/VertexAssociatorBase.h"

class VertexAssociatorByTracks : public VertexAssociatorBase {
  
 public:
  explicit VertexAssociatorByTracks( const edm::ParameterSet& );  
  ~VertexAssociatorByTracks();
  
/* Associate TrackingVertex to RecoVertex By Hits */

  reco::VertexRecoToSimCollection 
    associateRecoToSim (edm::Handle<reco::VertexCollection>& vc, 
                        edm::Handle<TrackingVertexCollection>& tvc,
                        const edm::Event&    event, 
                        reco::RecoToSimCollection& trackAssocResult);

  reco::VertexSimToRecoCollection 
    associateSimToReco (edm::Handle<reco::VertexCollection>& vc, 
                        edm::Handle<TrackingVertexCollection>& tvc ,  
                        const edm::Event&    event, 
                        reco::RecoToSimCollection& trackAssocResult);

 private:
  // ----- member data
  const edm::ParameterSet& conf_;
//  const double theMinHitFraction;    
//  int LayerFromDetid(const DetId&);
};

#endif
