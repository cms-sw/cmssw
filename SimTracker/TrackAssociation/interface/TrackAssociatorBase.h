#ifndef TrackAssociatorBase_h
#define TrackAssociatorBase_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "FWCore/Framework/interface/Handle.h"


namespace reco{

  typedef edm::AssociationMap<edm::OneToManyWithQuality
    <TrackingParticleCollection, reco::TrackCollection, double> >
    SimToRecoCollection;  
  typedef edm::AssociationMap<edm::OneToManyWithQuality 
    <reco::TrackCollection, TrackingParticleCollection, double> >
    RecoToSimCollection;  
  
}


class TrackAssociatorBase {
 public:
  TrackAssociatorBase() {;} 
  virtual ~TrackAssociatorBase() {;}
  virtual  reco::RecoToSimCollection associateRecoToSim (edm::Handle<reco::TrackCollection>&, 
							 edm::Handle<TrackingParticleCollection>& ) = 0;
  virtual  reco::SimToRecoCollection associateSimToReco (edm::Handle<reco::TrackCollection>&, 
							 edm::Handle<TrackingParticleCollection>& ) = 0;

};


#endif
