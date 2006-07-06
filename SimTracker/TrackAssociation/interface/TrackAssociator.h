#ifndef TrackAssociation_TrackAssociator_h
#define TrackAssociation_TrackAssociator_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
//#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
//#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <vector>

class Track;
class ParticleTrack;

/** Abstract base class for track associators
 */


class TrackAssociator {
public:
  //  typedef edm::AssociationMap<edm::OneToMany<TrackingParticleContainer, reco::TrackCollection, unsigned int> > SimToRecoCollection;  
  //  typedef edm::AssociationMap<edm::OneToMany<reco::TrackCollection, TrackingParticleContainer, unsigned int> > RecoToSimCollection;  
 /*  typedef edm::AssociationMap<edm::OneToMany<edm::SimTrackContainer, reco::TrackCollection, unsigned int> > SimToRecoCollection;   */
/*   typedef edm::AssociationMap<edm::OneToMany<reco::TrackCollection, edm::SimTrackContainer, unsigned int> > RecoToSimCollection;   */
  
  
  TrackAssociator() {}
  virtual ~TrackAssociator() {}

};


#endif // TrackAssociation_TrackAssociator_h
