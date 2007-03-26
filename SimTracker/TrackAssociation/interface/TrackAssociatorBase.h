#ifndef TrackAssociatorBase_h
#define TrackAssociatorBase_h

/** \class TrackAssociatorBase
 *  Base class for TrackAssociators. Methods take as input the handle of Track and TrackingPArticle collections and return an AssociationMap (oneToManyWithQuality)
 *
 *  $Date: 2007/03/26 10:13:49 $
 *  $Revision: 1.1 $
 *  \author magni, cerati
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"


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
  /// Constructor
  TrackAssociatorBase() {;} 
  /// Destructor
  virtual ~TrackAssociatorBase() {;}

  /// Association Reco To Sim
  virtual  reco::RecoToSimCollection associateRecoToSim (edm::Handle<reco::TrackCollection>& tc, 
							 edm::Handle<TrackingParticleCollection>& tpc,
							 const edm::Event * event = 0 ) const = 0;
  /// Association Sim To Reco
  virtual  reco::SimToRecoCollection associateSimToReco (edm::Handle<reco::TrackCollection>& tc, 
							 edm::Handle<TrackingParticleCollection> & tpc ,  
							 const edm::Event * event = 0 ) const = 0;

};


#endif
