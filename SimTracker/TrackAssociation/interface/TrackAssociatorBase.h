#ifndef TrackAssociatorBase_h
#define TrackAssociatorBase_h

/** \class TrackAssociatorBase
 *  Base class for TrackAssociators. Methods take as input the handle of Track and TrackingPArticle collections and return an AssociationMap (oneToManyWithQuality)
 *
 *  $Date: 2007/10/26 14:35:20 $
 *  $Revision: 1.11 $
 *  \author magni, cerati
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

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

  /// Association Reco To Sim with Collections
  virtual  reco::RecoToSimCollection associateRecoToSim(const reco::TrackCollection& tc,
                                                        const TrackingParticleCollection& tpc,
                                                        edm::Provenance const* provTC,
                                                        edm::Provenance const* provTPC,
                                                        const edm::Event * event = 0 ) const {
    edm::Handle<reco::TrackCollection> tcH = edm::Handle<reco::TrackCollection>(&tc, provTC);
    edm::Handle<TrackingParticleCollection> tpcH = edm::Handle<TrackingParticleCollection>(&tpc, provTPC);
    return associateRecoToSim(tcH,tpcH);
  }

  /// Association Sim To Reco with Collections
  virtual  reco::SimToRecoCollection associateSimToReco(reco::TrackCollection& tc,
                                                        TrackingParticleCollection& tpc ,
                                                        edm::Provenance const* provTC,
                                                        edm::Provenance const* provTPC,
                                                        const edm::Event * event = 0 ) const {
    edm::Handle<reco::TrackCollection> tcH = edm::Handle<reco::TrackCollection>(&tc, provTC);
    edm::Handle<TrackingParticleCollection> tpcH = edm::Handle<TrackingParticleCollection>(&tpc, provTPC);
    return associateSimToReco(tcH,tpcH);
  }
  
};


#endif
