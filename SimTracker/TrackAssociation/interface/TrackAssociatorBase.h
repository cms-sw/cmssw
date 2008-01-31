#ifndef TrackAssociatorBase_h
#define TrackAssociatorBase_h

/** \class TrackAssociatorBase
 *  Base class for TrackAssociators. Methods take as input the handle of Track and TrackingPArticle collections and return an AssociationMap (oneToManyWithQuality)
 *
 *  $Date: 2007/12/18 16:15:32 $
 *  $Revision: 1.13 $
 *  \author magni, cerati
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

namespace reco{
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric
    <TrackingParticleCollection, TrajectorySeedCollection, double> >
    SimToRecoCollectionSeed;  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric 
    <TrajectorySeedCollection, TrackingParticleCollection, double> >
    RecoToSimCollectionSeed;  
  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric
    <TrackingParticleCollection, TrackCandidateCollection, double> >
    SimToRecoCollectionTCandidate;  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric 
    <TrajectorySeedCollection, TrackCandidateCollection, double> >
    RecoToSimCollectionTCandidate;  
  }

class TrackAssociatorBase {
 public:
  /// Constructor
  TrackAssociatorBase() {;} 
  /// Destructor
  virtual ~TrackAssociatorBase() {;}


  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  virtual reco::RecoToSimCollection associateRecoToSim(edm::Handle<edm::View<reco::Track> >& tCH, 
						       edm::Handle<TrackingParticleCollection>& tPCH, 
						       const edm::Event * event = 0) const {
    edm::RefToBaseVector<reco::Track> tc(tCH);
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(edm::RefToBase<reco::Track>(tCH,j));

    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));

    return associateRecoToSim(tc,tpc,event);
  }
  
  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  virtual reco::SimToRecoCollection associateSimToReco(edm::Handle<edm::View<reco::Track> >& tCH, 
						       edm::Handle<TrackingParticleCollection>& tPCH,
						       const edm::Event * event = 0) const {
    edm::RefToBaseVector<reco::Track> tc(tCH);
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(edm::RefToBase<reco::Track>(tCH,j));

    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));

    return associateSimToReco(tc,tpc,event);
  }  
  
  /// Association Reco To Sim with Collections
  virtual  reco::RecoToSimCollection associateRecoToSim(edm::RefToBaseVector<reco::Track> & tc,
                                                        edm::RefVector<TrackingParticleCollection>& tpc,
                                                        const edm::Event * event = 0 ) const = 0 ;
  /// Association Sim To Reco with Collections
  virtual  reco::SimToRecoCollection associateSimToReco(edm::RefToBaseVector<reco::Track> & tc,
                                                        edm::RefVector<TrackingParticleCollection>& tpc ,
                                                        const edm::Event * event = 0 ) const = 0 ; 

  //TrajectorySeed
  virtual reco::RecoToSimCollectionSeed associateRecoToSim(edm::Handle<TrajectorySeedCollection>&, 
							   edm::Handle<TrackingParticleCollection>&, 
							   const edm::Event * event = 0) const {
    reco::RecoToSimCollectionSeed empty;
    return empty;
  }
  
  virtual reco::SimToRecoCollectionSeed associateSimToReco(edm::Handle<TrajectorySeedCollection>&, 
							   edm::Handle<TrackingParticleCollection>&, 
							   const edm::Event * event = 0) const {
    reco::SimToRecoCollectionSeed empty;
    return empty;
  }

  //TrackCandidate
  virtual reco::RecoToSimCollectionTCandidate associateRecoToSim(edm::Handle<TrackCandidateCollection>&, 
								 edm::Handle<TrackingParticleCollection>&, 
								 const edm::Event * event = 0) const {
    reco::RecoToSimCollectionTCandidate empty;
    return empty;
  }
  
  virtual reco::SimToRecoCollectionTCandidate associateSimToReco(edm::Handle<TrackCandidateCollection>&, 
								 edm::Handle<TrackingParticleCollection>&, 
								 const edm::Event * event = 0) const {
    reco::SimToRecoCollectionTCandidate empty;
    return empty;
  }

};


#endif
