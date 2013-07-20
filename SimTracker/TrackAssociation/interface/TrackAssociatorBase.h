#ifndef TrackAssociatorBase_h
#define TrackAssociatorBase_h

/** \class TrackAssociatorBase
 *  Base class for TrackAssociators. Methods take as input the handle of Track and TrackingPArticle collections and return an AssociationMap (oneToManyWithQuality)
 *
 *  $Date: 2013/03/12 13:38:08 $
 *  $Revision: 1.19 $
 *  \author magni, cerati
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

namespace reco{
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric
    <TrackingParticleCollection, edm::View<TrajectorySeed>, double> >
    SimToRecoCollectionSeed;  
  typedef edm::AssociationMap<edm::OneToManyWithQualityGeneric 
    <edm::View<TrajectorySeed>, TrackingParticleCollection, double> >
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
						       const edm::Event * event ,
                                                       const edm::EventSetup * setup ) const {
    edm::RefToBaseVector<reco::Track> tc(tCH);
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(edm::RefToBase<reco::Track>(tCH,j));

    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));

    return associateRecoToSim(tc,tpc,event,setup);
  }
  
  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  virtual reco::SimToRecoCollection associateSimToReco(edm::Handle<edm::View<reco::Track> >& tCH, 
						       edm::Handle<TrackingParticleCollection>& tPCH,
						       const edm::Event * event ,
                                                       const edm::EventSetup * setup ) const {
    edm::RefToBaseVector<reco::Track> tc(tCH);
    for (unsigned int j=0; j<tCH->size();j++)
      tc.push_back(edm::RefToBase<reco::Track>(tCH,j));

    edm::RefVector<TrackingParticleCollection> tpc(tPCH.id());
    for (unsigned int j=0; j<tPCH->size();j++)
      tpc.push_back(edm::Ref<TrackingParticleCollection>(tPCH,j));

    return associateSimToReco(tc,tpc,event,setup);
  }  
  
  /// Association Reco To Sim with Collections
  virtual  reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track> & tc,
                                                        const edm::RefVector<TrackingParticleCollection>& tpc,
                                                        const edm::Event * event ,
                                                        const edm::EventSetup * setup  ) const = 0 ;
  /// Association Sim To Reco with Collections
  virtual  reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track> & tc,
                                                        const edm::RefVector<TrackingParticleCollection>& tpc ,
                                                        const edm::Event * event ,
                                                        const edm::EventSetup * setup  ) const = 0 ; 

  //TrajectorySeed
  virtual reco::RecoToSimCollectionSeed associateRecoToSim(edm::Handle<edm::View<TrajectorySeed> >&, 
							   edm::Handle<TrackingParticleCollection>&, 
							   const edm::Event * event ,
                                                           const edm::EventSetup * setup ) const {
    reco::RecoToSimCollectionSeed empty;
    return empty;
  }
  
  virtual reco::SimToRecoCollectionSeed associateSimToReco(edm::Handle<edm::View<TrajectorySeed> >&, 
							   edm::Handle<TrackingParticleCollection>&, 
							   const edm::Event * event ,
                                                           const edm::EventSetup * setup ) const {
    reco::SimToRecoCollectionSeed empty;
    return empty;
  }

  //TrackCandidate
  virtual reco::RecoToSimCollectionTCandidate associateRecoToSim(edm::Handle<TrackCandidateCollection>&, 
								 edm::Handle<TrackingParticleCollection>&, 
								 const edm::Event * event ,
                                                                 const edm::EventSetup * setup ) const {
    reco::RecoToSimCollectionTCandidate empty;
    return empty;
  }
  
  virtual reco::SimToRecoCollectionTCandidate associateSimToReco(edm::Handle<TrackCandidateCollection>&, 
								 edm::Handle<TrackingParticleCollection>&, 
								 const edm::Event * event ,
                                                                 const edm::EventSetup * setup ) const {
    reco::SimToRecoCollectionTCandidate empty;
    return empty;
  }

};


#endif
