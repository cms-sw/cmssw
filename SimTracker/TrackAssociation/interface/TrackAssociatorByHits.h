#ifndef TrackAssociatorByHits_h
#define TrackAssociatorByHits_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

//reco track
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

class TrackAssociatorByHits : public TrackAssociatorBase {
  
 public:

  enum SimToRecoDenomType {denomnone,denomsim,denomreco};

  explicit TrackAssociatorByHits( const edm::ParameterSet& );  
  ~TrackAssociatorByHits();
  
  /* Associate SimTracks to RecoTracks By Hits */
 
  /// Association Reco To Sim with Collections
  virtual
  reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event ,
                                               const edm::EventSetup * setup  ) const override;
  /// Association Sim To Reco with Collections
  virtual
  reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&,
					       const edm::Event * event ,
                                               const edm::EventSetup * setup  ) const override;
  
  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  virtual
  reco::RecoToSimCollection associateRecoToSim(edm::Handle<edm::View<reco::Track> >& tCH, 
					       edm::Handle<TrackingParticleCollection>& tPCH, 
					       const edm::Event * event ,
                                               const edm::EventSetup * setup ) const override {
    return TrackAssociatorBase::associateRecoToSim(tCH,tPCH,event,setup);
  }
  
  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  virtual
  reco::SimToRecoCollection associateSimToReco(edm::Handle<edm::View<reco::Track> >& tCH, 
					       edm::Handle<TrackingParticleCollection>& tPCH,
					       const edm::Event * event ,
                                               const edm::EventSetup * setup ) const override {
    return TrackAssociatorBase::associateSimToReco(tCH,tPCH,event,setup);
  }  

  //seed
  virtual
  reco::RecoToSimCollectionSeed associateRecoToSim(edm::Handle<edm::View<TrajectorySeed> >&, 
						   edm::Handle<TrackingParticleCollection>&, 
						   const edm::Event * event ,
                                                   const edm::EventSetup * setup ) const override;
  
  virtual
  reco::SimToRecoCollectionSeed associateSimToReco(edm::Handle<edm::View<TrajectorySeed> >&, 
						   edm::Handle<TrackingParticleCollection>&, 
						   const edm::Event * event ,
                                                   const edm::EventSetup * setup ) const override;
  template<typename iter>
  void getMatchedIds(std::vector<SimHitIdpr>&, 
		     std::vector<SimHitIdpr>&, 
		     int&, 
		     iter,
		     iter,
		     TrackerHitAssociator*) const;
  
  int getShared(std::vector<SimHitIdpr>&, 
		std::vector<SimHitIdpr>&,
		TrackingParticle const& ) const;

  template<typename iter>
  int getDoubleCount(iter,iter,TrackerHitAssociator*,TrackingParticle const&) const;

 private:
  // ----- member data
  const edm::ParameterSet& conf_;
  const bool AbsoluteNumberOfHits;
  SimToRecoDenomType SimToRecoDenominator;
  const double quality_SimToReco;
  const double purity_SimToReco;
  const double cut_RecoToSim;
  const bool UsePixels;
  const bool UseGrouped;
  const bool UseSplitting;
  const bool ThreeHitTracksAreSpecial;

  const TrackingRecHit* getHitPtr(edm::OwnVector<TrackingRecHit>::const_iterator iter) const {return &*iter;}
  const TrackingRecHit* getHitPtr(trackingRecHit_iterator iter) const {return &**iter;}

  edm::InputTag _simHitTpMapTag;
};

#endif
