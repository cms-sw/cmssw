//
// modified & integrated by C. Battilana (INFN BO)
// from code by G. Abbiendi: SimMuon/MCTruth/plugins/MuonTrackSelector.cc
//
#include "SimMuon/MCTruth/plugins/HLTFilterToTrackProducer.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


HLTFilterToTrackProducer::HLTFilterToTrackProducer(const edm::ParameterSet& config) :
  m_trigEvToken(consumes<trigger::TriggerEvent>(config.getParameter< edm::InputTag >("trigEvTag"))),
  m_filterToken(consumes<trigger::TriggerFilterObjectWithRefs>(config.getParameter< edm::InputTag >("filterTag"))),
  m_filterName(config.getParameter< edm::InputTag >("filterTag").label())
{

  edm::LogVerbatim("HLTFilterToTrackProducer") << "constructing  HLTFilterToTrackProducer" << config.dump();

  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();
  produces<TrackingRecHitCollection>();

}

HLTFilterToTrackProducer::~HLTFilterToTrackProducer() 
{

}

void HLTFilterToTrackProducer::produce(edm::Event& event, const edm::EventSetup& iSetup) 
{

  std::unique_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
  std::unique_ptr<reco::TrackExtraCollection> selectedTrackExtras( new reco::TrackExtraCollection() );
  std::unique_ptr<TrackingRecHitCollection> selectedTrackHits( new TrackingRecHitCollection() );

  reco::TrackExtraRefProd refTrackExtras;
  TrackingRecHitRefProd refHits;

  refTrackExtras = event.template getRefBeforePut<reco::TrackExtraCollection>();
  refHits        = event.template getRefBeforePut<TrackingRecHitCollection>();
  

  edm::Handle<trigger::TriggerEvent> triggerEvent;
  event.getByToken(m_trigEvToken, triggerEvent);
  
  const trigger::size_type nFilters(triggerEvent->sizeFilters());
    
  bool hasFilterName = false;
  
  for (trigger::size_type iFilter=0; iFilter!=nFilters; ++iFilter) 
    {
        std::string filterName(triggerEvent->filterTag(iFilter).encode());

        if (filterName.find(m_filterName) != std::string::npos )
        {
            hasFilterName = true;
            break;
        }
            
    }
    
  if (hasFilterName)
  {

  edm::Handle<trigger::TriggerFilterObjectWithRefs> triggerFiltersWithRefs;
  event.getByToken(m_filterToken,triggerFiltersWithRefs);

  vector<reco::RecoChargedCandidateRef> triggerCands;
  triggerFiltersWithRefs->getObjects(trigger::TriggerMuon,triggerCands);

  for (auto const & triggerCand : triggerCands) 
    {

      auto const & triggerCandTkRef = triggerCand->track();

      triggerCand->track()->extra();

      //edm::LogVerbatim("HLTFilterToTrackProducer") 
      std::cout << "[HLTFilterToTrackProducer] Track reference pT " << triggerCandTkRef->pt()
		<< " eta : " << triggerCandTkRef->eta()
		<< " phi : " << triggerCandTkRef->phi()
		<< "\n";

      selectedTracks->push_back( (*triggerCandTkRef) );

      // TrackExtras
      selectedTrackExtras->push_back( reco::TrackExtra( triggerCandTkRef->outerPosition(), 
      							triggerCandTkRef->outerMomentum(), 
      							triggerCandTkRef->outerOk(),
      							triggerCandTkRef->innerPosition(), 
      							triggerCandTkRef->innerMomentum(), 
      							triggerCandTkRef->innerOk(),
      							triggerCandTkRef->outerStateCovariance(), 
      							triggerCandTkRef->outerDetId(),
      							triggerCandTkRef->innerStateCovariance(), 
      							triggerCandTkRef->innerDetId(),
      							triggerCandTkRef->seedDirection() 
      						      ) 
      				     );

      selectedTracks->back().setExtra( reco::TrackExtraRef( refTrackExtras, 
				       selectedTrackExtras->size() - 1) );


      // TrackingRecHits
      reco::TrackExtra & trackExtra = selectedTrackExtras->back();
      auto const firstHitIndex = selectedTrackHits->size();

      for( trackingRecHit_iterator hit  = triggerCandTkRef->recHitsBegin(); 
	                           hit != triggerCandTkRef->recHitsEnd(); 
	                           ++hit ) 
	{
	  selectedTrackHits->push_back( (*hit)->clone() );
	}

      trackExtra.setHits( refHits, firstHitIndex, 
			  selectedTrackHits->size() - firstHitIndex 
			);

      trackExtra.setTrajParams( triggerCandTkRef->extra()->trajParams(),
				triggerCandTkRef->extra()->chi2sX5()
			      );

      assert(trackExtra.trajParams().size() == trackExtra.recHitsSize());

    }
  }

  event.put(std::move(selectedTracks));
  event.put(std::move(selectedTrackExtras));
  event.put(std::move(selectedTrackHits));

}
