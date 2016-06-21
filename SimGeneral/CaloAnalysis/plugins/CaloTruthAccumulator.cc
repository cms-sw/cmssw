// S Zenz, May 2016
// Loosely based on TrackingTruthAccumulator (M Grimes)

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/CaloAnalysis/plugins/CaloTruthAccumulator.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

CaloTruthAccumulator::CaloTruthAccumulator( const edm::ParameterSet & config, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
		messageCategory_("CaloTruthAccumulator"),
		maximumPreviousBunchCrossing_( config.getParameter<unsigned int>("maximumPreviousBunchCrossing") ),
		maximumSubsequentBunchCrossing_( config.getParameter<unsigned int>("maximumSubsequentBunchCrossing") ),
		simTrackLabel_( config.getParameter<edm::InputTag>("simTrackCollection") ),
		simVertexLabel_( config.getParameter<edm::InputTag>("simVertexCollection") ),
		collectionTags_( ),
		genParticleLabel_( config.getParameter<edm::InputTag>("genParticleCollection") ),
		hepMCproductLabel_( config.getParameter<edm::InputTag>("HepMCProductLabel") )
{
  barcodeLogicWarningAlready_ = false;

  mixMod.produces<CaloParticleCollection>();
  
  iC.consumes<std::vector<SimTrack> >(simTrackLabel_);
  iC.consumes<std::vector<SimVertex> >(simVertexLabel_);
  iC.consumes<std::vector<reco::GenParticle> >(genParticleLabel_);
  iC.consumes<std::vector<int> >(genParticleLabel_);
  iC.consumes<std::vector<int> >(hepMCproductLabel_);
  
  // Fill the collection tags
  const edm::ParameterSet& simHitCollectionConfig=config.getParameterSet("simHitCollections");
  std::vector<std::string> parameterNames=simHitCollectionConfig.getParameterNames();
  
  for( const auto& parameterName : parameterNames )
    {
      std::vector<edm::InputTag> tags=simHitCollectionConfig.getParameter<std::vector<edm::InputTag> >(parameterName);
      collectionTags_.insert(collectionTags_.end(), tags.begin(), tags.end());
    }
  
  for( const auto& collectionTag : collectionTags_ ) {
    iC.consumes<std::vector<PCaloHit> >(collectionTag);
  }
}

void CaloTruthAccumulator::initializeEvent( edm::Event const& event, edm::EventSetup const& setup )
{
  /*
	if( createUnmergedCollection_ )
	{
		unmergedOutput_.pTrackingParticles.reset( new TrackingParticleCollection );
		unmergedOutput_.pTrackingVertices.reset( new TrackingVertexCollection );
		unmergedOutput_.refTrackingParticles=const_cast<edm::Event&>( event ).getRefBeforePut<TrackingParticleCollection>();
		unmergedOutput_.refTrackingVertexes=const_cast<edm::Event&>( event ).getRefBeforePut<TrackingVertexCollection>();
	}

	if( createMergedCollection_ )
	{
		mergedOutput_.pTrackingParticles.reset( new TrackingParticleCollection );
		mergedOutput_.pTrackingVertices.reset( new TrackingVertexCollection );
		mergedOutput_.refTrackingParticles=const_cast<edm::Event&>( event ).getRefBeforePut<TrackingParticleCollection>("MergedTrackTruth");
		mergedOutput_.refTrackingVertexes=const_cast<edm::Event&>( event ).getRefBeforePut<TrackingVertexCollection>("MergedTrackTruth");
	}

	if( createInitialVertexCollection_ )
	{
		pInitialVertices_.reset( new TrackingVertexCollection );
	}
  */
}

/// create handle to edm::HepMCProduct here because event.getByLabel with edm::HepMCProduct only works for edm::Event
/// but not for PileUpEventPrincipal; PileUpEventPrincipal::getByLabel tries to call T::value_type and T::iterator
/// (where T is the type of the object one wants to get a handle to) which is only implemented for container-like objects
/// like std::vector but not for edm::HepMCProduct!

void CaloTruthAccumulator::accumulate( edm::Event const& event, edm::EventSetup const& setup )
{
	// Call the templated version that does the same for both signal and pileup events
	
	edm::Handle< edm::HepMCProduct > hepmc;
	event.getByLabel(hepMCproductLabel_, hepmc);
	
	edm::LogInfo(messageCategory_) << " CaloTruthAccumulator::accumulate (signal)";
	accumulateEvent( event, setup, hepmc );
}

void CaloTruthAccumulator::accumulate( PileUpEventPrincipal const& event, edm::EventSetup const& setup, edm::StreamID const& )
{
	// If this bunch crossing is outside the user configured limit, don't do anything.
	if( event.bunchCrossing()>=-static_cast<int>(maximumPreviousBunchCrossing_) && event.bunchCrossing()<=static_cast<int>(maximumSubsequentBunchCrossing_) )
	{
		//edm::LogInfo(messageCategory_) << "Analysing pileup event for bunch crossing " << event.bunchCrossing();
		
		//simply create empty handle as we do not have a HepMCProduct in PU anyway
		edm::Handle< edm::HepMCProduct > hepmc;
		edm::LogInfo(messageCategory_) << " CaloTruthAccumulator::accumulate (pileup) bunchCrossing=" << event.bunchCrossing();
		accumulateEvent( event, setup, hepmc );
	}
	else edm::LogInfo(messageCategory_) << "Skipping pileup event for bunch crossing " << event.bunchCrossing();
}

void CaloTruthAccumulator::finalizeEvent( edm::Event& event, edm::EventSetup const& setup )
{

  /*
	if( createUnmergedCollection_ )
	{
		edm::LogInfo("CaloTruthAccumulator") << "Adding " << unmergedOutput_.pTrackingParticles->size() << " TrackingParticles and " << unmergedOutput_.pTrackingVertices->size()
				<< " TrackingVertexs to the event.";

		event.put( unmergedOutput_.pTrackingParticles );
		event.put( unmergedOutput_.pTrackingVertices );
	}

	if( createMergedCollection_ )
	{
		edm::LogInfo("CaloTruthAccumulator") << "Adding " << mergedOutput_.pTrackingParticles->size() << " merged TrackingParticles and " << mergedOutput_.pTrackingVertices->size()
				<< " merged TrackingVertexs to the event.";

		event.put( mergedOutput_.pTrackingParticles, "MergedTrackTruth" );
		event.put( mergedOutput_.pTrackingVertices, "MergedTrackTruth" );
	}

	if( createInitialVertexCollection_ )
	{
		edm::LogInfo("CaloTruthAccumulator") << "Adding " << pInitialVertices_->size() << " initial TrackingVertexs to the event.";

		event.put( pInitialVertices_, "InitialVertices" );
	}
  */
}

//void CaloTruthAccumulator::beginLuminosityBlock( LuminosityBlock const& iLumiBlock, const EventSetup& iSetup ) {
//  iSetup.get<CaloGeometryRecord>().get(geoHandle_);
//  iSetup.get<IdealGeometryRecord>().get("HGCalEESensitive",hgceeGeoHandle_) ; 
//  iSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",hgchefGeoHandle_) ; 
//  iSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive",hgchebGeoHandle_) ; 
//}

template<class T> void CaloTruthAccumulator::accumulateEvent( const T& event, const edm::EventSetup& setup, const edm::Handle< edm::HepMCProduct >& hepMCproduct)
{
	//
	// Get the collections
	//
	edm::Handle<std::vector<SimTrack> > hSimTracks;
	edm::Handle<std::vector<SimVertex> > hSimVertices;
	edm::Handle< std::vector<reco::GenParticle> > hGenParticles;
	edm::Handle< std::vector<int> > hGenParticleIndices;

	event.getByLabel( simTrackLabel_, hSimTracks );
	event.getByLabel( simVertexLabel_, hSimVertices );

	std::cout << "CaloTruthAccumulator::accumulateEvent  hSimTracks size: " << hSimTracks->size() << std::endl;
	std::cout << "CaloTruthAccumulator::accumulateEvent  hSimVertices size: " << hSimVertices->size() << std::endl;

	try
	{
		event.getByLabel( genParticleLabel_, hGenParticles );
		event.getByLabel( genParticleLabel_, hGenParticleIndices );
		std::cout << "CaloTruthAccumulator::accumulateEvent  hGenParticles size: " << hGenParticles->size() << std::endl;
	}
	catch( cms::Exception& exception )
	{
		//
		// The Monte Carlo is not always available, e.g. for pileup events. The information
		// is only used if it's available, but for some reason the PileUpEventPrincipal
		// wrapper throws an exception here rather than waiting to see if the handle is
		// used (as is the case for edm::Event). So I just want to catch this exception
		// and use the normal handle checking later on.
		//
	}

	std::vector<const PCaloHit*> simHitPointers;
        fillSimHits( simHitPointers, event, setup ); 

	std::cout << "CaloTruthAccumulator::accumulateEvent  simHitPointers size: " << simHitPointers.size() << std::endl;

	// Clear maps from previous event fill them for this one
	m_simHitBarcodeToIndex.clear();
	for (unsigned int i = 0 ; i < simHitPointers.size(); i++) {
	  m_simHitBarcodeToIndex.emplace(simHitPointers[i]->geantTrackId(),i);
	}
	m_genParticleBarcodeToIndex.clear();
	for (unsigned int i = 0 ; i < hGenParticles->size() ; i++) {
	  m_genParticleBarcodeToIndex.emplace(hGenParticleIndices->at(i),i);
	}
	m_genBarcodeToSimTrackIndex.clear();
	m_simVertexBarcodeToSimTrackBarcode.clear();
	m_simTrackBarcodeToIndex.clear();
	for (unsigned int i = 0 ; i < hSimTracks->size() ; i++) {
	  if( !hSimTracks->at(i).noGenpart() ) {
	    m_genBarcodeToSimTrackIndex.emplace(hSimTracks->at(i).genpartIndex(), i);
	  }
	  if( !hSimTracks->at(i).noVertex() ) {
	    m_simVertexBarcodeToSimTrackBarcode.emplace(hSimTracks->at(i).vertIndex(), hSimTracks->at(i).trackId());
	  }
	  m_simTrackBarcodeToIndex.emplace(hSimTracks->at(i).trackId(), i);
	}
	m_simVertexBarcodes.clear();
	m_simVertexBarcodeToIndex.clear();
	m_simTrackToSimVertex.clear();
	m_simVertexToSimTrackParent.clear();
        for (unsigned int i = 0 ; i < hSimVertices->size() ; i++) {
	  m_simVertexBarcodes.push_back(i);
          m_simVertexBarcodeToIndex.emplace(hSimVertices->at(i).vertexId(), i);
	  if (!hSimVertices->at(i).noParent()) {
	    m_simTrackToSimVertex.emplace(hSimVertices->at(i).parentIndex(), i);
	    m_simVertexToSimTrackParent.emplace( hSimVertices->at(i).vertexId(), hSimVertices->at(i).parentIndex() );
	  }
	}
	
	bool hitdisplayed = false;
	for (auto & hit : simHitPointers) {
	  int subdet, layer, cell, sec, subsec, zp;
	  uint32_t simId = hit->id();
	  HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell); 
	  DetId id = HGCalDetId((ForwardSubdetector)subdet,zp,layer,subsec,sec,cell);
	  if (!hitdisplayed) {
	    std::cout << "First hit subdet, layer, cell, sec, subsec, zp, id: " << subdet
		      << " " << layer << " " << cell << " " << sec << " " << subsec << " " << zp << " " << id.rawId() << std::endl;
	    hitdisplayed = true;
	  }
	}

	//	bool useGenParticles_ = false;
	double minEnergy_ = 5.;
	double maxPseudoRapidity_ = 5.;

	std::vector<Index_t> tracksToBecomeClustersInitial;
	std::vector<Barcode_t> descendantTracks;
	std::vector<std::unique_ptr<SimHitInfoPerSimTrack_t> > hitInfoList;
        for (unsigned int i = 0 ; i < hSimTracks->size() ; i++) {
	  if (hSimTracks->at(i).momentum().E() < minEnergy_ || fabs(hSimTracks->at(i).momentum().Eta()) >= maxPseudoRapidity_) continue;
	  std::cout << "TOP-LEVEL SCZ DEBUG BEORE " << i << std::endl;
	  auto dummy = CaloTruthAccumulator::descendantSimClusters( hSimTracks->at(i).trackId(),simHitPointers );
	  std::cout << "TOP-LEVEL SCZ DEBUG AFTER " << i << " descendantSimClusters size: " << dummy.size() << std::endl;
	  /*
	  if (useGenParticles_) {
	    if (!hSimTracks->at(i).noGenpart() ) {
	      tracksToBecomeClustersInitial.push_back(i);
	      hitInfoList.emplace_back(CaloTruthAccumulator::allAttachedSimHitInfo(hSimTracks->at(i).trackId(),simHitPointers,false));
	    }
	  } else { // use particles hitting calorimeter
	    std::unique_ptr<SimHitInfoPerSimTrack_t> hit_info = std::move(CaloTruthAccumulator::attachedSimHitInfo(hSimTracks->at(i).trackId(),simHitPointers, false));
	    std::unique_ptr<SimHitInfoPerSimTrack_t> inclusive_hit_info = std::move(CaloTruthAccumulator::allAttachedSimHitInfo(hSimTracks->at(i).trackId(),simHitPointers, false));
            if (hit_info->size() > 0 && hit_info->size() != inclusive_hit_info->size()) {
	      std::cout << " DEBUG SURPRISE! SIZES DIFFER! " << hit_info->size() << " " << inclusive_hit_info->size() << std::endl;
            }
	    if (hit_info->size() > 0) {
	      std::cout << " push_back " << i << std::endl;
	      tracksToBecomeClustersInitial.push_back(i);
	      auto descendants = CaloTruthAccumulator::descendantTrackBarcodes(hSimTracks->at(i).trackId());
	      descendantTracks.insert(descendantTracks.end(),descendants.begin(),descendants.end());
	      hitInfoList.push_back(std::move(hit_info));
	      std::cout << " pushed back " << i << std::endl;
	    }
	  }
	  */
	}

	/*
	std::cout << "DESCENDANT TRACKS:";
        for (unsigned i = 0 ; i < descendantTracks.size() ; i++) std::cout << " " << descendantTracks[i];
	std::cout << std::endl;

	std::cout << "TRACKS TO BECOME CLUSTERS (BEFORE):";
	for (unsigned i = 0 ; i < tracksToBecomeClustersInitial.size() ; i++) std::cout << " " << tracksToBecomeClustersInitial[i];
	std::cout << std::endl;

	std::vector<Index_t> tracksToBecomeClusters;
	for (unsigned i = 0 ; i < tracksToBecomeClustersInitial.size() ; i++) {
          Barcode_t trackId = hSimTracks->at(i).trackId();
	  if ( std::find(descendantTracks.begin(), descendantTracks.end(), trackId) != descendantTracks.end()  ) {
	    std::cout << " removing track " << i << " with id e eta phi " << trackId << " " << hSimTracks->at(i).momentum().E() 
		      << " " << hSimTracks->at(i).momentum().Eta() << " " << hSimTracks->at(i).momentum().Phi() << " "<< hSimTracks->at(i).type()
		      << " because it is a descendant" << std::endl;
	  } else {
	    tracksToBecomeClusters.push_back( i );
	  }
	}

	std::cout << "TRACKS TO BECOME CLUSTERS (AFTER):";
        for (unsigned i = 0 ; i < tracksToBecomeClusters.size() ; i++) std::cout << " " << tracksToBecomeClustersInitial[i];
	std::cout << std::endl;

	for (unsigned i = 0 ; i < tracksToBecomeClusters.size() ; i++) {
	  Barcode_t trackId = hSimTracks->at(i).trackId();
	  std::cout << "  TRACK " << i << ": " << trackId << " " << hSimTracks->at(i).momentum().E() << " " << hSimTracks->at(i).momentum().Eta() << " "
                    << hSimTracks->at(i).momentum().Phi() << " "<< hSimTracks->at(i).type() << std::endl;
	  for ( auto iter = hitInfoList[i]->begin() ; iter != hitInfoList[i]->end() ; iter++) {
	    std::cout << "    HIT " << iter->first.rawId() << " " << iter->second << std::endl;
	  }
	}
	*/	


	/*
	// I only want to create these collections if they're actually required
	std::auto_ptr< ::OutputCollectionWrapper> pUnmergedCollectionWrapper;
	std::auto_ptr< ::OutputCollectionWrapper> pMergedCollectionWrapper;
	if( createUnmergedCollection_ ) pUnmergedCollectionWrapper.reset( new ::OutputCollectionWrapper( decayChain, unmergedOutput_ ) );
	if( createMergedCollection_ ) pMergedCollectionWrapper.reset( new ::OutputCollectionWrapper( decayChain, mergedOutput_ ) );

	std::vector<const PCaloHit*> simHitPointers;
	fillSimHits( simHitPointers, event, setup );
	TrackingParticleFactory objectFactory( decayChain, hGenParticles, hepMCproduct, hGenParticleIndices, simHitPointers, volumeRadius_, volumeZ_, vertexDistanceCut_, allowDifferentProcessTypeForDifferentDetectors_ );

	// While I'm testing, perform some checks.
	// TODO - drop this call once I'm happy it works in all situations.
	//decayChain.integrityCheck();

	TrackingParticleSelector* pSelector=NULL;
	if( selectorFlag_ ) pSelector=&selector_;

	// Run over all of the SimTracks, but because I'm interested in the decay hierarchy
	// do it through the DecayChainTrack objects. These are looped over in sequence here
	// but they have the hierarchy information for the functions called to traverse the
	// decay chain.

	for( size_t index=0; index<decayChain.decayTracksSize; ++index )
	{
		::DecayChainTrack* pDecayTrack=&decayChain.decayTracks[index];
		const SimTrack& simTrack=hSimTracks->at(pDecayTrack->simTrackIndex);


		// Perform some quick checks to see if we can drop out early. Note that these are
		// a subset of the cuts in the selector_ so the created TrackingParticle could still
		// fail. The selector_ requires the full TrackingParticle to be made however, which
		// can be computationally expensive.
		if( chargedOnly_ && simTrack.charge()==0 ) continue;
		if( signalOnly_ && (simTrack.eventId().bunchCrossing()!=0 || simTrack.eventId().event()!=0) ) continue;

		// Also perform a check to see if the production vertex is inside the tracker volume (if required).
		if( ignoreTracksOutsideVolume_ )
		{
			const SimVertex& simVertex=hSimVertices->at( pDecayTrack->pParentVertex->simVertexIndex );
			if( !objectFactory.vectorIsInsideVolume( simVertex.position() ) ) continue;
		}


		// This function creates the TrackinParticle and adds it to the collection if it
		// passes the selection criteria specified in the configuration. If the config
		// specifies adding ancestors, the function is called recursively to do that.
		::addTrack( pDecayTrack, pSelector, pUnmergedCollectionWrapper.get(), pMergedCollectionWrapper.get(), objectFactory, addAncestors_, tTopo );
	}

	// If configured to create a collection of initial vertices, add them from this bunch
	// crossing. No selection is applied on this collection, but it also has no links to
	// the TrackingParticle decay products.
	// There are a lot of "initial vertices", I'm not entirely sure what they all are
	// (nuclear interactions in the detector maybe?), but the one for the main event is
	// the one with vertexId==0.
	if( createInitialVertexCollection_ )
	{
		// Pretty sure the one with vertexId==0 is always the first one, but doesn't hurt to check
		for( const auto& pRootVertex : decayChain.rootVertices )
		{
			const SimVertex& vertex=hSimVertices->at(decayChain.rootVertices[0]->simVertexIndex);
			if( vertex.vertexId()!=0 ) continue;

			pInitialVertices_->push_back( objectFactory.createTrackingVertex(pRootVertex) );
			break;
		}
	}
	*/
}

std::vector<Barcode_t> CaloTruthAccumulator::descendantTrackBarcodes( Barcode_t barcode ) {
  std::vector<Barcode_t> result;
  if (m_simTrackToSimVertex.count(barcode)) {
    auto vertex_range = m_simTrackToSimVertex.equal_range(barcode);
    for ( auto vertex_iter = vertex_range.first ; vertex_iter != vertex_range.second ; vertex_iter++ ) {
      Index_t decayVertexIndex = vertex_iter->second;
      Barcode_t decayVertexBarcode = m_simVertexBarcodes[decayVertexIndex];
      auto track_range = m_simVertexBarcodeToSimTrackBarcode.equal_range( decayVertexBarcode );
      for ( auto track_iter = track_range.first ; track_iter != track_range.second ; track_iter++ ) {
	std::cout << " CaloTruthAccumulator::descendantTrackBarcodes push_back " << track_iter->second << std::endl;
	result.push_back( track_iter->second );
	std::vector<Barcode_t> daughter_result = CaloTruthAccumulator::descendantTrackBarcodes( track_iter->second );
	result.insert(result.end(),daughter_result.begin(),daughter_result.end());
      }
    }
  }
  return result;
}

SimClusterCollection CaloTruthAccumulator::descendantSimClusters( Barcode_t barcode, const std::vector<const PCaloHit*>& hits ) {
  SimClusterCollection result;
  if (CaloTruthAccumulator::consideredBarcode( barcode )) {
    std::cout << "SCZ DEBUG Ignoring descendantSimClusters call because this particle is already marked used: " << barcode << std::endl;
  }
  std::unique_ptr<SimHitInfoPerSimTrack_t> hit_info = std::move(CaloTruthAccumulator::attachedSimHitInfo(barcode,hits, false));
  std::cout << " Special SCZ DEBUG call of inclusive_hit_info on barcode " << barcode << "..." << std::endl;
  std::unique_ptr<SimHitInfoPerSimTrack_t> inclusive_hit_info = std::move(CaloTruthAccumulator::allAttachedSimHitInfo(barcode,hits, false) );
  std::cout << " After Special SCZ DEBUG call of inclusive_hit_info on barcode " << barcode 
	    << "... inclusive_hit_info->size()=" << inclusive_hit_info->size() << std::endl;
  if (hit_info->size() > 0) {
    //    std::unique_ptr<SimHitInfoPerSimTrack_t> inclusive_hit_info = std::move(CaloTruthAccumulator::allAttachedSimHitInfo(barcode,hits, true) );
    std::cout << " SCZ DEBUG we should make a SimCluster out of particle: " << barcode << std::endl;
  } else {
    if (m_simTrackToSimVertex.count(barcode)) {
      auto vertex_range = m_simTrackToSimVertex.equal_range(barcode);
      for ( auto vertex_iter = vertex_range.first ; vertex_iter != vertex_range.second ; vertex_iter++ ) {
	Index_t decayVertexIndex = vertex_iter->second;
	Barcode_t decayVertexBarcode = m_simVertexBarcodes[decayVertexIndex];
	auto track_range = m_simVertexBarcodeToSimTrackBarcode.equal_range( decayVertexBarcode );
	for ( auto track_iter = track_range.first ; track_iter != track_range.second ; track_iter++ ) {
	  std::cout << "     SCZ DEBUG From mother " << barcode << ", calling CaloTruthAccumulator::descendantSimClusters on daughter: " 
		    << track_iter->second << std::endl;
	  auto daughter_result = CaloTruthAccumulator::descendantSimClusters(track_iter->second,hits);
	  result.insert(result.end(),daughter_result.begin(),daughter_result.end());
	}
      }
    }
  }
  return result;
}


  /*
  if (m_simTrackToSimVertex.count(barcode)) {
    auto vertex_range = m_simTrackToSimVertex.equal_range(barcode);
    for ( auto vertex_iter = vertex_range.first ; vertex_iter != vertex_range.second ; vertex_iter++ ) {
      Index_t decayVertexIndex = vertex_iter->second;
      Barcode_t decayVertexBarcode = m_simVertexBarcodes[decayVertexIndex];
      auto track_range = m_simVertexBarcodeToSimTrackBarcode.equal_range( decayVertexBarcode );
      for ( auto track_iter = track_range.first ; track_iter != track_range.second ; track_iter++ ) {
	std::unique_ptr<SimHitInfoPerSimTrack_t> hit_info = std::move(CaloTruthAccumulator::attachedSimHitInfo(track_iter->second,hits, false));
	if (hit_info->size() > 0) {
	  std::unique_ptr<SimHitInfoPerSimTrack_t> inclusive_hit_info = std::move(CaloTruthAccumulator::allAttachedSimHitInfo(track_iter->second,hits, true) );
	  std::cout << " SCZ DEBUG we should make a SimCluster out of particle: " << barcode << std::endl;
	}
      }
    }
  }
  return result;
}
  */

std::unique_ptr<SimHitInfoPerSimTrack_t> CaloTruthAccumulator::attachedSimHitInfo( Barcode_t barcode , const std::vector<const PCaloHit*>& hits, 
										   bool includeOwn , bool includeOther, bool markUsed ) {
  std::unique_ptr<SimHitInfoPerSimTrack_t> result(new SimHitInfoPerSimTrack_t);
  std::cout << " SCZ DEBUG CaloTruthAccumulator::attachedSimHitInfo " << barcode << " " << includeOwn << " " << includeOther << " " << markUsed << std::endl;
  /*
  if (barcode == 417 || barcode == 418) {
    std::cout << "CaloTruthAccumulator::attachedSimHitInfo DEBUG " << std::endl;
    std::cout << "m_simTrackToSimVertex.count(barcode) " << m_simTrackToSimVertex.count(barcode) << std::endl;
    auto vertex_range = m_simTrackToSimVertex.equal_range(barcode);
    for ( auto vertex_iter = vertex_range.first ; vertex_iter != vertex_range.second ; vertex_iter++ ) {
      Index_t decayVertexIndex = vertex_iter->second;
      std::cout << "   decayVertexIndex " << decayVertexIndex << std::endl;
      Barcode_t decayVertexBarcode = m_simVertexBarcodes[decayVertexIndex];
      std::cout<< "   decayVertexBarcode " << decayVertexBarcode << std::endl;
      auto track_range = m_simVertexBarcodeToSimTrackBarcode.equal_range( decayVertexBarcode );
      for ( auto track_iter = track_range.first ; track_iter != track_range.second ; track_iter++ ) {
	std::cout << "         Daugher barcode: " << track_iter->second << std::endl;
      }
    }
  }
  */
  if ( markUsed ) {
    std::cout << " SCZ DEBUG CaloTruthAccumulator::attachedSimHitInfo markUsed " << std::endl;
    if ( CaloTruthAccumulator::consideredBarcode( barcode ) ) {
      return result;
    }
    CaloTruthAccumulator::setConsideredBarcode( barcode );
  }
  if (includeOwn) {
    auto range = m_simHitBarcodeToIndex.equal_range( barcode );
    unsigned n = 0;
    for ( auto iter = range.first ; iter != range.second ; iter++ ) {
      int subdet, layer, cell, sec, subsec, zp;
      uint32_t simId = hits[iter->second]->id();
      HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);
      DetId id = HGCalDetId((ForwardSubdetector)subdet,zp,layer,subsec,sec,cell);
      result->emplace_back(id,hits[iter->second]->energy());
      n++;
    }
    if (n > 0) std::cout << " SCZ DEBUG inside CaloTruthAccumulator::attachedSimHitInfo includeOwn and we have " 
			 << n << " hits for barcode " << barcode << std::endl;
  }
  if (includeOther) {
    if (m_simTrackToSimVertex.count(barcode)) {
      auto vertex_range = m_simTrackToSimVertex.equal_range(barcode);
      for ( auto vertex_iter = vertex_range.first ; vertex_iter != vertex_range.second ; vertex_iter++ ) {
	Index_t decayVertexIndex = vertex_iter->second;
	Barcode_t decayVertexBarcode = m_simVertexBarcodes[decayVertexIndex];
	auto track_range = m_simVertexBarcodeToSimTrackBarcode.equal_range( decayVertexBarcode );
	for ( auto track_iter = track_range.first ; track_iter != track_range.second ; track_iter++ ) {
	  if( !barcodeLogicWarningAlready_ && track_iter->second < barcode ) {
	    barcodeLogicWarningAlready_ = true;
	    edm::LogWarning(messageCategory_) << " Daughter particle has a lower barcode than parent. This may screw up the logic!" << std::endl;
	  }
	  std::cout << "     SCZ DEBUG From mother " << barcode << ", calling CaloTruthAccumulator::allAttachedSimHitInfo on daughter: "
                    << track_iter->second << std::endl;
	  std::unique_ptr<SimHitInfoPerSimTrack_t> daughter_result = std::move(CaloTruthAccumulator::allAttachedSimHitInfo(track_iter->second,hits,markUsed));
	  result->insert(result->end(),daughter_result->begin(),daughter_result->end());
	}
      }
    }
  }
  std::cout << " SCZ DEBUG CaloTruthAccumulator::attachedSimHitInfo " << barcode << " " << includeOwn << " " << includeOther << " " << markUsed 
	    << " " << result->size() << std::endl;
  return result;
}

std::unique_ptr<SimHitInfoPerSimTrack_t> CaloTruthAccumulator::descendantOnlySimHitInfo( Barcode_t barcode, const std::vector<const PCaloHit*>& hits, bool markUsed) {
  return CaloTruthAccumulator::attachedSimHitInfo(barcode,hits,false,true,markUsed);
}

std::unique_ptr<SimHitInfoPerSimTrack_t> CaloTruthAccumulator::allAttachedSimHitInfo( Barcode_t barcode, const std::vector<const PCaloHit*>& hits, bool markUsed ) {
  return CaloTruthAccumulator::attachedSimHitInfo(barcode,hits,true,true,markUsed);
}

template<class T> void CaloTruthAccumulator::fillSimHits( std::vector<const PCaloHit*>& returnValue, const T& event, const edm::EventSetup& setup )
{

	// loop over the collections
	for( const auto& collectionTag : collectionTags_ )
	{
		edm::Handle< std::vector<PCaloHit> > hSimHits;
		event.getByLabel( collectionTag, hSimHits );
		for( const auto& simHit : *hSimHits )
		{
			returnValue.push_back( &simHit );
		}
	} // end of loop over InputTags
}

// Register with the framework
DEFINE_DIGI_ACCUMULATOR (CaloTruthAccumulator);
