// S Zenz/L Gray, May 2016
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

#include <iterator>

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

  mixMod.produces<SimClusterCollection>("MergedCaloTruth");
  mixMod.produces<CaloParticleCollection>("MergedCaloTruth");
  
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
  output_.pSimClusters.reset( new SimClusterCollection() );
  output_.pCaloParticles.reset( new CaloParticleCollection() );

  m_detIdToCluster.clear();
  m_detIdToTotalSimEnergy.clear();
}

/// create handle to edm::HepMCProduct here because event.getByLabel with edm::HepMCProduct only works for edm::Event
/// but not for PileUpEventPrincipal; PileUpEventPrincipal::getByLabel tries to call T::value_type and T::iterator
/// (where T is the type of the object one wants to get a handle to) which is only implemented for container-like objects
/// like std::vector but not for edm::HepMCProduct!

void CaloTruthAccumulator::accumulate( edm::Event const& event, edm::EventSetup const& setup ) {
  // Call the templated version that does the same for both signal and pileup events
  
  edm::Handle< edm::HepMCProduct > hepmc;
  event.getByLabel(hepMCproductLabel_, hepmc);
  
  edm::LogInfo(messageCategory_) << " CaloTruthAccumulator::accumulate (signal)";
  accumulateEvent( event, setup, hepmc );
}

void CaloTruthAccumulator::accumulate( PileUpEventPrincipal const& event, edm::EventSetup const& setup, edm::StreamID const& ) {
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

void CaloTruthAccumulator::finalizeEvent( edm::Event& event, edm::EventSetup const& setup ) {
  edm::LogInfo("CaloTruthAccumulator") << "Adding " << output_.pSimClusters->size() 
				       << " SimParticles and " << output_.pCaloParticles->size()
				       << " CaloParticles to the event.";

  // now we need to normalize the hits and energies into hits and fractions
  // (since we have looped over all pileup events)

  for( auto& sc : *(output_.pSimClusters) ) {
    auto hitsAndEnergies = sc.hits_and_fractions();
    sc.clearHitsAndFractions();
    for( auto& hAndE : hitsAndEnergies ) {
      const float fraction = hAndE.second/m_detIdToTotalSimEnergy[hAndE.first];
      sc.addRecHitAndFraction(hAndE.first,fraction);
    }
  }
  
  // save the SimCluster orphan handle so we can fill the calo particles
  auto scHandle = event.put( std::move(output_.pSimClusters), "MergedCaloTruth" );
  
  // now fill the calo particles
  for( unsigned i = 0; i < output_.pCaloParticles->size(); ++i ) {
    auto& cp = (*output_.pCaloParticles)[i];
    for( unsigned j = m_caloParticles.sc_start_[i]; j < m_caloParticles.sc_stop_[i]; ++j ) {
      edm::Ref<SimClusterCollection> ref(scHandle,j);
      cp.addSimCluster(ref);
    }
  }  

  event.put( std::move(output_.pCaloParticles), "MergedCaloTruth" );

  calo_particles().swap(m_caloParticles);

  std::unordered_map<Index_t,float>().swap(m_detIdToTotalSimEnergy);

  std::unordered_map<Barcode_t,Index_t>().swap(m_genParticleBarcodeToIndex);
  std::unordered_map<Barcode_t,Index_t>().swap(m_simTrackBarcodeToIndex);
  std::unordered_map<Barcode_t,Index_t>().swap(m_genBarcodeToSimTrackIndex);
  std::unordered_map<Barcode_t,Index_t>().swap(m_simVertexBarcodeToIndex);

  std::unordered_multimap<Index_t,Index_t>().swap(m_detIdToCluster);
  std::unordered_multimap<Barcode_t,Index_t>().swap(m_simHitBarcodeToIndex);
  std::unordered_multimap<Barcode_t,Barcode_t>().swap(m_simVertexBarcodeToSimTrackBarcode);
  std::unordered_multimap<Barcode_t,Index_t>().swap(m_simTrackToSimVertex);
  std::unordered_multimap<Barcode_t,Index_t>().swap(m_simVertexToSimTrackParent); 
  std::vector<Barcode_t>().swap(m_simVertexBarcodes);
  std::unordered_map<Index_t,float>().swap(m_detIdToTotalSimEnergy);
}

template<class T> 
void CaloTruthAccumulator::accumulateEvent( const T& event, 
					    const edm::EventSetup& setup, 
					    const edm::Handle< edm::HepMCProduct >& hepMCproduct) {
  //
  // Get the collections
  //
  
  edm::Handle< std::vector<reco::GenParticle> > hGenParticles;
  edm::Handle< std::vector<int> > hGenParticleIndices;
  
  event.getByLabel( simTrackLabel_, hSimTracks );
  event.getByLabel( simVertexLabel_, hSimVertices );
  
  try {
    event.getByLabel( genParticleLabel_, hGenParticles );
    event.getByLabel( genParticleLabel_, hGenParticleIndices );
  } catch( cms::Exception& exception ) {
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
  
  // Clear maps from previous event fill them for this one
  m_simHitBarcodeToIndex.clear();
  m_simTracksConsideredForSimClusters.clear();
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
  
  // now we build the lists of gen particles and descendent sim clusters.
  //	bool useGenParticles_ = false;
  double minEnergy_ = 0.25;
  double maxPseudoRapidity_ = 5.;
  
  std::vector<Index_t> tracksToBecomeClustersInitial;
  std::vector<Barcode_t> descendantTracks;
  SimClusterCollection simClustersForGenParts;
  std::vector<std::unique_ptr<SimHitInfoPerSimTrack_t> > hitInfoList;
  std::vector<std::vector<uint32_t> > simClusterPrimitives;
  std::unordered_multimap<Index_t,Index_t> genPartsToSimClusters;
  const auto& simTracks = *hSimTracks;
  // loop over 
  for (unsigned int i = 0 ; i < simTracks.size() ; ++i) {
    if ( simTracks[i].momentum().E() < minEnergy_ || std::abs(simTracks[i].momentum().Eta()) >= maxPseudoRapidity_ ) continue;
    if ( simTracks[i].noGenpart() ) continue;
    auto temp = CaloTruthAccumulator::descendantSimClusters( simTracks[i].trackId(),simHitPointers );
    if( temp.size() ) {
      output_.pCaloParticles->emplace_back(simTracks[i]);
      m_caloParticles.sc_start_.push_back(output_.pSimClusters->size());
      auto mbegin = std::make_move_iterator(temp.begin());
      auto mend = std::make_move_iterator(temp.end());
      output_.pSimClusters->insert(output_.pSimClusters->end(), mbegin, mend);
      m_caloParticles.sc_stop_.push_back(output_.pSimClusters->size());
    }    
  }  
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
  const auto& simTracks = *hSimTracks;
  if ( CaloTruthAccumulator::consideredBarcode( barcode ) ) {
    LogDebug("CaloParticles") << "SCZ DEBUG Ignoring descendantSimClusters call because this particle is already marked used: " << barcode << std::endl;
    
  }
  std::unique_ptr<SimHitInfoPerSimTrack_t> hit_info = std::move(CaloTruthAccumulator::attachedSimHitInfo(barcode,hits, true, false, false));
  std::unique_ptr<SimHitInfoPerSimTrack_t> inclusive_hit_info = std::move(CaloTruthAccumulator::allAttachedSimHitInfo(barcode,hits, false) );
  if (hit_info->size() > 0) {
    // define the sim cluster starting from the earliest particle that has hits in the calorimeter
    // grab everything that descends from it
    std::unique_ptr<SimHitInfoPerSimTrack_t> marked_hit_info = 
      std::move( CaloTruthAccumulator::allAttachedSimHitInfo(barcode,hits,true) );
    
    const auto& simTrack = simTracks[m_simTrackBarcodeToIndex[barcode]];
    
    result.emplace_back(simTrack);
    auto& simcluster = result.back();

    std::unordered_map<uint32_t,float> acc_frac;
    
    for( const auto& hit_and_fraction : *marked_hit_info ) {
      const uint32_t id = hit_and_fraction.first.rawId();
      if( acc_frac.count(id) ) acc_frac[id] += hit_and_fraction.second;
      else acc_frac[id] = hit_and_fraction.second;
    }

    for( const auto& hit_and_fraction : acc_frac ) {
      simcluster.addRecHitAndFraction(hit_and_fraction.first,hit_and_fraction.second);
    }
  } else {
    if (m_simTrackToSimVertex.count(barcode)) {
      auto vertex_range = m_simTrackToSimVertex.equal_range(barcode);
      for ( auto vertex_iter = vertex_range.first ; vertex_iter != vertex_range.second ; vertex_iter++ ) {
	Index_t decayVertexIndex = vertex_iter->second;
	Barcode_t decayVertexBarcode = m_simVertexBarcodes[decayVertexIndex];
	auto track_range = m_simVertexBarcodeToSimTrackBarcode.equal_range( decayVertexBarcode );
	for ( auto track_iter = track_range.first ; track_iter != track_range.second ; track_iter++ ) {
	  auto daughter_result = CaloTruthAccumulator::descendantSimClusters(track_iter->second,hits);
	  result.insert(result.end(),daughter_result.begin(),daughter_result.end());
	}
      }
    }
  }
  return result;
}  

std::unique_ptr<SimHitInfoPerSimTrack_t> CaloTruthAccumulator::attachedSimHitInfo( Barcode_t barcode , const std::vector<const PCaloHit*>& hits, 
										   bool includeOwn , bool includeOther, bool markUsed ) {
  std::unique_ptr<SimHitInfoPerSimTrack_t> result(new SimHitInfoPerSimTrack_t);
  
  if ( markUsed ) {
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
      ++n;
    }    
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
	  std::unique_ptr<SimHitInfoPerSimTrack_t> daughter_result = std::move(CaloTruthAccumulator::allAttachedSimHitInfo(track_iter->second,hits,markUsed));
	  result->insert(result->end(),daughter_result->begin(),daughter_result->end());
	}
      }
    }
  }  
  return result;
}

std::unique_ptr<SimHitInfoPerSimTrack_t> 
CaloTruthAccumulator::descendantOnlySimHitInfo( Barcode_t barcode, 
						const std::vector<const PCaloHit*>& hits, 
						bool markUsed) {
  return CaloTruthAccumulator::attachedSimHitInfo(barcode,hits,false,true,markUsed);
}

std::unique_ptr<SimHitInfoPerSimTrack_t> 
CaloTruthAccumulator::allAttachedSimHitInfo( Barcode_t barcode, 
					     const std::vector<const PCaloHit*>& hits, 
					     bool markUsed ) {
  return CaloTruthAccumulator::attachedSimHitInfo(barcode,hits,true,true,markUsed);
}

template<class T> void CaloTruthAccumulator::fillSimHits( std::vector<const PCaloHit*>& returnValue, const T& event, const edm::EventSetup& setup ) {
  // loop over the collections
  for( const auto& collectionTag : collectionTags_ ) {
    edm::Handle< std::vector<PCaloHit> > hSimHits;
    event.getByLabel( collectionTag, hSimHits );
    for( const auto& simHit : *hSimHits ) {
      const uint32_t simId = simHit.id();
      int subdet, layer, cell, sec, subsec, zp;
      HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell); 
      DetId id = HGCalDetId((ForwardSubdetector)subdet,zp,layer,subsec,sec,cell);
      uint32_t detId = id.rawId();
      returnValue.push_back( &simHit );
      if( m_detIdToTotalSimEnergy.count(detId) ) m_detIdToTotalSimEnergy[detId] += simHit.energy();
      else m_detIdToTotalSimEnergy[detId] = simHit.energy();
    }
  } // end of loop over InputTags
}

// Register with the framework
DEFINE_DIGI_ACCUMULATOR (CaloTruthAccumulator);
