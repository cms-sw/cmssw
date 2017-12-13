#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/CaloAnalysis/plugins/CaloTruthAccumulatorWithGraph.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

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
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include <iterator>
#include <numeric> // for std::accumulate

#define DEBUG std::cout << __FILE__ << ":" << __LINE__ << "\t"

/* Graph utility functions */

using namespace boost;
namespace {
  template < typename Edge, typename Graph, typename Visitor>
    void print_edge(Edge &e, const Graph & g, Visitor * v) {
      auto const edge_property = get(edge_weight, g, e);
      v->total_simHits += edge_property.simHits;
      std::cout << "Examining edges " << e
        << " --> particle " << edge_property.simTrack->type()
        << "(" << edge_property.simTrack->trackId() << ")"
        << " with SimClusters: " << edge_property.simHits
        << " and total Energy: " << edge_property.energy
        << " Accumulated SimClusters: " << v->total_simHits << std::endl;
    }
  template < typename Vertex, typename Graph >
    void print_vertex(Vertex &u, const Graph & g) {
      auto const vertex_property = get(vertex_name, g, u);
      std::cout << "At " << u;
      // The Mother of all vertices has **no** SimTrack associated.
      if (vertex_property.simTrack)
        std::cout << "[" << vertex_property.simTrack->type() << "]"
                  << "(" << vertex_property.simTrack->trackId() << ")";
      std::cout << std::endl;
    }
  class Custom_dfs_visitor : public boost::default_dfs_visitor {
    public:
      int total_simHits = 0;
      template < typename Edge, typename Graph >
        void examine_edge(Edge e, const Graph& g) {
          print_edge(e, g, this);
        }
      template < typename Edge, typename Graph >
        void finish_edge(Edge e, const Graph & g) {
          auto const edge_property = get(edge_weight, g, e);
          auto src = source(e, g);
          auto trg = target(e, g);
          auto cumulative = edge_property.simHits
                          + get(vertex_name, g, trg).cumulative_simHits
                          + (get(vertex_name, g, src).simTrack ? get(vertex_name, g, src).cumulative_simHits : 0); // when we hit the root vertex we have to stop adding back its contribution.
          auto const src_vertex_property = get(vertex_name, g, src);
          put(get(vertex_name, const_cast<Graph&>(g)),
              src,
              VertexProperty(src_vertex_property.simTrack, cumulative));
          put(get(edge_weight, const_cast<Graph&>(g)), e,
              EdgeProperty(edge_property.simTrack,
                           edge_property.simHits,
                           cumulative,
                           edge_property.energy));
          std::cout << "Finished edge: " << e
                    << " Track id: " << get(edge_weight, g, e).simTrack->trackId()
                    << " has cumulated " << cumulative
                    << " hits" << std::endl;
          std::cout << " SrcVtx: " << src << "\t"
                    << get(vertex_name, g, src).simTrack << "\t"
                    << get(vertex_name, g, src).cumulative_simHits << std::endl;
          std::cout << " TrgVtx: " << trg << "\t"
                    << get(vertex_name, g, trg).simTrack << "\t"
                    << get(vertex_name, g, trg).cumulative_simHits << std::endl;
      }
  };
  class CaloParticle_dfs_visitor : public boost::default_dfs_visitor {
    public:
      int total_simHits = 0;
      CaloParticle_dfs_visitor(CaloTruthAccumulatorWithGraph::OutputCollections & output,
          std::unordered_multimap<Barcode_t,Index_t> & simHitBarcodeToIndex,
          std::map<int, std::map<int, float> > & simTrackDetIdEnergyMap)
        : output_(output),
        simHitBarcodeToIndex_(simHitBarcodeToIndex),
        simTrackDetIdEnergyMap_(simTrackDetIdEnergyMap){}
      template < typename Vertex, typename Graph >
        void discover_vertex(Vertex u, const Graph & g) {
          print_vertex(u, g);
          auto const vertex_property = get(vertex_name, g, u);
          if (!vertex_property.simTrack)
            return;
          auto trackIdx = vertex_property.simTrack->trackId();
          std::cout << "Found " << simHitBarcodeToIndex_.count(trackIdx)
            << " associated simHits" << std::endl;
          if (simHitBarcodeToIndex_.count(trackIdx)) {
            output_.pSimClusters->emplace_back(*vertex_property.simTrack);
            auto& simcluster = output_.pSimClusters->back();
            std::unordered_map<uint32_t,float> acc_energy;
            for( const auto& hit_and_energy : simTrackDetIdEnergyMap_[trackIdx]) {
              const uint32_t id = hit_and_energy.first;
              if( acc_energy.count(id) ) acc_energy[id] += hit_and_energy.second;
              else acc_energy[id] = hit_and_energy.second;
            }

            for( const auto& hit_and_energy : acc_energy ) {
              simcluster.addRecHitAndFraction(hit_and_energy.first,hit_and_energy.second);
            }
          }
        }
      template < typename Edge, typename Graph >
        void examine_edge(Edge e, const Graph& g) {
          print_edge(e, g, this);
        }
    private:
      CaloTruthAccumulatorWithGraph::OutputCollections & output_;
      std::unordered_multimap<Barcode_t,Index_t> & simHitBarcodeToIndex_;
      std::map<int, std::map<int, float> > & simTrackDetIdEnergyMap_;
  };
}

CaloTruthAccumulatorWithGraph::CaloTruthAccumulatorWithGraph( const edm::ParameterSet & config, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  messageCategory_("CaloTruthAccumulatorWithGraph"),
  maximumPreviousBunchCrossing_( config.getParameter<unsigned int>("maximumPreviousBunchCrossing") ),
  maximumSubsequentBunchCrossing_( config.getParameter<unsigned int>("maximumSubsequentBunchCrossing") ),
  simTrackLabel_( config.getParameter<edm::InputTag>("simTrackCollection") ),
  simVertexLabel_( config.getParameter<edm::InputTag>("simVertexCollection") ),
  collectionTags_( ),
  genParticleLabel_( config.getParameter<edm::InputTag>("genParticleCollection") ),
  hepMCproductLabel_( config.getParameter<edm::InputTag>("HepMCProductLabel") ),
  minEnergy_( config.getParameter<double>("MinEnergy") ),
  maxPseudoRapidity_( config.getParameter<double>("MaxPseudoRapidity") )
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

void CaloTruthAccumulatorWithGraph::beginLuminosityBlock( edm::LuminosityBlock const& iLumiBlock, const edm::EventSetup& iSetup ) {
  edm::ESHandle<CaloGeometry> geom;
  iSetup.get<CaloGeometryRecord>().get(geom);
  const HGCalGeometry *eegeom, *fhgeom;
  const HcalGeometry *bhgeom;

  eegeom = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward,HGCEE));
  fhgeom = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward,HGCHEF));
  bhgeom = dynamic_cast<const HcalGeometry*>(geom->getSubdetectorGeometry(DetId::Hcal,HcalEndcap));

  hgtopo_[0] = &(eegeom->topology());
  hgtopo_[1] = &(fhgeom->topology());

  for( unsigned i = 0; i < 2; ++i ) {
    hgddd_[i] = &(hgtopo_[i]->dddConstants());
  }

  hcddd_    = bhgeom->topology().dddConstants();

  caloStartZ = hgddd_[0]->waferZ(1,false)*10.0; // get the location of the first plane of silicon, put in mm
}

void CaloTruthAccumulatorWithGraph::initializeEvent( edm::Event const& event, edm::EventSetup const& setup ) {
  output_.pSimClusters.reset( new SimClusterCollection() );
  output_.pCaloParticles.reset( new CaloParticleCollection() );

  m_detIdToCluster.clear();
  m_detIdToTotalSimEnergy.clear();
}

/// create handle to edm::HepMCProduct here because event.getByLabel with edm::HepMCProduct only works for edm::Event
/// but not for PileUpEventPrincipal; PileUpEventPrincipal::getByLabel tries to call T::value_type and T::iterator
/// (where T is the type of the object one wants to get a handle to) which is only implemented for container-like objects
/// like std::vector but not for edm::HepMCProduct!

void CaloTruthAccumulatorWithGraph::accumulate( edm::Event const& event, edm::EventSetup const& setup ) {
  // Call the templated version that does the same for both signal and pileup events

  edm::Handle< edm::HepMCProduct > hepmc;
  event.getByLabel(hepMCproductLabel_, hepmc);

  edm::LogInfo(messageCategory_) << " CaloTruthAccumulatorWithGraph::accumulate (signal)";
  accumulateEvent( event, setup, hepmc );
}

void CaloTruthAccumulatorWithGraph::accumulate( PileUpEventPrincipal const& event, edm::EventSetup const& setup, edm::StreamID const& ) {
  // If this bunch crossing is outside the user configured limit, don't do anything.
  if( event.bunchCrossing()>=-static_cast<int>(maximumPreviousBunchCrossing_) && event.bunchCrossing()<=static_cast<int>(maximumSubsequentBunchCrossing_) )
    {
      //edm::LogInfo(messageCategory_) << "Analysing pileup event for bunch crossing " << event.bunchCrossing();

      //simply create empty handle as we do not have a HepMCProduct in PU anyway
      edm::Handle< edm::HepMCProduct > hepmc;
      edm::LogInfo(messageCategory_) << " CaloTruthAccumulatorWithGraph::accumulate (pileup) bunchCrossing=" << event.bunchCrossing();
      accumulateEvent( event, setup, hepmc );
    }
  else edm::LogInfo(messageCategory_) << "Skipping pileup event for bunch crossing " << event.bunchCrossing();
}

void CaloTruthAccumulatorWithGraph::finalizeEvent( edm::Event& event, edm::EventSetup const& setup ) {
  edm::LogInfo("CaloTruthAccumulatorWithGraph") << "Adding " << output_.pSimClusters->size()
				       << " SimParticles and " << output_.pCaloParticles->size()
				       << " CaloParticles to the event.";

  // now we need to normalize the hits and energies into hits and fractions
  // (since we have looped over all pileup events)

  for( auto& sc : *(output_.pSimClusters) ) {
    auto hitsAndEnergies = sc.hits_and_fractions();
    sc.clearHitsAndFractions();
    for( auto& hAndE : hitsAndEnergies ) {
      const float totalenergy = m_detIdToTotalSimEnergy[hAndE.first];
      float fraction = 0.;
      if(totalenergy>0) fraction = hAndE.second/totalenergy;
      else edm::LogWarning("CaloTruthAccumulatorWithGraph") << "TotalSimEnergy for hit " << hAndE.first << " is 0! The fraction for this hit cannot be computed.";
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
  std::unordered_map<Barcode_t,Barcode_t>().swap(m_simTrackBarcodeToSimVertexParentBarcode);
  std::unordered_multimap<Barcode_t,Index_t>().swap(m_simTrackToSimVertex);
  std::unordered_multimap<Barcode_t,Index_t>().swap(m_simVertexToSimTrackParent);
  std::vector<Barcode_t>().swap(m_simVertexBarcodes);
  std::unordered_map<Index_t,float>().swap(m_detIdToTotalSimEnergy);
}

template<class T>
void CaloTruthAccumulatorWithGraph::accumulateEvent( const T& event,
					    const edm::EventSetup& setup,
					    const edm::Handle< edm::HepMCProduct >& hepMCproduct) {
  //
  // Get the collections
  //

  edm::Handle< std::vector<reco::GenParticle> > hGenParticles;
  edm::Handle< std::vector<int> > hGenParticleIndices;

  event.getByLabel( simTrackLabel_, hSimTracks );
  event.getByLabel( simVertexLabel_, hSimVertices );

  event.getByLabel( genParticleLabel_, hGenParticles );
  event.getByLabel( genParticleLabel_, hGenParticleIndices );

  std::vector<std::pair<DetId,const PCaloHit*> > simHitPointers;
  std::map<int, std::map<int, float> > simTrackDetIdEnergyMap;
  fillSimHits( simHitPointers, simTrackDetIdEnergyMap, event, setup );

  // Clear maps from previous event fill them for this one
  m_simHitBarcodeToIndex.clear();
  m_simTracksConsideredForSimClusters.clear();
  for (unsigned int i = 0 ; i < simHitPointers.size(); ++i) {
    m_simHitBarcodeToIndex.emplace(simHitPointers[i].second->geantTrackId(),i);
  }
  m_genParticleBarcodeToIndex.clear();
  if( hGenParticles.isValid() && hGenParticleIndices.isValid() ) {
    for (unsigned int i = 0 ; i < hGenParticles->size() ; ++i) {
      m_genParticleBarcodeToIndex.emplace(hGenParticleIndices->at(i),i);
    }
  }
  m_genBarcodeToSimTrackIndex.clear();
  m_simVertexBarcodeToSimTrackBarcode.clear();
  m_simTrackBarcodeToSimVertexParentBarcode.clear();
  m_simTrackBarcodeToIndex.clear();
  for (unsigned int i = 0 ; i < hSimTracks->size() ; i++) {
    if( !hSimTracks->at(i).noGenpart() ) {
      m_genBarcodeToSimTrackIndex.emplace(hSimTracks->at(i).genpartIndex(), i);
    }
    if( !hSimTracks->at(i).noVertex() ) {
      m_simVertexBarcodeToSimTrackBarcode.emplace(hSimTracks->at(i).vertIndex(), hSimTracks->at(i).trackId());
      m_simTrackBarcodeToSimVertexParentBarcode.emplace(hSimTracks->at(i).trackId(), hSimTracks->at(i).vertIndex());
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

  const auto& tracks = *hSimTracks;
  const auto& vertices = *hSimVertices;
  std::map<int, int> trackid_to_track_index;
  DecayChain decay;
  int idx = 0;
  // Build the main decay graph and assign the SimTrack to each edge. The graph
  // built here will only contain the particles that have a decay vertex
  // associated to them. In order to recover also the particles that will not
  // decay, we need to keep track of the SimTrack used here and add,
  // a-posteriori, the ones not used, associating a ghost vertex (starting from
  // the highest simulated vertex number +1), in order to build the edge and
  // identify them immediately as stable (i.e. not decayed).
  std::cout << "TRACKS" << std::endl;
  for (auto const & t : tracks) {
    std::cout << idx << "\t" << t.trackId() << "\t" << t << std::endl;
    trackid_to_track_index[t.trackId()] = idx;
    idx++;
  }
  idx = 0;
  std::vector<bool> used_sim_tracks(tracks.size(), false);
  std::cout << "VERTICES" << std::endl;
  for (auto const & v: vertices) {
    std::cout << idx++ << "\t" << v << std::endl;
    if (v.parentIndex() != -1) {
      add_edge(tracks.at(trackid_to_track_index[v.parentIndex()]).vertIndex(),
          v.vertexId(),
          EdgeProperty(&tracks.at(trackid_to_track_index[v.parentIndex()]),
            simTrackDetIdEnergyMap[trackid_to_track_index[v.parentIndex()]].size(),
            0,
            std::accumulate(simTrackDetIdEnergyMap[trackid_to_track_index[v.parentIndex()]].begin(),
              simTrackDetIdEnergyMap[trackid_to_track_index[v.parentIndex()]].end(), 0.,
              [&](float partial, std::pair<int, float> current) {
              return partial + current.second/m_detIdToTotalSimEnergy[current.first]; })),
          decay);
      used_sim_tracks[trackid_to_track_index[v.parentIndex()]] = true;
    }
  }
  // Assign the motherParticle property to each vertex
  auto const & vertexMothersProp = get(vertex_name, decay);
  // Now recover the particles that did not decay. Append them with an index
  // bigger than the size of the generated vertices.
  int offset = vertices.size() + 1;
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (!used_sim_tracks[i]) {
      add_edge(tracks.at(i).vertIndex(), offset,
               EdgeProperty(&tracks.at(i),
                 simTrackDetIdEnergyMap[tracks.at(i).trackId()].size(),
                 0,
                 std::accumulate(simTrackDetIdEnergyMap[tracks.at(i).trackId()].begin(),
                                 simTrackDetIdEnergyMap[tracks.at(i).trackId()].end(), 0.,
                                 [&](float partial, std::pair<int, float> current) {
                                    return partial + current.second/m_detIdToTotalSimEnergy[current.first];
                                 })), decay);
      // The properties for "fake" vertices associated to stable particles have
      // to be set inside this loop, since they do not belong to the vertices
      // collection and would be skipped by that loop (coming next)
      put(vertexMothersProp, offset, VertexProperty(&tracks.at(i), 0));
      offset++;
    }
  }
  for (auto const & v: vertices) {
    if (v.parentIndex() != -1) {
      put(vertexMothersProp, v.vertexId(), VertexProperty(&tracks.at(trackid_to_track_index[v.parentIndex()]), 0));
    }
  }
#define DFS
#ifdef  DFS
  Custom_dfs_visitor vis;
  depth_first_search(decay, visitor(vis));
  auto const first_generation = out_edges(0, decay);
  for (auto edge = first_generation.first; edge != first_generation.second; ++edge) {
    auto const edge_property = get(edge_weight, decay, *edge);
    // Apply selection on SimTracks in order to promote them to be CaloParticles.
    if (edge_property.cumulative_simHits == 0
        or edge_property.simTrack->noGenpart()
        or edge_property.simTrack->momentum().E() < minEnergy_
        or std::abs(edge_property.simTrack->momentum().Eta()) >= maxPseudoRapidity_)
      continue;
    output_.pCaloParticles->emplace_back(*(edge_property.simTrack));
    m_caloParticles.sc_start_.push_back(output_.pSimClusters->size());
    CaloParticle_dfs_visitor caloParticleCreator(output_,
                                                 m_simHitBarcodeToIndex,
                                                 simTrackDetIdEnergyMap);
    depth_first_search(decay, visitor(caloParticleCreator).root_vertex(target(*edge, decay)));
    m_caloParticles.sc_stop_.push_back(output_.pSimClusters->size());
    std::cout << "Creating CaloParticle particle: "
              << edge_property.simTrack->type()
              << "(" << edge_property.simTrack->trackId() << ")"
              << " with total SimClusters: " << edge_property.cumulative_simHits
              << " and total Energy: " << edge_property.energy
              << std::endl;
  }
#else
  Custom_bfs_visitor vis;
  breadth_first_search(decay, 0, visitor(vis));
#endif

  const auto& simTracks = *hSimTracks;
  // loop over
  for (unsigned int i = 0 ; i < simTracks.size() ; ++i) {
    if ( simTracks[i].momentum().E() < minEnergy_ || std::abs(simTracks[i].momentum().Eta()) >= maxPseudoRapidity_ ) continue;
    if ( simTracks[i].noGenpart() ) continue;
    DEBUG << "INIT Analysing simTrack: " << i << "\t" << simTracks[i] << std::endl;
    auto temp = CaloTruthAccumulatorWithGraph::descendantSimClusters( simTracks[i].trackId(),simHitPointers );
    std::cout << "Return size from descendantSimClusters for track: " << i << " is " << temp.size() << std::endl;
    for (auto const & v : temp) {
      std::cout << "Adding a simTrack: " << v << std::endl;
    }
    if( !temp.empty() ) {
      output_.pCaloParticles->emplace_back(simTracks[i]);
      m_caloParticles.sc_start_.push_back(output_.pSimClusters->size());
      auto mbegin = std::make_move_iterator(temp.begin());
      auto mend = std::make_move_iterator(temp.end());
      output_.pSimClusters->insert(output_.pSimClusters->end(), mbegin, mend);
      m_caloParticles.sc_stop_.push_back(output_.pSimClusters->size());
    }
  }
}


SimClusterCollection CaloTruthAccumulatorWithGraph::descendantSimClusters( Barcode_t barcode, const std::vector<std::pair<DetId,const PCaloHit*> >& hits ) {
  DEBUG << __FUNCTION__ << std::endl;
  SimClusterCollection result;
  const auto& simTracks = *hSimTracks;
  if ( CaloTruthAccumulatorWithGraph::consideredBarcode( barcode ) ) {
    DEBUG << "SCZ DEBUG Ignoring descendantSimClusters call because this particle is already marked used: " << barcode << std::endl;
    //return result;
  }

  std::unique_ptr<SimHitInfoPerSimTrack_t> hit_info = std::move(CaloTruthAccumulatorWithGraph::attachedSimHitInfo(barcode,hits, true, false, false));
  //std::unique_ptr<SimHitInfoPerSimTrack_t> inclusive_hit_info = std::move(CaloTruthAccumulatorWithGraph::allAttachedSimHitInfo(barcode,hits, false) );

  const auto& simTrack = simTracks[m_simTrackBarcodeToIndex[barcode]];
  Barcode_t vtxBarcode = m_simTrackBarcodeToSimVertexParentBarcode[barcode];
  const auto& vtx = hSimVertices->at(m_simVertexBarcodeToIndex[vtxBarcode]);
  const bool isInCalo = (std::abs(vtx.position().z()) > caloStartZ*0.1 - 30.0); // add a buffer region in front of the calo face
  DEBUG << "Vertex Z-position: " << std::abs(vtx.position().z())
        << " caloStartZ[cm]: " << caloStartZ*0.1
        << " isInCalo: " << isInCalo << std::endl;

  if (!hit_info->empty()) {
    // define the sim cluster starting from the earliest particle that has hits in the calorimeter
    // grab everything that descends from it
    std::unique_ptr<SimHitInfoPerSimTrack_t> marked_hit_info;


    if( isInCalo  ) {
      marked_hit_info = std::move( CaloTruthAccumulatorWithGraph::allAttachedSimHitInfo(barcode,hits,true) );
    } else {
      marked_hit_info = std::move( CaloTruthAccumulatorWithGraph::attachedSimHitInfo(barcode,hits,true,false,true) );
    }

    if( !marked_hit_info->empty() ) {
      result.emplace_back(simTrack);
      auto& simcluster = result.back();

      std::unordered_map<uint32_t,float> acc_energy;

      for( const auto& hit_and_energy : *marked_hit_info ) {
	const uint32_t id = hit_and_energy.first.rawId();
	if( acc_energy.count(id) ) acc_energy[id] += hit_and_energy.second;
	else acc_energy[id] = hit_and_energy.second;
      }

      for( const auto& hit_and_energy : acc_energy ) {
        DEBUG << "Barcode: " << barcode
              << " adding DetId: " << hit_and_energy.first
              << "\twith energy: " << hit_and_energy.second << std::endl;
	simcluster.addRecHitAndFraction(hit_and_energy.first,hit_and_energy.second);
      }
    }
  }

  if ( m_simTrackToSimVertex.count(barcode) ) {
    auto vertex_range = m_simTrackToSimVertex.equal_range(barcode);
    for ( auto vertex_iter = vertex_range.first ; vertex_iter != vertex_range.second ; vertex_iter++ ) {
      Index_t decayVertexIndex = vertex_iter->second;
      Barcode_t decayVertexBarcode = m_simVertexBarcodes[decayVertexIndex];
      auto track_range = m_simVertexBarcodeToSimTrackBarcode.equal_range( decayVertexBarcode );
      for ( auto track_iter = track_range.first ; track_iter != track_range.second ; track_iter++ ) {

	auto daughter_result = CaloTruthAccumulatorWithGraph::descendantSimClusters(track_iter->second,hits);
	result.insert(result.end(),daughter_result.begin(),daughter_result.end());
      }
    }
  }

  return result;
}

std::unique_ptr<SimHitInfoPerSimTrack_t> CaloTruthAccumulatorWithGraph::attachedSimHitInfo( Barcode_t barcode , const std::vector<std::pair<DetId,const PCaloHit*> >& hits,
										   bool includeOwn , bool includeOther, bool markUsed ) {
  DEBUG << __FUNCTION__ << std::endl;
  const auto& simTracks = *hSimTracks;
  std::unique_ptr<SimHitInfoPerSimTrack_t> result(new SimHitInfoPerSimTrack_t);

  const auto& simTrack = simTracks[m_simTrackBarcodeToIndex[barcode]];
  DEBUG << "Track with barcode: " << barcode
        << " is used: " << markUsed
        << " includeOwn: " << includeOwn
        << " includeOther: " << includeOther
        << std::endl;

  if ( markUsed ) {
    if ( CaloTruthAccumulatorWithGraph::consideredBarcode( barcode ) ) {
      return result;
    }
    CaloTruthAccumulatorWithGraph::setConsideredBarcode( barcode );
  }
  if (includeOwn) {
    DEBUG << "Adding self hits for track with barcode: " << barcode << std::endl;
    auto range = m_simHitBarcodeToIndex.equal_range( barcode );
    unsigned n = 0;
    for ( auto iter = range.first ; iter != range.second ; iter++ ) {
      const auto& the_hit = hits[iter->second];
      DEBUG << "Adding self " << the_hit.first.rawId()
            << " for track with barcode " << barcode
            << " with energy: " << the_hit.second->energy() << std::endl;
      result->emplace_back(the_hit.first,the_hit.second->energy());
      ++n;
    }
  }

  // need to sim to the next sim track if we explicitly ask or
  // if we are in the calorimeter next (no interaction)
  // or if this is a continuation of the same particle
  DEBUG << "Track with barcode: " << barcode << " has decayed: " << m_simTrackToSimVertex.count(barcode)
        << std::endl;
  if (m_simTrackToSimVertex.count(barcode)) {
    auto vertex_range = m_simTrackToSimVertex.equal_range(barcode);
    for ( auto vertex_iter = vertex_range.first ; vertex_iter != vertex_range.second ; vertex_iter++ ) {
      Index_t decayVertexIndex = vertex_iter->second;
      DEBUG << "Considering vertex number: " << decayVertexIndex << std::endl;
      const auto& nextVtx = (*hSimVertices)[decayVertexIndex];
      const bool nextInCalo = (std::abs(nextVtx.position().z()) > caloStartZ*0.1 - 30.0);  // add a buffer region in front of the calo face
      DEBUG << "The vertex is nextInCalo: " << nextInCalo << std::endl;

      Barcode_t decayVertexBarcode = m_simVertexBarcodes[decayVertexIndex];
	auto track_range = m_simVertexBarcodeToSimTrackBarcode.equal_range( decayVertexBarcode );
	for ( auto track_iter = track_range.first ; track_iter != track_range.second ; track_iter++ ) {
	  if( !barcodeLogicWarningAlready_ && track_iter->second < barcode ) {
	    barcodeLogicWarningAlready_ = true;
	    DEBUG << " Daughter particle has a lower barcode than parent. This may screw up the logic!" << std::endl;
	  }
          DEBUG << "Considering daugther with barcode: " << track_iter->second << std::endl;
	  const auto& daughter = simTracks[m_simTrackBarcodeToIndex[track_iter->second]];

	  if( includeOther || nextInCalo ) {
	    std::unique_ptr<SimHitInfoPerSimTrack_t> daughter_result = std::move(CaloTruthAccumulatorWithGraph::allAttachedSimHitInfo(track_iter->second,hits,markUsed));
	    result->insert(result->end(),daughter_result->begin(),daughter_result->end());
	  } else if ( daughter.type() == simTrack.type() ) {
	    std::unique_ptr<SimHitInfoPerSimTrack_t> daughter_result = std::move(CaloTruthAccumulatorWithGraph::attachedSimHitInfo(track_iter->second,hits,includeOwn, includeOther, markUsed));
	     result->insert(result->end(),daughter_result->begin(),daughter_result->end());
	  }
	}
      }
    }
  return result;
}


std::unique_ptr<SimHitInfoPerSimTrack_t>
CaloTruthAccumulatorWithGraph::allAttachedSimHitInfo( Barcode_t barcode,
					     const std::vector<std::pair<DetId,const PCaloHit*> >& hits,
					     bool markUsed ) {
  DEBUG << "allAttachedSimHitInfo for track with barcode: " << barcode << std::endl;
  return CaloTruthAccumulatorWithGraph::attachedSimHitInfo(barcode,hits,true,true,markUsed);
}

template<class T> void CaloTruthAccumulatorWithGraph::fillSimHits(
    std::vector<std::pair<DetId, const PCaloHit*> >& returnValue,
    std::map<int, std::map<int, float> > & simTrackDetIdEnergyMap,
    const T& event,
    const edm::EventSetup& setup ) {
  // loop over the collections
  for( const auto& collectionTag : collectionTags_ ) {
    edm::Handle< std::vector<PCaloHit> > hSimHits;
    const bool isHcal = ( collectionTag.instance().find("HcalHits") != std::string::npos );
    event.getByLabel( collectionTag, hSimHits );
    for( const auto& simHit : *hSimHits ) {
      DetId id(0);
      const uint32_t simId = simHit.id();
      if( isHcal ) {
        HcalDetId hid = HcalHitRelabeller::relabel(simId, hcddd_);
        if(hid.subdet()==HcalEndcap) id = hid;
      } else {
	int subdet, layer, cell, sec, subsec, zp;
	HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);
	const HGCalDDDConstants* ddd = hgddd_[subdet-3];
	std::pair<int,int> recoLayerCell = ddd->simToReco(cell,layer,sec,
							  hgtopo_[subdet-3]->detectorType());
	cell  = recoLayerCell.first;
	layer = recoLayerCell.second;
	// skip simhits with bad barcodes or non-existant layers
	if( layer == -1  || simHit.geantTrackId() == 0 ) continue;
	id = HGCalDetId((ForwardSubdetector)subdet,zp,layer,subsec,sec,cell);
      }

      if( DetId(0) == id ) continue;

      uint32_t detId = id.rawId();
      returnValue.emplace_back(id, &simHit);
      if (simTrackDetIdEnergyMap.count(simHit.geantTrackId())
          && simTrackDetIdEnergyMap[simHit.geantTrackId()].count(id.rawId()))
        simTrackDetIdEnergyMap[simHit.geantTrackId()][id.rawId()] += simHit.energy();
      else
        simTrackDetIdEnergyMap[simHit.geantTrackId()][id.rawId()] = simHit.energy();

      if( m_detIdToTotalSimEnergy.count(detId) ) m_detIdToTotalSimEnergy[detId] += simHit.energy();
      else m_detIdToTotalSimEnergy[detId] = simHit.energy();
    }
  } // end of loop over InputTags
}

// Register with the framework
DEFINE_DIGI_ACCUMULATOR (CaloTruthAccumulatorWithGraph);
