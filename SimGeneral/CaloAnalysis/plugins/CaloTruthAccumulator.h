#ifndef CaloAnalysis_CaloTruthAccumulator_h
#define CaloAnalysis_CaloTruthAccumulator_h

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include <memory> // required for std::auto_ptr
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <unordered_map>
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

typedef unsigned Index_t;
typedef int Barcode_t;
typedef std::pair<Index_t,Index_t> IndexPair_t;
typedef std::pair<IndexPair_t,float> SimHitInfo_t;
typedef std::pair<Barcode_t,Index_t> BarcodeIndexPair_t;
typedef std::pair<Barcode_t,Barcode_t> BarcodePair_t;
typedef std::pair<DetId,float> SimHitInfoPerRecoDetId_t;
typedef std::vector<SimHitInfoPerRecoDetId_t> SimHitInfoPerSimTrack_t;


// typedef uint32_t RecoDetId_t;

// Forward declarations
namespace edm {
  class ParameterSet;
  class ConsumesCollector;
  namespace stream {
    class EDProducerBase;
  }
  class Event;
  class EventSetup;
  class StreamID;
}
class PileUpEventPrincipal;
class PCaloHit;
class SimTrack;
class SimVertex;

class CaloTruthAccumulator : public DigiAccumulatorMixMod {
 public:
  explicit CaloTruthAccumulator( const edm::ParameterSet& config, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC);
 private:
  virtual void initializeEvent( const edm::Event& event, const edm::EventSetup& setup ) override;
  virtual void accumulate( const edm::Event& event, const edm::EventSetup& setup ) override;
  virtual void accumulate( const PileUpEventPrincipal& event, const edm::EventSetup& setup, edm::StreamID const& ) override;
  virtual void finalizeEvent( edm::Event& event, const edm::EventSetup& setup ) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;
  
  /** @brief Both forms of accumulate() delegate to this templated method. */
  template<class T> void accumulateEvent( const T& event, const edm::EventSetup& setup, const edm::Handle< edm::HepMCProduct >& hepMCproduct );
  
  /** @brief Fills the supplied vector with pointers to the SimHits, checking for bad modules if required */
  template<class T> void fillSimHits( std::vector<std::pair<DetId,const PCaloHit*> > & returnValue, const T& event, const edm::EventSetup& setup );
  
  std::vector<Barcode_t> descendantTrackBarcodes( Barcode_t barcode );
  std::unique_ptr<SimHitInfoPerSimTrack_t> attachedSimHitInfo( Barcode_t st, const std::vector<std::pair<DetId,const PCaloHit*> > & hits, 
							       bool includeOwn = true, bool includeOther = false, bool markUsed = false);
  std::unique_ptr<SimHitInfoPerSimTrack_t> descendantOnlySimHitInfo( Barcode_t st, const std::vector<std::pair<DetId,const PCaloHit*> > & hits, bool markUsed = false);
  std::unique_ptr<SimHitInfoPerSimTrack_t> allAttachedSimHitInfo( Barcode_t st, const std::vector<std::pair<DetId,const PCaloHit*> > & hits, bool markUsed = false);
  
  SimClusterCollection descendantSimClusters( Barcode_t barcode, const std::vector<std::pair<DetId,const PCaloHit*> > & hits );
  std::set<Barcode_t> m_simTracksConsideredForSimClusters;
  void setConsideredBarcode( Barcode_t barcode ) { m_simTracksConsideredForSimClusters.insert( barcode ); }
  bool consideredBarcode( Barcode_t barcode ) { 
    //	  return (std::find(m_simTracksConsideredForSimClusters.begin(), m_simTracksConsideredForSimClusters.end(), barcode) != m_simTracksConsideredForSimClusters.end());
    return m_simTracksConsideredForSimClusters.count( barcode );
  }
  
  const std::string messageCategory_; ///< The message category used to send messages to MessageLogger
  
  struct calo_particles {
    std::vector<uint32_t> sc_start_;
    std::vector<uint32_t> sc_stop_;

    void swap(calo_particles& oth) {
      sc_start_.swap(oth.sc_start_);
      sc_stop_.swap(oth.sc_stop_);
    }

    void clear() {
      sc_start_.clear();
      sc_stop_.clear();
    }
  };

  calo_particles m_caloParticles;
  double caloStartZ;

  std::unordered_map<Index_t,float> m_detIdToTotalSimEnergy; // keep track of cell normalizations
  std::unordered_map<Barcode_t,Index_t> m_genParticleBarcodeToIndex;
  std::unordered_map<Barcode_t,Index_t> m_simTrackBarcodeToIndex;
  std::unordered_map<Barcode_t,Index_t> m_genBarcodeToSimTrackIndex;
  std::unordered_map<Barcode_t,Index_t> m_simVertexBarcodeToIndex;
  std::unordered_multimap<Index_t,Index_t> m_detIdToCluster;
  std::unordered_multimap<Barcode_t,Index_t> m_simHitBarcodeToIndex;
  std::unordered_multimap<Barcode_t,Barcode_t> m_simVertexBarcodeToSimTrackBarcode;
  std::unordered_map<Barcode_t,Barcode_t> m_simTrackBarcodeToSimVertexParentBarcode;
  std::unordered_multimap<Barcode_t,Index_t> m_simTrackToSimVertex; 
  std::unordered_multimap<Barcode_t,Index_t> m_simVertexToSimTrackParent; 
  //	std::unordered_multimap<RecoDetId_t,SimHitInfo_t> m_recoDetIdToSimHits;
  
  std::vector<Barcode_t> m_simVertexBarcodes;
  
  //	const double volumeRadius_;
  //	const double volumeZ_;
  /// maximum distance for HepMC::GenVertex to be added to SimVertex
  //	const double vertexDistanceCut_;
  //	const bool ignoreTracksOutsideVolume_;
  
  /** The maximum bunch crossing BEFORE the signal crossing to create TrackinParticles for. Use positive values. If set to zero no
   * previous bunches are added and only in-time, signal and after bunches (defined by maximumSubsequentBunchCrossing_) are used.*/
  const unsigned int maximumPreviousBunchCrossing_;
  /** The maximum bunch crossing AFTER the signal crossing to create TrackinParticles for. E.g. if set to zero only
   * uses the signal and in time pileup (and previous bunches defined by the maximumPreviousBunchCrossing_ parameter). */
  const unsigned int maximumSubsequentBunchCrossing_;
  /// If bremsstrahlung merging, whether to also add the unmerged collection to the event or not.
  //	const bool createUnmergedCollection_;
  //	const bool createMergedCollection_;
  /// Whether or not to create a separate collection for just the initial interaction vertices
  //	const bool createInitialVertexCollection_;
  /// Whether or not to add the full parentage of any TrackingParticle that is inserted in the collection.
  //	const bool addAncestors_;
  
  const edm::InputTag simTrackLabel_;
  const edm::InputTag simVertexLabel_;
  edm::Handle<std::vector<SimTrack> > hSimTracks;
  edm::Handle<std::vector<SimVertex> > hSimVertices;

  std::vector<edm::InputTag> collectionTags_;
  edm::InputTag genParticleLabel_;
  /// Needed to add HepMC::GenVertex to SimVertex
  edm::InputTag hepMCproductLabel_;

  const double minEnergy_, maxPseudoRapidity_;
  
  bool selectorFlag_;
  /// Uses the same config as selector_, but can be used to drop out early since selector_ requires the TrackingParticle to be created first.
  bool chargedOnly_;
  /// Uses the same config as selector_, but can be used to drop out early since selector_ requires the TrackingParticle to be created first.
  bool signalOnly_;
  
  bool barcodeLogicWarningAlready_;
  
  /** @brief When counting hits, allows hits in different detectors to have a different process type.
   *
   * Fast sim PCaloHits seem to have a peculiarity where the process type (as reported by PCaloHit::processType()) is
   * different for the tracker than the muons. When counting how many hits there are, the code usually only counts
   * the number of hits that have the same process type as the first hit. Setting this to true will also count hits
   * that have the same process type as the first hit in the second detector.<br/>
   */
  //	bool allowDifferentProcessTypeForDifferentDetectors_;
 public:
  // These always go hand in hand, and I need to pass them around in the internal
  // functions, so I might as well package them up in a struct.
  struct OutputCollections
  {
    std::unique_ptr<SimClusterCollection> pSimClusters;
    std::unique_ptr<CaloParticleCollection> pCaloParticles;
    //		std::auto_ptr<TrackingVertexCollection> pTrackingVertices;
    //		TrackingParticleRefProd refTrackingParticles;
    //		TrackingVertexRefProd refTrackingVertexes;
  };
 private:
  const HGCalTopology*     hgtopo_[2];
  const HGCalDDDConstants* hgddd_[2];
  const HcalDDDRecConstants* hcddd_;
  OutputCollections output_;  
};

#endif // end of "#ifndef CaloAnalysis_CaloTruthAccumulator_h"
