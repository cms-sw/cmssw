// -*- C++ -*-
//
// Package:    NtupleDump/TrackingNtuple
// Class:      TrackingNtuple
//
/**\class TrackingNtuple TrackingNtuple.cc NtupleDump/TrackingNtuple/plugins/TrackingNtuple.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Tue, 25 Aug 2015 13:22:49 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "Validation/RecoTrack/interface/trackFromSeedFitFailed.h"

#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include "TTree.h"

/*
todo: 
add refitted hit position after track/seed fit
add local angle, path length!
*/

namespace {
  std::string subdetstring(int subdet) {
    switch(subdet) {
    case StripSubdetector::TIB:         return "- TIB";
    case StripSubdetector::TOB:         return "- TOB";
    case StripSubdetector::TEC:         return "- TEC";
    case StripSubdetector::TID:         return "- TID";
    case PixelSubdetector::PixelBarrel: return "- PixBar";
    case PixelSubdetector::PixelEndcap: return "- PixFwd";
    default:                            return "UNKNOWN TRACKER HIT TYPE";
    }
  }

  struct ProductIDSetPrinter {
    ProductIDSetPrinter(const std::set<edm::ProductID>& set): set_(set) {}

    void print(std::ostream& os) const {
      for(const auto& item: set_) {
        os << item << " ";
      }
    }

    const std::set<edm::ProductID>& set_;
  };
  std::ostream& operator<<(std::ostream& os, const ProductIDSetPrinter& o) {
    o.print(os);
    return os;
  }
  template <typename T>
  struct ProductIDMapPrinter {
    ProductIDMapPrinter(const std::map<edm::ProductID, T>& map): map_(map) {}

    void print(std::ostream& os) const {
      for(const auto& item: map_) {
        os << item.first << " ";
      }
    }

    const std::map<edm::ProductID, T>& map_;
  };
  template <typename T>
  auto make_ProductIDMapPrinter(const std::map<edm::ProductID, T>& map) {
    return ProductIDMapPrinter<T>(map);
  }
  template <typename T>
  std::ostream& operator<<(std::ostream& os, const ProductIDMapPrinter<T>& o) {
    o.print(os);
    return os;
  }

  template <typename T>
  struct VectorPrinter {
    VectorPrinter(const std::vector<T>& vec): vec_(vec) {}

    void print(std::ostream& os) const {
      for(const auto& item: vec_) {
        os << item << " ";
      }
    }

    const std::vector<T>& vec_;
  };
  template <typename T>
  auto make_VectorPrinter(const std::vector<T>& vec) {
    return VectorPrinter<T>(vec);
  }
  template <typename T>
  std::ostream& operator<<(std::ostream& os, const VectorPrinter<T>& o) {
    o.print(os);
    return os;
  }

  void checkProductID(const std::set<edm::ProductID>& set, const edm::ProductID& id, const char *name) {
    if(set.find(id) == set.end())
      throw cms::Exception("Configuration") << "Got " << name << " with a hit with ProductID " << id
                                            << " which does not match to the set of ProductID's for the hits: "
                                            << ProductIDSetPrinter(set)
                                            << ". Usually this is caused by a wrong hit collection in the configuration.";
  }

  template <typename SimLink, typename Func>
  void forEachMatchedSimLink(const edm::DetSet<SimLink>& digiSimLinks, uint32_t channel, Func func) {
    for(const auto& link: digiSimLinks) {
      if(link.channel() == channel) {
        func(link);
      }
    }
  }

  std::map<unsigned int, double>  chargeFraction(const SiPixelCluster& cluster, const DetId& detId,
                                                 const edm::DetSetVector<PixelDigiSimLink>& digiSimLink) {
    std::map<unsigned int, double> simTrackIdToAdc;

    auto idetset = digiSimLink.find(detId);
    if(idetset == digiSimLink.end())
      return simTrackIdToAdc;

    double adcSum = 0;
    PixelDigiSimLink found;
    for(int iPix=0; iPix != cluster.size(); ++iPix) {
      const SiPixelCluster::Pixel& pixel = cluster.pixel(iPix);
      adcSum += pixel.adc;
      uint32_t channel = PixelChannelIdentifier::pixelToChannel(pixel.x, pixel.y);
      forEachMatchedSimLink(*idetset, channel, [&](const PixelDigiSimLink& simLink){
          double& adc = simTrackIdToAdc[simLink.SimTrackId()];
          adc += pixel.adc*simLink.fraction();
        });
    }

    for(auto& pair: simTrackIdToAdc) {
      if(adcSum == 0.)
        pair.second = 0.;
      else
        pair.second /= adcSum;
    }

    return simTrackIdToAdc;
  }

  std::map<unsigned int, double> chargeFraction(const SiStripCluster& cluster, const DetId& detId,
                                                const edm::DetSetVector<StripDigiSimLink>& digiSimLink) {
    std::map<unsigned int, double> simTrackIdToAdc;

    auto idetset = digiSimLink.find(detId);
    if(idetset == digiSimLink.end())
      return simTrackIdToAdc;

    double adcSum = 0;
    StripDigiSimLink found;
    int first  = cluster.firstStrip();
    for(size_t i=0; i<cluster.amplitudes().size(); ++i) {
      adcSum += cluster.amplitudes()[i];
      forEachMatchedSimLink(*idetset, first+i, [&](const StripDigiSimLink& simLink){
          double& adc = simTrackIdToAdc[simLink.SimTrackId()];
          adc += cluster.amplitudes()[i]*simLink.fraction();
        });

      for(const auto& pair: simTrackIdToAdc) {
        simTrackIdToAdc[pair.first] = (adcSum != 0. ? pair.second/adcSum : 0.);
      }

    }
    return simTrackIdToAdc;
  }
}

//
// class declaration
//

class TrackingNtuple : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TrackingNtuple(const edm::ParameterSet&);
  ~TrackingNtuple();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  void clearVariables();

  // This pattern is copied from QuickTrackAssociatorByHitsImpl. If
  // further needs arises, it wouldn't hurt to abstract it somehow.
  typedef std::unordered_set<reco::RecoToSimCollection::index_type> TrackingParticleRefKeySet;
  typedef std::unordered_map<reco::RecoToSimCollection::index_type, size_t> TrackingParticleRefKeyToIndex;
  typedef TrackingParticleRefKeyToIndex TrackingVertexRefKeyToIndex;
  typedef std::pair<TrackPSimHitRef::key_type, edm::ProductID> SimHitFullKey;
  typedef std::map<SimHitFullKey, size_t> SimHitRefKeyToIndex;

  enum class HitType {
    Pixel = 0,
    Strip = 1,
    Glued = 2,
    Invalid = 3,
    Unknown = 99
  };

  // This gives the "best" classification of a reco hit
  // In case of reco hit mathing to multiple sim, smaller number is
  // considered better
  // To be kept in synch with class HitSimType in ntuple.py
  enum class HitSimType {
    Signal = 0,
    ITPileup = 1,
    OOTPileup = 2,
    Noise = 3,
    Unknown = 99
  };

  struct TPHitIndex {
    TPHitIndex(unsigned int tp=0, unsigned int simHit=0, float to=0, unsigned int id=0): tpKey(tp), simHitIdx(simHit), tof(to), detId(id) {}
    unsigned int tpKey;
    unsigned int simHitIdx;
    float tof;
    unsigned int detId;
  };
  static bool tpHitIndexListLess(const TPHitIndex& i, const TPHitIndex& j) { return (i.tpKey < j.tpKey); }
  static bool tpHitIndexListLessSort(const TPHitIndex& i, const TPHitIndex& j) {
    if(i.tpKey == j.tpKey) {
      if(edm::isNotFinite(i.tof) && edm::isNotFinite(j.tof)) {
        return i.detId < j.detId;
      }
      return i.tof < j.tof; // works as intended if either one is NaN
    }
    return i.tpKey < j.tpKey;
  }

  void fillBeamSpot(const reco::BeamSpot& bs);
  void fillPixelHits(const edm::Event& iEvent,
                     const ClusterTPAssociation& clusterToTPMap,
                     const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                     const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                     const edm::DetSetVector<PixelDigiSimLink>& digiSimLink,
                     const TransientTrackingRecHitBuilder& theTTRHBuilder,
                     const TrackerTopology& tTopo,
                     const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                     std::set<edm::ProductID>& hitProductIds
                     );

  void fillStripRphiStereoHits(const edm::Event& iEvent,
                               const ClusterTPAssociation& clusterToTPMap,
                               const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                               const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                               const edm::DetSetVector<StripDigiSimLink>& digiSimLink,
                               const TransientTrackingRecHitBuilder& theTTRHBuilder,
                               const TrackerTopology& tTopo,
                               const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                               std::set<edm::ProductID>& hitProductIds
                               );

  void fillStripMatchedHits(const edm::Event& iEvent,
                            const TransientTrackingRecHitBuilder& theTTRHBuilder,
                            const TrackerTopology& tTopo,
                            std::vector<std::pair<int, int> >& monoStereoClusterList
                            );

  void fillSeeds(const edm::Event& iEvent,
                 const TrackingParticleRefVector& tpCollection,
                 const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                 const reco::BeamSpot& bs,
                 const reco::TrackToTrackingParticleAssociator& associatorByHits,
                 const TransientTrackingRecHitBuilder& theTTRHBuilder,
                 const MagneticField *theMF,
                 const std::vector<std::pair<int, int> >& monoStereoClusterList,
                 const std::set<edm::ProductID>& hitProductIds,
                 std::map<edm::ProductID, size_t>& seedToCollIndex
                 );

  void fillTracks(const edm::RefToBaseVector<reco::Track>& tracks,
                  const TrackingParticleRefVector& tpCollection,
                  const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                  const reco::BeamSpot& bs,
                  const reco::TrackToTrackingParticleAssociator& associatorByHits,
                  const TransientTrackingRecHitBuilder& theTTRHBuilder,
                  const TrackerTopology& tTopo,
                  const std::set<edm::ProductID>& hitProductIds,
                  const std::map<edm::ProductID, size_t>& seedToCollIndex
                  );

  void fillSimHits(const TrackerGeometry& tracker,
                   const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                   const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                   const TrackerTopology& tTopo,
                   SimHitRefKeyToIndex& simHitRefKeyToIndex,
                   std::vector<TPHitIndex>& tpHitList);

  void fillTrackingParticles(const edm::Event& iEvent, const edm::EventSetup& iSetup,
                             const edm::RefToBaseVector<reco::Track>& tracks,
                             const TrackingParticleRefVector& tpCollection,
                             const TrackingVertexRefKeyToIndex& tvKeyToIndex,
                             const reco::TrackToTrackingParticleAssociator& associatorByHits,
                             const std::vector<TPHitIndex>& tpHitList
                             );

  void fillVertices(const reco::VertexCollection& vertices,
                    const edm::RefToBaseVector<reco::Track>& tracks);

  void fillTrackingVertices(const TrackingVertexRefVector& trackingVertices,
                            const TrackingParticleRefKeyToIndex& tpKeyToIndex
                            );



  struct SimHitData {
    std::vector<int> matchingSimHit;
    std::vector<float> chargeFraction;
    std::vector<int> bunchCrossing;
    std::vector<int> event;
    HitSimType type = HitSimType::Unknown;
  };

  template <typename SimLink>
  SimHitData matchCluster(const OmniClusterRef& cluster,
                          DetId hitId, int clusterKey,
                          const TransientTrackingRecHit::RecHitPointer& ttrh,
                          const ClusterTPAssociation& clusterToTPMap,
                          const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                          const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                          const edm::DetSetVector<SimLink>& digiSimLinks,
                          const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                          HitType hitType
                          );

  // ----------member data ---------------------------
  std::vector<edm::EDGetTokenT<edm::View<reco::Track> > > seedTokens_;
  edm::EDGetTokenT<edm::View<reco::Track> > trackToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;
  edm::EDGetTokenT<TrackingParticleRefVector> trackingParticleRefToken_;
  edm::EDGetTokenT<ClusterTPAssociation> clusterTPMapToken_;
  edm::EDGetTokenT<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitTPMapToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociatorToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > pixelSimLinkToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > stripSimLinkToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> stripMatchedRecHitToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int> > tpNLayersToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int> > tpNPixelLayersToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int> > tpNStripStereoLayersToken_;
  std::string builderName_;
  std::string parametersDefinerName_;
  const bool includeSeeds_;
  const bool includeAllHits_;

  TTree* t;
  // event
  edm::RunNumber_t ev_run;
  edm::LuminosityBlockNumber_t ev_lumi;
  edm::EventNumber_t ev_event;

  ////////////////////
  // tracks
  // (first) index runs through tracks
  std::vector<float> trk_px       ;
  std::vector<float> trk_py       ;
  std::vector<float> trk_pz       ;
  std::vector<float> trk_pt       ;
  std::vector<float> trk_inner_px ;
  std::vector<float> trk_inner_py ;
  std::vector<float> trk_inner_pz ;
  std::vector<float> trk_inner_pt ;
  std::vector<float> trk_outer_px ;
  std::vector<float> trk_outer_py ;
  std::vector<float> trk_outer_pz ;
  std::vector<float> trk_outer_pt ;
  std::vector<float> trk_eta      ;
  std::vector<float> trk_lambda   ;
  std::vector<float> trk_cotTheta ;
  std::vector<float> trk_phi      ;
  std::vector<float> trk_dxy      ;
  std::vector<float> trk_dz       ;
  std::vector<float> trk_ptErr    ;
  std::vector<float> trk_etaErr   ;
  std::vector<float> trk_lambdaErr;
  std::vector<float> trk_phiErr   ;
  std::vector<float> trk_dxyErr   ;
  std::vector<float> trk_dzErr    ;
  std::vector<float> trk_refpoint_x;
  std::vector<float> trk_refpoint_y;
  std::vector<float> trk_refpoint_z;
  std::vector<float> trk_nChi2    ;
  std::vector<int> trk_q       ;
  std::vector<unsigned int> trk_nValid  ;
  std::vector<unsigned int> trk_nInvalid;
  std::vector<unsigned int> trk_nPixel  ;
  std::vector<unsigned int> trk_nStrip  ;
  std::vector<unsigned int> trk_nPixelLay;
  std::vector<unsigned int> trk_nStripLay;
  std::vector<unsigned int> trk_n3DLay  ;
  std::vector<unsigned int> trk_nOuterLost;
  std::vector<unsigned int> trk_nInnerLost;
  std::vector<unsigned int> trk_algo    ;
  std::vector<unsigned int> trk_originalAlgo;
  std::vector<decltype(reco::TrackBase().algoMaskUL())> trk_algoMask;
  std::vector<unsigned int> trk_stopReason;
  std::vector<short> trk_isHP    ;
  std::vector<int> trk_seedIdx ;
  std::vector<int> trk_vtxIdx;
  std::vector<std::vector<float> > trk_shareFrac; // second index runs through matched TrackingParticles
  std::vector<std::vector<int> > trk_simTrkIdx;   // second index runs through matched TrackingParticles
  std::vector<std::vector<int> > trk_hitIdx;      // second index runs through hits
  std::vector<std::vector<int> > trk_hitType;     // second index runs through hits
  ////////////////////
  // sim tracks
  // (first) index runs through TrackingParticles
  std::vector<int>   sim_event    ;
  std::vector<int>   sim_bunchCrossing;
  std::vector<int>   sim_pdgId    ;
  std::vector<float> sim_px       ;
  std::vector<float> sim_py       ;
  std::vector<float> sim_pz       ;
  std::vector<float> sim_pt       ;
  std::vector<float> sim_eta      ;
  std::vector<float> sim_phi      ;
  std::vector<float> sim_pca_pt   ;
  std::vector<float> sim_pca_eta  ;
  std::vector<float> sim_pca_lambda;
  std::vector<float> sim_pca_cotTheta;
  std::vector<float> sim_pca_phi  ;
  std::vector<float> sim_pca_dxy  ;
  std::vector<float> sim_pca_dz   ;
  std::vector<int> sim_q       ;
  std::vector<unsigned int> sim_nValid  ;
  std::vector<unsigned int> sim_nPixel  ;
  std::vector<unsigned int> sim_nStrip  ;
  std::vector<unsigned int> sim_nLay;
  std::vector<unsigned int> sim_nPixelLay;
  std::vector<unsigned int> sim_n3DLay  ;
  std::vector<std::vector<int> > sim_trkIdx;      // second index runs through matched tracks
  std::vector<std::vector<float> > sim_shareFrac; // second index runs through matched tracks
  std::vector<int> sim_parentVtxIdx;
  std::vector<std::vector<int> > sim_decayVtxIdx; // second index runs through decay vertices
  std::vector<std::vector<int> > sim_simHitIdx;   // second index runs through SimHits
  ////////////////////
  // pixel hits
  // (first) index runs through hits
  std::vector<short> pix_isBarrel ;
  std::vector<unsigned short> pix_det    ;
  std::vector<unsigned short> pix_lay    ;
  std::vector<unsigned int> pix_detId    ;
  std::vector<std::vector<int> > pix_trkIdx;    // second index runs through tracks containing this hit
  std::vector<std::vector<int> > pix_seeIdx;    // second index runs through seeds containing this hit
  std::vector<std::vector<int> > pix_simHitIdx; // second index runs through SimHits inducing this hit
  std::vector<std::vector<float> > pix_chargeFraction; // second index runs through SimHits inducing this hit
  std::vector<unsigned short> pix_simType;
  std::vector<float> pix_x    ;
  std::vector<float> pix_y    ;
  std::vector<float> pix_z    ;
  std::vector<float> pix_xx   ;
  std::vector<float> pix_xy   ;
  std::vector<float> pix_yy   ;
  std::vector<float> pix_yz   ;
  std::vector<float> pix_zz   ;
  std::vector<float> pix_zx   ;
  std::vector<float> pix_radL ;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> pix_bbxi ;
  ////////////////////
  // strip hits
  // (first) index runs through hits
  std::vector<short> str_isBarrel ;
  std::vector<short> str_isStereo ;
  std::vector<unsigned short> str_det    ;
  std::vector<unsigned short> str_lay    ;
  std::vector<unsigned int> str_detId    ;
  std::vector<std::vector<int> > str_trkIdx;    // second index runs through tracks containing this hit
  std::vector<std::vector<int> > str_seeIdx;    // second index runs through seeds containing this hitw
  std::vector<std::vector<int> > str_simHitIdx; // second index runs through SimHits inducing this hit
  std::vector<std::vector<float> > str_chargeFraction; // second index runs through SimHits inducing this hit
  std::vector<unsigned short> str_simType;
  std::vector<float> str_x    ;
  std::vector<float> str_y    ;
  std::vector<float> str_z    ;
  std::vector<float> str_xx   ;
  std::vector<float> str_xy   ;
  std::vector<float> str_yy   ;
  std::vector<float> str_yz   ;
  std::vector<float> str_zz   ;
  std::vector<float> str_zx   ;
  std::vector<float> str_radL ;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> str_bbxi ;
  ////////////////////
  // strip matched hits
  // (first) index runs through hits
  std::vector<short> glu_isBarrel ;
  std::vector<unsigned int> glu_det      ;
  std::vector<unsigned int> glu_lay      ;
  std::vector<unsigned int> glu_detId    ;
  std::vector<int> glu_monoIdx  ;
  std::vector<int> glu_stereoIdx;
  std::vector<std::vector<int> > glu_seeIdx; // second index runs through seeds containing this hit
  std::vector<float> glu_x    ;
  std::vector<float> glu_y    ;
  std::vector<float> glu_z    ;
  std::vector<float> glu_xx   ;
  std::vector<float> glu_xy   ;
  std::vector<float> glu_yy   ;
  std::vector<float> glu_yz   ;
  std::vector<float> glu_zz   ;
  std::vector<float> glu_zx   ;
  std::vector<float> glu_radL ;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> glu_bbxi ;
  ////////////////////
  // invalid (missing/inactive/etc) hits
  // (first) index runs through hits
  std::vector<short> inv_isBarrel;
  std::vector<unsigned short> inv_det;
  std::vector<unsigned short> inv_lay;
  std::vector<unsigned int> inv_detId;
  std::vector<unsigned short> inv_type;
  ////////////////////
  // sim hits
  // (first) index runs through hits
  std::vector<unsigned short> simhit_det;
  std::vector<unsigned short> simhit_lay;
  std::vector<unsigned int> simhit_detId;
  std::vector<float> simhit_x;
  std::vector<float> simhit_y;
  std::vector<float> simhit_z;
  std::vector<int> simhit_particle;
  std::vector<short> simhit_process;
  std::vector<float> simhit_eloss;
  std::vector<float> simhit_tof;
  //std::vector<unsigned int> simhit_simTrackId; // can be useful for debugging, but not much of general interest
  std::vector<int> simhit_simTrkIdx;
  std::vector<std::vector<int> > simhit_hitIdx;  // second index runs through induced reco hits
  std::vector<std::vector<int> > simhit_hitType; // second index runs through induced reco hits
  ////////////////////
  // beam spot
  float bsp_x;
  float bsp_y;
  float bsp_z;
  float bsp_sigmax;
  float bsp_sigmay;
  float bsp_sigmaz;
  ////////////////////
  // seeds
  // (first) index runs through seeds
  std::vector<short> see_fitok     ;
  std::vector<float> see_px       ;
  std::vector<float> see_py       ;
  std::vector<float> see_pz       ;
  std::vector<float> see_pt       ;
  std::vector<float> see_eta      ;
  std::vector<float> see_phi      ;
  std::vector<float> see_dxy      ;
  std::vector<float> see_dz       ;
  std::vector<float> see_ptErr    ;
  std::vector<float> see_etaErr   ;
  std::vector<float> see_phiErr   ;
  std::vector<float> see_dxyErr   ;
  std::vector<float> see_dzErr    ;
  std::vector<float> see_chi2     ;
  std::vector<int> see_q       ;
  std::vector<unsigned int> see_nValid  ;
  std::vector<unsigned int> see_nPixel  ;
  std::vector<unsigned int> see_nGlued  ;
  std::vector<unsigned int> see_nStrip  ;
  std::vector<unsigned int> see_algo    ;
  std::vector<int> see_trkIdx;
  std::vector<std::vector<float> > see_shareFrac; // second index runs through matched TrackingParticles
  std::vector<std::vector<int> > see_simTrkIdx;   // second index runs through matched TrackingParticles
  std::vector<std::vector<int> > see_hitIdx;      // second index runs through hits
  std::vector<std::vector<int> > see_hitType;     // second index runs through hits
  //seed algo offset, index runs through iterations
  std::vector<unsigned int> see_offset  ;


  ////////////////////
  // Vertices
  // (first) index runs through vertices
  std::vector<float> vtx_x;
  std::vector<float> vtx_y;
  std::vector<float> vtx_z;
  std::vector<float> vtx_xErr;
  std::vector<float> vtx_yErr;
  std::vector<float> vtx_zErr;
  std::vector<float> vtx_ndof;
  std::vector<float> vtx_chi2;
  std::vector<short> vtx_fake;
  std::vector<short> vtx_valid;
  std::vector<std::vector<int> > vtx_trkIdx; // second index runs through tracks used in the vertex fit

  ////////////////////
  // Tracking vertices
  // (first) index runs through TrackingVertices
  std::vector<int>   simvtx_event;
  std::vector<int>   simvtx_bunchCrossing;
  std::vector<unsigned int> simvtx_processType; // only from first SimVertex of TrackingVertex
  std::vector<float> simvtx_x;
  std::vector<float> simvtx_y;
  std::vector<float> simvtx_z;
  std::vector<std::vector<int> > simvtx_sourceSimIdx;   // second index runs through source TrackingParticles
  std::vector<std::vector<int> > simvtx_daughterSimIdx; // second index runs through daughter TrackingParticles
  std::vector<int> simpv_idx;
};

//
// constructors and destructor
//
TrackingNtuple::TrackingNtuple(const edm::ParameterSet& iConfig):
  seedTokens_(edm::vector_transform(iConfig.getUntrackedParameter<std::vector<edm::InputTag> >("seedTracks"), [&](const edm::InputTag& tag) {
        return consumes<edm::View<reco::Track> >(tag);
      })),
  trackToken_(consumes<edm::View<reco::Track> >(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
  clusterTPMapToken_(consumes<ClusterTPAssociation>(iConfig.getUntrackedParameter<edm::InputTag>("clusterTPMap"))),
  simHitTPMapToken_(consumes<SimHitTPAssociationProducer::SimHitTPAssociationList>(iConfig.getUntrackedParameter<edm::InputTag>("simHitTPMap"))),
  trackAssociatorToken_(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociator"))),
  pixelSimLinkToken_(consumes<edm::DetSetVector<PixelDigiSimLink> >(iConfig.getUntrackedParameter<edm::InputTag>("pixelDigiSimLink"))),
  stripSimLinkToken_(consumes<edm::DetSetVector<StripDigiSimLink> >(iConfig.getUntrackedParameter<edm::InputTag>("stripDigiSimLink"))),
  beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamSpot"))),
  pixelRecHitToken_(consumes<SiPixelRecHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("pixelRecHits"))),
  stripRphiRecHitToken_(consumes<SiStripRecHit2DCollection>(iConfig.getUntrackedParameter<edm::InputTag>("stripRphiRecHits"))),
  stripStereoRecHitToken_(consumes<SiStripRecHit2DCollection>(iConfig.getUntrackedParameter<edm::InputTag>("stripStereoRecHits"))),
  stripMatchedRecHitToken_(consumes<SiStripMatchedRecHit2DCollection>(iConfig.getUntrackedParameter<edm::InputTag>("stripMatchedRecHits"))),
  vertexToken_(consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vertices"))),
  trackingVertexToken_(consumes<TrackingVertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trackingVertices"))),
  tpNLayersToken_(consumes<edm::ValueMap<unsigned int> >(iConfig.getUntrackedParameter<edm::InputTag>("trackingParticleNlayers"))),
  tpNPixelLayersToken_(consumes<edm::ValueMap<unsigned int> >(iConfig.getUntrackedParameter<edm::InputTag>("trackingParticleNpixellayers"))),
  tpNStripStereoLayersToken_(consumes<edm::ValueMap<unsigned int> >(iConfig.getUntrackedParameter<edm::InputTag>("trackingParticleNstripstereolayers"))),
  builderName_(iConfig.getUntrackedParameter<std::string>("TTRHBuilder")),
  parametersDefinerName_(iConfig.getUntrackedParameter<std::string>("parametersDefiner")),
  includeSeeds_(iConfig.getUntrackedParameter<bool>("includeSeeds")),
  includeAllHits_(iConfig.getUntrackedParameter<bool>("includeAllHits"))
{
  const bool tpRef = iConfig.getUntrackedParameter<bool>("trackingParticlesRef");
  const auto tpTag = iConfig.getUntrackedParameter<edm::InputTag>("trackingParticles");
  if(tpRef) {
    trackingParticleRefToken_ = consumes<TrackingParticleRefVector>(tpTag);
  }
  else {
    trackingParticleToken_ = consumes<TrackingParticleCollection>(tpTag);
  }

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  t = fs->make<TTree>("tree","tree");

  t->Branch("event"        , &ev_event);
  t->Branch("lumi"         , &ev_lumi);
  t->Branch("run"          , &ev_run);

  //tracks
  t->Branch("trk_px"       , &trk_px);
  t->Branch("trk_py"       , &trk_py);
  t->Branch("trk_pz"       , &trk_pz);
  t->Branch("trk_pt"       , &trk_pt);
  t->Branch("trk_inner_px" , &trk_inner_px);
  t->Branch("trk_inner_py" , &trk_inner_py);
  t->Branch("trk_inner_pz" , &trk_inner_pz);
  t->Branch("trk_inner_pt" , &trk_inner_pt);
  t->Branch("trk_outer_px" , &trk_outer_px);
  t->Branch("trk_outer_py" , &trk_outer_py);
  t->Branch("trk_outer_pz" , &trk_outer_pz);
  t->Branch("trk_outer_pt" , &trk_outer_pt);
  t->Branch("trk_eta"      , &trk_eta);
  t->Branch("trk_lambda"   , &trk_lambda);
  t->Branch("trk_cotTheta" , &trk_cotTheta);
  t->Branch("trk_phi"      , &trk_phi);
  t->Branch("trk_dxy"      , &trk_dxy      );
  t->Branch("trk_dz"       , &trk_dz       );
  t->Branch("trk_ptErr"    , &trk_ptErr    );
  t->Branch("trk_etaErr"   , &trk_etaErr   );
  t->Branch("trk_lambdaErr", &trk_lambdaErr);
  t->Branch("trk_phiErr"   , &trk_phiErr   );
  t->Branch("trk_dxyErr"   , &trk_dxyErr   );
  t->Branch("trk_dzErr"    , &trk_dzErr    );
  t->Branch("trk_refpoint_x", &trk_refpoint_x);
  t->Branch("trk_refpoint_y", &trk_refpoint_y);
  t->Branch("trk_refpoint_z", &trk_refpoint_z);
  t->Branch("trk_nChi2"    , &trk_nChi2);
  t->Branch("trk_q"        , &trk_q);
  t->Branch("trk_nValid"   , &trk_nValid  );
  t->Branch("trk_nInvalid" , &trk_nInvalid);
  t->Branch("trk_nPixel"   , &trk_nPixel  );
  t->Branch("trk_nStrip"   , &trk_nStrip  );
  t->Branch("trk_nPixelLay", &trk_nPixelLay);
  t->Branch("trk_nStripLay", &trk_nStripLay);
  t->Branch("trk_n3DLay"   , &trk_n3DLay  );
  t->Branch("trk_nOuterLost", &trk_nOuterLost  );
  t->Branch("trk_nInnerLost", &trk_nInnerLost  );
  t->Branch("trk_algo"     , &trk_algo    );
  t->Branch("trk_originalAlgo", &trk_originalAlgo);
  t->Branch("trk_algoMask" , &trk_algoMask);
  t->Branch("trk_stopReason", &trk_stopReason);
  t->Branch("trk_isHP"     , &trk_isHP    );
  if(includeSeeds_) {
    t->Branch("trk_seedIdx"  , &trk_seedIdx );
  }
  t->Branch("trk_vtxIdx"   , &trk_vtxIdx);
  t->Branch("trk_shareFrac", &trk_shareFrac);
  t->Branch("trk_simTrkIdx", &trk_simTrkIdx  );
  if(includeAllHits_) {
    t->Branch("trk_hitIdx" , &trk_hitIdx);
    t->Branch("trk_hitType", &trk_hitType);
  }
  //sim tracks
  t->Branch("sim_event"    , &sim_event    );
  t->Branch("sim_bunchCrossing", &sim_bunchCrossing);
  t->Branch("sim_pdgId"    , &sim_pdgId    );
  t->Branch("sim_px"       , &sim_px       );
  t->Branch("sim_py"       , &sim_py       );
  t->Branch("sim_pz"       , &sim_pz       );
  t->Branch("sim_pt"       , &sim_pt       );
  t->Branch("sim_eta"      , &sim_eta      );
  t->Branch("sim_phi"      , &sim_phi      );
  t->Branch("sim_pca_pt"   , &sim_pca_pt  );
  t->Branch("sim_pca_eta"  , &sim_pca_eta  );
  t->Branch("sim_pca_lambda", &sim_pca_lambda);
  t->Branch("sim_pca_cotTheta", &sim_pca_cotTheta);
  t->Branch("sim_pca_phi"  , &sim_pca_phi  );
  t->Branch("sim_pca_dxy"  , &sim_pca_dxy  );
  t->Branch("sim_pca_dz"   , &sim_pca_dz   );
  t->Branch("sim_q"        , &sim_q        );
  t->Branch("sim_nValid"   , &sim_nValid   );
  t->Branch("sim_nPixel"   , &sim_nPixel   );
  t->Branch("sim_nStrip"   , &sim_nStrip   );
  t->Branch("sim_nLay"     , &sim_nLay     );
  t->Branch("sim_nPixelLay", &sim_nPixelLay);
  t->Branch("sim_n3DLay"   , &sim_n3DLay   );
  t->Branch("sim_trkIdx"   , &sim_trkIdx   );
  t->Branch("sim_shareFrac", &sim_shareFrac);
  t->Branch("sim_parentVtxIdx", &sim_parentVtxIdx);
  t->Branch("sim_decayVtxIdx", &sim_decayVtxIdx);
  if(includeAllHits_) {
    t->Branch("sim_simHitIdx" , &sim_simHitIdx );
  }
  if(includeAllHits_) {
    //pixels
    t->Branch("pix_isBarrel"  , &pix_isBarrel );
    t->Branch("pix_det"       , &pix_det      );
    t->Branch("pix_lay"       , &pix_lay      );
    t->Branch("pix_detId"     , &pix_detId    );
    t->Branch("pix_trkIdx"    , &pix_trkIdx   );
    if(includeSeeds_) {
      t->Branch("pix_seeIdx"    , &pix_seeIdx   );
    }
    t->Branch("pix_simHitIdx" , &pix_simHitIdx);
    t->Branch("pix_chargeFraction", &pix_chargeFraction);
    t->Branch("pix_simType", &pix_simType);
    t->Branch("pix_x"     , &pix_x    );
    t->Branch("pix_y"     , &pix_y    );
    t->Branch("pix_z"     , &pix_z    );
    t->Branch("pix_xx"    , &pix_xx   );
    t->Branch("pix_xy"    , &pix_xy   );
    t->Branch("pix_yy"    , &pix_yy   );
    t->Branch("pix_yz"    , &pix_yz   );
    t->Branch("pix_zz"    , &pix_zz   );
    t->Branch("pix_zx"    , &pix_zx   );
    t->Branch("pix_radL"  , &pix_radL );
    t->Branch("pix_bbxi"  , &pix_bbxi );
    t->Branch("pix_bbxi"  , &pix_bbxi );
    //strips
    t->Branch("str_isBarrel"  , &str_isBarrel );
    t->Branch("str_isStereo"  , &str_isStereo );
    t->Branch("str_det"       , &str_det      );
    t->Branch("str_lay"       , &str_lay      );
    t->Branch("str_detId"     , &str_detId    );
    t->Branch("str_trkIdx"    , &str_trkIdx   );
    if(includeSeeds_) {
      t->Branch("str_seeIdx"    , &str_seeIdx   );
    }
    t->Branch("str_simHitIdx" , &str_simHitIdx);
    t->Branch("str_chargeFraction", &str_chargeFraction);
    t->Branch("str_simType", &str_simType);
    t->Branch("str_x"     , &str_x    );
    t->Branch("str_y"     , &str_y    );
    t->Branch("str_z"     , &str_z    );
    t->Branch("str_xx"    , &str_xx   );
    t->Branch("str_xy"    , &str_xy   );
    t->Branch("str_yy"    , &str_yy   );
    t->Branch("str_yz"    , &str_yz   );
    t->Branch("str_zz"    , &str_zz   );
    t->Branch("str_zx"    , &str_zx   );
    t->Branch("str_radL"  , &str_radL );
    t->Branch("str_bbxi"  , &str_bbxi );
    //matched hits
    t->Branch("glu_isBarrel"  , &glu_isBarrel );
    t->Branch("glu_det"       , &glu_det      );
    t->Branch("glu_lay"       , &glu_lay      );
    t->Branch("glu_detId"     , &glu_detId    );
    t->Branch("glu_monoIdx"   , &glu_monoIdx  );
    t->Branch("glu_stereoIdx" , &glu_stereoIdx);
    if(includeSeeds_) {
      t->Branch("glu_seeIdx"    , &glu_seeIdx   );
    }
    t->Branch("glu_x"         , &glu_x        );
    t->Branch("glu_y"         , &glu_y        );
    t->Branch("glu_z"         , &glu_z        );
    t->Branch("glu_xx"        , &glu_xx       );
    t->Branch("glu_xy"        , &glu_xy       );
    t->Branch("glu_yy"        , &glu_yy       );
    t->Branch("glu_yz"        , &glu_yz       );
    t->Branch("glu_zz"        , &glu_zz       );
    t->Branch("glu_zx"        , &glu_zx       );
    t->Branch("glu_radL"      , &glu_radL     );
    t->Branch("glu_bbxi"      , &glu_bbxi     );
    //invalid hits
    t->Branch("inv_isBarrel"  , &inv_isBarrel );
    t->Branch("inv_det"       , &inv_det      );
    t->Branch("inv_lay"       , &inv_lay      );
    t->Branch("inv_detId"     , &inv_detId    );
    t->Branch("inv_type"      , &inv_type    );
    //simhits
    t->Branch("simhit_det"     ,  &simhit_det);
    t->Branch("simhit_lay"    ,  &simhit_lay);
    t->Branch("simhit_detId"   ,  &simhit_detId);
    t->Branch("simhit_x"       ,  &simhit_x);
    t->Branch("simhit_y"       ,  &simhit_y);
    t->Branch("simhit_z"       ,  &simhit_z);
    t->Branch("simhit_particle",  &simhit_particle);
    t->Branch("simhit_process" ,  &simhit_process);
    t->Branch("simhit_eloss"   ,  &simhit_eloss);
    t->Branch("simhit_tof"     ,  &simhit_tof);
    //t->Branch("simhit_simTrackId", &simhit_simTrackId);
    t->Branch("simhit_simTrkIdx", &simhit_simTrkIdx);
    t->Branch("simhit_hitIdx"  ,  &simhit_hitIdx);
    t->Branch("simhit_hitType" ,  &simhit_hitType);
  }
  //beam spot
  t->Branch("bsp_x" , &bsp_x , "bsp_x/F");
  t->Branch("bsp_y" , &bsp_y , "bsp_y/F");
  t->Branch("bsp_z" , &bsp_z , "bsp_z/F");
  t->Branch("bsp_sigmax" , &bsp_sigmax , "bsp_sigmax/F");
  t->Branch("bsp_sigmay" , &bsp_sigmay , "bsp_sigmay/F");
  t->Branch("bsp_sigmaz" , &bsp_sigmaz , "bsp_sigmaz/F");
  if(includeSeeds_) {
    //seeds
    t->Branch("see_fitok"    , &see_fitok   );
    t->Branch("see_px"       , &see_px      );
    t->Branch("see_py"       , &see_py      );
    t->Branch("see_pz"       , &see_pz      );
    t->Branch("see_pt"       , &see_pt      );
    t->Branch("see_eta"      , &see_eta     );
    t->Branch("see_phi"      , &see_phi     );
    t->Branch("see_dxy"      , &see_dxy     );
    t->Branch("see_dz"       , &see_dz      );
    t->Branch("see_ptErr"    , &see_ptErr   );
    t->Branch("see_etaErr"   , &see_etaErr  );
    t->Branch("see_phiErr"   , &see_phiErr  );
    t->Branch("see_dxyErr"   , &see_dxyErr  );
    t->Branch("see_dzErr"    , &see_dzErr   );
    t->Branch("see_chi2"     , &see_chi2    );
    t->Branch("see_q"        , &see_q       );
    t->Branch("see_nValid"   , &see_nValid  );
    t->Branch("see_nPixel"   , &see_nPixel  );
    t->Branch("see_nGlued"   , &see_nGlued  );
    t->Branch("see_nStrip"   , &see_nStrip  );
    t->Branch("see_algo"     , &see_algo    );
    t->Branch("see_trkIdx"   , &see_trkIdx  );
    t->Branch("see_shareFrac", &see_shareFrac);
    t->Branch("see_simTrkIdx", &see_simTrkIdx  );
    if(includeAllHits_) {
      t->Branch("see_hitIdx" , &see_hitIdx  );
      t->Branch("see_hitType", &see_hitType );
    }
    //seed algo offset
    t->Branch("see_offset"  , &see_offset );
  }

  //vertices
  t->Branch("vtx_x"       , &vtx_x);
  t->Branch("vtx_y"       , &vtx_y);
  t->Branch("vtx_z"       , &vtx_z);
  t->Branch("vtx_xErr"    , &vtx_xErr);
  t->Branch("vtx_yErr"    , &vtx_yErr);
  t->Branch("vtx_zErr"    , &vtx_zErr);
  t->Branch("vtx_ndof"    , &vtx_ndof);
  t->Branch("vtx_chi2"    , &vtx_chi2);
  t->Branch("vtx_fake"    , &vtx_fake);
  t->Branch("vtx_valid"   , &vtx_valid);
  t->Branch("vtx_trkIdx"  , &vtx_trkIdx);

  // tracking vertices
  t->Branch("simvtx_event"   , &simvtx_event    );
  t->Branch("simvtx_bunchCrossing", &simvtx_bunchCrossing);
  t->Branch("simvtx_processType", &simvtx_processType);
  t->Branch("simvtx_x"       , &simvtx_x);
  t->Branch("simvtx_y"       , &simvtx_y);
  t->Branch("simvtx_z"       , &simvtx_z);
  t->Branch("simvtx_sourceSimIdx", &simvtx_sourceSimIdx);
  t->Branch("simvtx_daughterSimIdx", &simvtx_daughterSimIdx);

  t->Branch("simpv_idx"      , &simpv_idx);

  //t->Branch("" , &);
}


TrackingNtuple::~TrackingNtuple() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//
void TrackingNtuple::clearVariables() {

  ev_run = 0;
  ev_lumi = 0;
  ev_event = 0;

  //tracks
  trk_px       .clear();
  trk_py       .clear();
  trk_pz       .clear();
  trk_pt       .clear();
  trk_inner_px .clear();
  trk_inner_py .clear();
  trk_inner_pz .clear();
  trk_inner_pt .clear();
  trk_outer_px .clear();
  trk_outer_py .clear();
  trk_outer_pz .clear();
  trk_outer_pt .clear();
  trk_eta      .clear();
  trk_lambda   .clear();
  trk_cotTheta .clear();
  trk_phi      .clear();
  trk_dxy      .clear();
  trk_dz       .clear();
  trk_ptErr    .clear();
  trk_etaErr   .clear();
  trk_lambdaErr.clear();
  trk_phiErr   .clear();
  trk_dxyErr   .clear();
  trk_dzErr    .clear();
  trk_refpoint_x.clear();
  trk_refpoint_y.clear();
  trk_refpoint_z.clear();
  trk_nChi2    .clear();
  trk_q        .clear();
  trk_nValid   .clear();
  trk_nInvalid .clear();
  trk_nPixel   .clear();
  trk_nStrip   .clear();
  trk_nPixelLay.clear();
  trk_nStripLay.clear();
  trk_n3DLay   .clear();
  trk_nOuterLost.clear();
  trk_nInnerLost.clear();
  trk_algo     .clear();
  trk_originalAlgo.clear();
  trk_algoMask .clear();
  trk_stopReason.clear();
  trk_isHP     .clear();
  trk_seedIdx  .clear();
  trk_vtxIdx   .clear();
  trk_shareFrac.clear();
  trk_simTrkIdx.clear();
  trk_hitIdx   .clear();
  trk_hitType  .clear();
  //sim tracks
  sim_event    .clear();
  sim_bunchCrossing.clear();
  sim_pdgId    .clear();
  sim_px       .clear();
  sim_py       .clear();
  sim_pz       .clear();
  sim_pt       .clear();
  sim_eta      .clear();
  sim_phi      .clear();
  sim_pca_pt   .clear();
  sim_pca_eta  .clear();
  sim_pca_lambda.clear();
  sim_pca_cotTheta.clear();
  sim_pca_phi  .clear();
  sim_pca_dxy  .clear();
  sim_pca_dz   .clear();
  sim_q        .clear();
  sim_nValid   .clear();
  sim_nPixel   .clear();
  sim_nStrip   .clear();
  sim_nLay     .clear();
  sim_nPixelLay.clear();
  sim_n3DLay   .clear();
  sim_trkIdx   .clear();
  sim_shareFrac.clear();
  sim_parentVtxIdx.clear();
  sim_decayVtxIdx.clear();
  sim_simHitIdx   .clear();
  //pixels
  pix_isBarrel .clear();
  pix_det      .clear();
  pix_lay      .clear();
  pix_detId    .clear();
  pix_trkIdx   .clear();
  pix_seeIdx   .clear();
  pix_simHitIdx.clear();
  pix_chargeFraction.clear();
  pix_simType.clear();
  pix_x    .clear();
  pix_y    .clear();
  pix_z    .clear();
  pix_xx   .clear();
  pix_xy   .clear();
  pix_yy   .clear();
  pix_yz   .clear();
  pix_zz   .clear();
  pix_zx   .clear();
  pix_radL .clear();
  pix_bbxi .clear();
  //strips
  str_isBarrel .clear();
  str_isStereo .clear();
  str_det      .clear();
  str_lay      .clear();
  str_detId    .clear();
  str_trkIdx   .clear();
  str_seeIdx   .clear();
  str_simHitIdx.clear();
  str_chargeFraction.clear();
  str_simType.clear();
  str_x    .clear();
  str_y    .clear();
  str_z    .clear();
  str_xx   .clear();
  str_xy   .clear();
  str_yy   .clear();
  str_yz   .clear();
  str_zz   .clear();
  str_zx   .clear();
  str_radL .clear();
  str_bbxi .clear();
  //matched hits
  glu_isBarrel .clear();
  glu_det      .clear();
  glu_lay      .clear();
  glu_detId    .clear();
  glu_monoIdx  .clear();
  glu_stereoIdx.clear();
  glu_seeIdx   .clear();
  glu_x        .clear();
  glu_y        .clear();
  glu_z        .clear();
  glu_xx       .clear();
  glu_xy       .clear();
  glu_yy       .clear();
  glu_yz       .clear();
  glu_zz       .clear();
  glu_zx       .clear();
  glu_radL     .clear();
  glu_bbxi     .clear();
  //invalid hits
  inv_isBarrel .clear();
  inv_det      .clear();
  inv_lay      .clear();
  inv_detId    .clear();
  inv_type     .clear();
  // simhits
  simhit_det.clear();
  simhit_lay.clear();
  simhit_detId.clear();
  simhit_x.clear();
  simhit_y.clear();
  simhit_z.clear();
  simhit_particle.clear();
  simhit_process.clear();
  simhit_eloss.clear();
  simhit_tof.clear();
  //simhit_simTrackId.clear();
  simhit_simTrkIdx.clear();
  simhit_hitIdx.clear();
  simhit_hitType.clear();
  //beamspot
  bsp_x = -9999.;
  bsp_y = -9999.;
  bsp_z = -9999.;
  bsp_sigmax = -9999.;
  bsp_sigmay = -9999.;
  bsp_sigmaz = -9999.;
  //seeds
  see_fitok   .clear();
  see_px      .clear();
  see_py      .clear();
  see_pz      .clear();
  see_pt      .clear();
  see_eta     .clear();
  see_phi     .clear();
  see_dxy     .clear();
  see_dz      .clear();
  see_ptErr   .clear();
  see_etaErr  .clear();
  see_phiErr  .clear();
  see_dxyErr  .clear();
  see_dzErr   .clear();
  see_chi2    .clear();
  see_q       .clear();
  see_nValid  .clear();
  see_nPixel  .clear();
  see_nGlued  .clear();
  see_nStrip  .clear();
  see_algo    .clear();
  see_trkIdx  .clear();
  see_shareFrac.clear();
  see_simTrkIdx.clear();
  see_hitIdx  .clear();
  see_hitType .clear();
  //seed algo offset
  see_offset.clear();

  // vertices
  vtx_x.clear();
  vtx_y.clear();
  vtx_z.clear();
  vtx_xErr.clear();
  vtx_yErr.clear();
  vtx_zErr.clear();
  vtx_ndof.clear();
  vtx_chi2.clear();
  vtx_fake.clear();
  vtx_valid.clear();
  vtx_trkIdx.clear();

  // Tracking vertices
  simvtx_event.clear();
  simvtx_bunchCrossing.clear();
  simvtx_x.clear();
  simvtx_y.clear();
  simvtx_z.clear();
  simvtx_sourceSimIdx.clear();
  simvtx_daughterSimIdx.clear();
  simpv_idx.clear();
}


// ------------ method called for each event  ------------
void TrackingNtuple::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace reco;
  using namespace std;

  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);

  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;
  iSetup.get<TransientRecHitRecord>().get(builderName_,theTTRHBuilder);

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology& tTopo = *tTopoHandle;

  edm::ESHandle<TrackerGeometry> geometryHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  const TrackerGeometry &tracker = *geometryHandle;

  edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
  iEvent.getByToken(trackAssociatorToken_, theAssociator);
  const reco::TrackToTrackingParticleAssociator& associatorByHits = *theAssociator;

  LogDebug("TrackingNtuple") << "Analyzing new event";

  //initialize tree variables
  clearVariables();

  // FIXME: we really need to move to edm::View for reading the
  // TrackingParticles... Unfortunately it has non-trivial
  // consequences on the associator/association interfaces etc.
  TrackingParticleRefVector tmpTP;
  TrackingParticleRefKeySet tmpTPkeys;
  const TrackingParticleRefVector *tmpTPptr = nullptr;
  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  edm::Handle<TrackingParticleRefVector>  TPCollectionHRefVector;

  if(!trackingParticleToken_.isUninitialized()) {
    iEvent.getByToken(trackingParticleToken_, TPCollectionH);
    for(size_t i=0, size=TPCollectionH->size(); i<size; ++i) {
      tmpTP.push_back(TrackingParticleRef(TPCollectionH, i));
    }
    tmpTPptr = &tmpTP;
  }
  else {
    iEvent.getByToken(trackingParticleRefToken_, TPCollectionHRefVector);
    tmpTPptr = TPCollectionHRefVector.product();
    for(const auto& ref: *tmpTPptr) {
      tmpTPkeys.insert(ref.key());
    }
  }
  const TrackingParticleRefVector& tpCollection = *tmpTPptr;

  // Fill mapping from Ref::key() to index
  TrackingParticleRefKeyToIndex tpKeyToIndex;
  for(size_t i=0; i<tpCollection.size(); ++i) {
    tpKeyToIndex[tpCollection[i].key()] = i;
  }

  // tracking vertices
  edm::Handle<TrackingVertexCollection> htv;
  iEvent.getByToken(trackingVertexToken_, htv);
  const TrackingVertexCollection& tvs = *htv;

  // Fill mapping from Ref::key() to index
  TrackingVertexRefVector tvRefs;
  TrackingVertexRefKeyToIndex tvKeyToIndex;
  for(size_t i=0; i<tvs.size(); ++i) {
    const TrackingVertex& v = tvs[i];
    if(v.eventId().bunchCrossing() != 0) // Ignore OOTPU; would be better to not to hardcode?
      continue;
    tvKeyToIndex[i] = tvRefs.size();
    tvRefs.push_back(TrackingVertexRef(htv, i));
  }

  //get association maps, etc.
  Handle<ClusterTPAssociation> pCluster2TPListH;
  iEvent.getByToken(clusterTPMapToken_, pCluster2TPListH);
  const ClusterTPAssociation& clusterToTPMap = *pCluster2TPListH;
  edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
  iEvent.getByToken(simHitTPMapToken_, simHitsTPAssoc);

  // SimHit key -> index mapping
  SimHitRefKeyToIndex simHitRefKeyToIndex;

  //make a list to link TrackingParticles to its simhits
  std::vector<TPHitIndex> tpHitList;

  std::set<edm::ProductID> hitProductIds;
  std::map<edm::ProductID, size_t> seedCollToOffset;

  ev_run = iEvent.id().run();
  ev_lumi = iEvent.id().luminosityBlock();
  ev_event = iEvent.id().event();

  // Digi->Sim links for pixels and strips
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > pixelDigiSimLinksHandle;
  iEvent.getByToken(pixelSimLinkToken_, pixelDigiSimLinksHandle);
  const auto& pixelDigiSimLinks = *pixelDigiSimLinksHandle;

  edm::Handle<edm::DetSetVector<StripDigiSimLink> > stripDigiSimLinksHandle;
  iEvent.getByToken(stripSimLinkToken_, stripDigiSimLinksHandle);
  const auto& stripDigiSimLinks = *stripDigiSimLinksHandle;

  //beamspot
  Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_, recoBeamSpotHandle);
  BeamSpot const & bs = *recoBeamSpotHandle;
  fillBeamSpot(bs);

  //prapare list to link matched hits to collection
  vector<pair<int,int> > monoStereoClusterList;
  if(includeAllHits_) {
    // simhits
    fillSimHits(tracker, tpKeyToIndex, *simHitsTPAssoc, tTopo, simHitRefKeyToIndex, tpHitList);

    //pixel hits
    fillPixelHits(iEvent, clusterToTPMap, tpKeyToIndex, *simHitsTPAssoc, pixelDigiSimLinks, *theTTRHBuilder, tTopo, simHitRefKeyToIndex, hitProductIds);

    //strip hits
    fillStripRphiStereoHits(iEvent, clusterToTPMap, tpKeyToIndex, *simHitsTPAssoc, stripDigiSimLinks, *theTTRHBuilder, tTopo, simHitRefKeyToIndex, hitProductIds);

    //matched hits
    fillStripMatchedHits(iEvent, *theTTRHBuilder, tTopo, monoStereoClusterList);
  }

  //seeds
  if(includeSeeds_) {
    fillSeeds(iEvent, tpCollection, tpKeyToIndex, bs, associatorByHits, *theTTRHBuilder, theMF.product(), monoStereoClusterList, hitProductIds, seedCollToOffset);
  }

  //tracks
  edm::Handle<edm::View<reco::Track> > tracksHandle;
  iEvent.getByToken(trackToken_, tracksHandle);
  const edm::View<reco::Track>& tracks = *tracksHandle;
  // The associator interfaces really need to be fixed...
  edm::RefToBaseVector<reco::Track> trackRefs;
  for(edm::View<Track>::size_type i=0; i<tracks.size(); ++i) {
    trackRefs.push_back(tracks.refAt(i));
  }
  fillTracks(trackRefs, tpCollection, tpKeyToIndex, bs, associatorByHits, *theTTRHBuilder, tTopo, hitProductIds, seedCollToOffset);

  //tracking particles
  //sort association maps with simHits
  std::sort( tpHitList.begin(), tpHitList.end(), tpHitIndexListLessSort );
  fillTrackingParticles(iEvent, iSetup, trackRefs, tpCollection, tvKeyToIndex, associatorByHits, tpHitList);

  // vertices
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vertexToken_, vertices);
  fillVertices(*vertices, trackRefs);

  // tracking vertices
  fillTrackingVertices(tvRefs, tpKeyToIndex);

  t->Fill();

}

void TrackingNtuple::fillBeamSpot(const reco::BeamSpot& bs) {
  bsp_x = bs.x0();
  bsp_y = bs.y0();
  bsp_z = bs.x0();
  bsp_sigmax = bs.BeamWidthX();
  bsp_sigmay = bs.BeamWidthY();
  bsp_sigmaz = bs.sigmaZ();
}

namespace {
  template <typename SimLink> struct GetCluster;
  template <>
  struct GetCluster<PixelDigiSimLink> {
    static const SiPixelCluster& call(const OmniClusterRef& cluster) { return cluster.pixelCluster(); }
  };
  template <>
  struct GetCluster<StripDigiSimLink> {
    static const SiStripCluster& call(const OmniClusterRef& cluster) { return cluster.stripCluster(); }
  };
}

template <typename SimLink>
TrackingNtuple::SimHitData TrackingNtuple::matchCluster(const OmniClusterRef& cluster,
                                                        DetId hitId, int clusterKey,
                                                        const TransientTrackingRecHit::RecHitPointer& ttrh,
                                                        const ClusterTPAssociation& clusterToTPMap,
                                                        const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                                        const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                                                        const edm::DetSetVector<SimLink>& digiSimLinks,
                                                        const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                                                        HitType hitType
                                                        ) {
  SimHitData ret;

  auto simTrackIdToChargeFraction = chargeFraction(GetCluster<SimLink>::call(cluster), hitId, digiSimLinks);

  ret.type = HitSimType::Noise;
  auto range = clusterToTPMap.equal_range( cluster );
  if( range.first != range.second ) {
    for( auto ip=range.first; ip != range.second; ++ip ) {
      const TrackingParticleRef& trackingParticle = ip->second;

      // Find out if the cluster is from signal/ITPU/OOTPU
      const auto event = trackingParticle->eventId().event();
      const auto bx = trackingParticle->eventId().bunchCrossing();
      HitSimType type = HitSimType::OOTPileup;
      if(bx == 0) {
        type = (event == 0 ? HitSimType::Signal : HitSimType::ITPileup);
      }
      ret.type = static_cast<HitSimType>(std::min(static_cast<int>(ret.type), static_cast<int>(type)));

      // Limit to only input TrackingParticles (usually signal+ITPU)
      auto tpIndex = tpKeyToIndex.find(trackingParticle.key());
      if( tpIndex == tpKeyToIndex.end())
        continue;
      if( trackingParticle->numberOfHits() == 0 ) continue;

      //now get the corresponding sim hit
      std::pair<TrackingParticleRef, TrackPSimHitRef> simHitTPpairWithDummyTP(trackingParticle,TrackPSimHitRef());
      //SimHit is dummy: for simHitTPAssociationListGreater sorting only the TP is needed
      auto range = std::equal_range(simHitsTPAssoc.begin(), simHitsTPAssoc.end(),
                                    simHitTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
      int simHitKey = -1;
      bool foundElectron = false;
      edm::ProductID simHitID;
      for(auto ip = range.first; ip != range.second; ++ip) {
        TrackPSimHitRef TPhit = ip->second;
        DetId dId = DetId(TPhit->detUnitId());
        if (dId.rawId()==hitId.rawId()) {
          // skip electron SimHits for non-electron TPs also here
          if(std::abs(TPhit->particleType()) == 11 && std::abs(trackingParticle->pdgId()) != 11) {
            foundElectron = true;
            continue;
          }

          simHitKey = TPhit.key();
          simHitID = TPhit.id();
          break;
        }
      }
      if(simHitKey < 0) {
        // In case we didn't find a simhit because of filtered-out
        // electron SimHit, just ignore the missing SimHit.
        if(foundElectron)
          continue;

        auto ex = cms::Exception("LogicError") << "Did not find SimHit for reco hit DetId " << hitId.rawId()
                                               << " for TP " << trackingParticle.key() << " bx:event " << bx << ":" << event
                                               << ".\nFound SimHits from detectors ";
        for(auto ip = range.first; ip != range.second; ++ip) {
          TrackPSimHitRef TPhit = ip->second;
          DetId dId = DetId(TPhit->detUnitId());
          ex << dId.rawId() << " ";
        }
        if(trackingParticle->eventId().event() != 0) {
          ex << "\nSince this is a TrackingParticle from pileup, check that you're running the pileup mixing in playback mode.";
        }
        throw ex;
      }
      auto simHitIndex = simHitRefKeyToIndex.at(std::make_pair(simHitKey, simHitID));
      ret.matchingSimHit.push_back(simHitIndex);

      double chargeFraction = 0.;
      for(const SimTrack& simtrk: trackingParticle->g4Tracks()) {
        auto found = simTrackIdToChargeFraction.find(simtrk.trackId());
        if(found != simTrackIdToChargeFraction.end()) {
          chargeFraction += found->second;
        }
      }
      ret.chargeFraction.push_back(chargeFraction);

      // only for debug prints
      ret.bunchCrossing.push_back(bx);
      ret.event.push_back(event);

      simhit_hitIdx[simHitIndex].push_back(clusterKey);
      simhit_hitType[simHitIndex].push_back(static_cast<int>(hitType));
    }
  }

  return ret;
}

void TrackingNtuple::fillSimHits(const TrackerGeometry& tracker,
                                 const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                 const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                                 const TrackerTopology& tTopo,
                                 SimHitRefKeyToIndex& simHitRefKeyToIndex,
                                 std::vector<TPHitIndex>& tpHitList) {

  for(const auto& assoc: simHitsTPAssoc) {
    auto tpKey = assoc.first.key();

    // SimHitTPAssociationList can contain more TrackingParticles than
    // what are given to this EDAnalyzer, so we can filter those out here.
    auto found = tpKeyToIndex.find(tpKey);
    if(found == tpKeyToIndex.end())
      continue;
    const auto tpIndex = found->second;

    // skip non-tracker simhits (mostly muons)
    const auto& simhit = *(assoc.second);
    auto detId = DetId(simhit.detUnitId());
    if(detId.det() != DetId::Tracker) continue;

    // Skip electron SimHits for non-electron TrackingParticles to
    // filter out delta rays. The delta ray hits just confuse. If we
    // need them later, let's add them as a separate "collection" of
    // hits of a TP
    const TrackingParticle& tp = *(assoc.first);
    if(std::abs(simhit.particleType()) == 11 && std::abs(tp.pdgId()) != 11) continue;

    auto simHitKey = std::make_pair(assoc.second.key(), assoc.second.id());

    if(simHitRefKeyToIndex.find(simHitKey) != simHitRefKeyToIndex.end()) {
      for(const auto& assoc2: simHitsTPAssoc) {
        if(std::make_pair(assoc2.second.key(), assoc2.second.id()) == simHitKey) {

#ifdef EDM_ML_DEBUG
          auto range1 = std::equal_range(simHitsTPAssoc.begin(), simHitsTPAssoc.end(),
                                         std::make_pair(assoc.first, TrackPSimHitRef()),
                                         SimHitTPAssociationProducer::simHitTPAssociationListGreater);
          auto range2 = std::equal_range(simHitsTPAssoc.begin(), simHitsTPAssoc.end(),
                                         std::make_pair(assoc2.first, TrackPSimHitRef()),
                                         SimHitTPAssociationProducer::simHitTPAssociationListGreater);

          LogTrace("TrackingNtuple") << "Earlier TP " << assoc2.first.key() << " SimTrack Ids";
          for(const auto& simTrack: assoc2.first->g4Tracks()) {
            edm::LogPrint("TrackingNtuple") << " SimTrack " << simTrack.trackId() << " BX:event " << simTrack.eventId().bunchCrossing() << ":" << simTrack.eventId().event();
          }
          for(auto iHit = range2.first; iHit != range2.second; ++iHit) {
            LogTrace("TrackingNtuple") << " SimHit " << iHit->second.key() << " " << iHit->second.id() << " tof " << iHit->second->tof() << " trackId " << iHit->second->trackId() << " BX:event " << iHit->second->eventId().bunchCrossing() << ":" << iHit->second->eventId().event();
          }
          LogTrace("TrackingNtuple") << "Current TP " << assoc.first.key() << " SimTrack Ids";
          for(const auto& simTrack: assoc.first->g4Tracks()) {
            edm::LogPrint("TrackingNtuple") << " SimTrack " << simTrack.trackId() << " BX:event " << simTrack.eventId().bunchCrossing() << ":" << simTrack.eventId().event();
          }
          for(auto iHit = range1.first; iHit != range1.second; ++iHit) {
            LogTrace("TrackingNtuple") << " SimHit " << iHit->second.key() << " " << iHit->second.id() << " tof " << iHit->second->tof() << " trackId " << iHit->second->trackId() << " BX:event " << iHit->second->eventId().bunchCrossing() << ":" << iHit->second->eventId().event();
          }
#endif

          throw cms::Exception("LogicError") << "Got second time the SimHit " << simHitKey.first << " of " << simHitKey.second << ", first time with TrackingParticle " << assoc2.first.key() << ", now with " << tpKey;
        }
      }
      throw cms::Exception("LogicError") << "Got second time the SimHit " << simHitKey.first << " of " << simHitKey.second << ", now with TrackingParticle " << tpKey << ", but I didn't find the first occurrance!";
    }

    auto det = tracker.idToDetUnit(detId);
    if(!det)
      throw cms::Exception("LogicError") << "Did not find a det unit for DetId " << simhit.detUnitId() << " from tracker geometry";

    const auto pos = det->surface().toGlobal(simhit.localPosition());
    const float tof = simhit.timeOfFlight();

    const auto simHitIndex = simhit_detId.size();
    simHitRefKeyToIndex[simHitKey] = simHitIndex;

    simhit_det.push_back(detId.subdetId());
    simhit_lay.push_back(tTopo.layer(detId));
    simhit_detId.push_back(detId.rawId());
    simhit_x.push_back(pos.x());
    simhit_y.push_back(pos.y());
    simhit_z.push_back(pos.z());
    simhit_particle.push_back(simhit.particleType());
    simhit_process.push_back(simhit.processType());
    simhit_eloss.push_back(simhit.energyLoss());
    simhit_tof.push_back(tof);
    //simhit_simTrackId.push_back(simhit.trackId());

    simhit_simTrkIdx.push_back(tpIndex);

    simhit_hitIdx.emplace_back();  // filled in matchCluster
    simhit_hitType.emplace_back(); // filled in matchCluster

    tpHitList.emplace_back(tpKey, simHitIndex, tof, simhit.detUnitId());
  }
}

void TrackingNtuple::fillPixelHits(const edm::Event& iEvent,
                                   const ClusterTPAssociation& clusterToTPMap,
                                   const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                   const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                                   const edm::DetSetVector<PixelDigiSimLink>& digiSimLink,
                                   const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                   const TrackerTopology& tTopo,
                                   const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                                   std::set<edm::ProductID>& hitProductIds
                                   ) {
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  iEvent.getByToken(pixelRecHitToken_, pixelHits);
  for (auto it = pixelHits->begin(); it!=pixelHits->end(); it++ ) {
    const DetId hitId = it->detId();
    for (auto hit = it->begin(); hit!=it->end(); hit++ ) {
      TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder.build(&*hit);

      hitProductIds.insert(hit->cluster().id());

      const int key = hit->cluster().key();
      const int lay = tTopo.layer(hitId);
      SimHitData simHitData = matchCluster(hit->firstClusterRef(), hitId, key, ttrh,
                                           clusterToTPMap, tpKeyToIndex, simHitsTPAssoc, digiSimLink, simHitRefKeyToIndex, HitType::Pixel);

      pix_isBarrel .push_back( hitId.subdetId()==1 );
      pix_det      .push_back( hitId.subdetId() );
      pix_lay      .push_back( lay );
      pix_detId    .push_back( hitId.rawId() );
      pix_trkIdx   .emplace_back(); // filled in fillTracks
      pix_seeIdx   .emplace_back(); // filled in fillSeeds
      pix_simHitIdx.push_back( simHitData.matchingSimHit );
      pix_simType.push_back( static_cast<int>(simHitData.type) );
      pix_x    .push_back( ttrh->globalPosition().x() );
      pix_y    .push_back( ttrh->globalPosition().y() );
      pix_z    .push_back( ttrh->globalPosition().z() );
      pix_xx   .push_back( ttrh->globalPositionError().cxx() );
      pix_xy   .push_back( ttrh->globalPositionError().cyx() );
      pix_yy   .push_back( ttrh->globalPositionError().cyy() );
      pix_yz   .push_back( ttrh->globalPositionError().czy() );
      pix_zz   .push_back( ttrh->globalPositionError().czz() );
      pix_zx   .push_back( ttrh->globalPositionError().czx() );
      pix_chargeFraction.push_back( simHitData.chargeFraction );
      pix_radL .push_back( ttrh->surface()->mediumProperties().radLen() );
      pix_bbxi .push_back( ttrh->surface()->mediumProperties().xi() );
      LogTrace("TrackingNtuple") << "pixHit cluster=" << key
                                 << " subdId=" << hitId.subdetId()
                                 << " lay=" << lay
                                 << " rawId=" << hitId.rawId()
                                 << " pos =" << ttrh->globalPosition()
                                 << " nMatchingSimHit=" << simHitData.matchingSimHit.size();
      if(!simHitData.matchingSimHit.empty()) {
        const auto simHitIdx = simHitData.matchingSimHit[0];
        LogTrace("TrackingNtuple") << " firstMatchingSimHit=" << simHitIdx
                                   << " simHitPos=" << GlobalPoint(simhit_x[simHitIdx], simhit_y[simHitIdx], simhit_z[simHitIdx])
                                   << " energyLoss=" << simhit_eloss[simHitIdx]
                                   << " particleType=" << simhit_particle[simHitIdx]
                                   << " processType=" << simhit_process[simHitIdx]
                                   << " bunchCrossing=" << simHitData.bunchCrossing[0]
                                   << " event=" << simHitData.event[0];
      }
    }
  }
}


void TrackingNtuple::fillStripRphiStereoHits(const edm::Event& iEvent,
                                             const ClusterTPAssociation& clusterToTPMap,
                                             const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                             const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                                             const edm::DetSetVector<StripDigiSimLink>& digiSimLink,
                                             const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                             const TrackerTopology& tTopo,
                                             const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                                             std::set<edm::ProductID>& hitProductIds
                                             ) {
  //index strip hit branches by cluster index
  edm::Handle<SiStripRecHit2DCollection> rphiHits;
  iEvent.getByToken(stripRphiRecHitToken_, rphiHits);
  edm::Handle<SiStripRecHit2DCollection> stereoHits;
  iEvent.getByToken(stripStereoRecHitToken_, stereoHits);
  int totalStripHits = rphiHits->dataSize()+stereoHits->dataSize();
  str_isBarrel .resize(totalStripHits);
  str_isStereo .resize(totalStripHits);
  str_det      .resize(totalStripHits);
  str_lay      .resize(totalStripHits);
  str_detId    .resize(totalStripHits);
  str_trkIdx   .resize(totalStripHits); // filled in fillTracks
  str_seeIdx   .resize(totalStripHits); // filled in fillSeeds
  str_simHitIdx.resize(totalStripHits);
  str_simType  .resize(totalStripHits);
  str_x    .resize(totalStripHits);
  str_y    .resize(totalStripHits);
  str_z    .resize(totalStripHits);
  str_xx   .resize(totalStripHits);
  str_xy   .resize(totalStripHits);
  str_yy   .resize(totalStripHits);
  str_yz   .resize(totalStripHits);
  str_zz   .resize(totalStripHits);
  str_zx   .resize(totalStripHits);
  str_chargeFraction.resize(totalStripHits);
  str_radL .resize(totalStripHits);
  str_bbxi .resize(totalStripHits);

  auto fill = [&](const SiStripRecHit2DCollection& hits, const char *name, bool isStereo) {
    for(const auto& detset: hits) {
      const DetId hitId = detset.detId();
      for(const auto& hit: detset) {
        TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder.build(&hit);

        hitProductIds.insert(hit.cluster().id());

        const int key = hit.cluster().key();
        const int lay = tTopo.layer(hitId);
        SimHitData simHitData = matchCluster(hit.firstClusterRef(), hitId, key, ttrh,
                                             clusterToTPMap, tpKeyToIndex, simHitsTPAssoc, digiSimLink, simHitRefKeyToIndex, HitType::Strip);
        str_isBarrel [key] = (hitId.subdetId()==StripSubdetector::TIB || hitId.subdetId()==StripSubdetector::TOB);
        str_isStereo [key] = isStereo;
        str_det      [key] = hitId.subdetId();
        str_lay      [key] = lay;
        str_detId    [key] = hitId.rawId();
        str_simHitIdx[key] = simHitData.matchingSimHit;
        str_simType  [key] = static_cast<int>(simHitData.type);
        str_x    [key] = ttrh->globalPosition().x();
        str_y    [key] = ttrh->globalPosition().y();
        str_z    [key] = ttrh->globalPosition().z();
        str_xx   [key] = ttrh->globalPositionError().cxx();
        str_xy   [key] = ttrh->globalPositionError().cyx();
        str_yy   [key] = ttrh->globalPositionError().cyy();
        str_yz   [key] = ttrh->globalPositionError().czy();
        str_zz   [key] = ttrh->globalPositionError().czz();
        str_zx   [key] = ttrh->globalPositionError().czx();
        str_chargeFraction[key] = simHitData.chargeFraction;
        str_radL [key] = ttrh->surface()->mediumProperties().radLen();
        str_bbxi [key] = ttrh->surface()->mediumProperties().xi();
        LogTrace("TrackingNtuple") << name << " cluster=" << key
                                   << " subdId=" << hitId.subdetId()
                                   << " lay=" << lay
                                   << " rawId=" << hitId.rawId()
                                   << " pos =" << ttrh->globalPosition()
                                   << " nMatchingSimHit=" << simHitData.matchingSimHit.size();
        if(!simHitData.matchingSimHit.empty()) {
          const auto simHitIdx = simHitData.matchingSimHit[0];
          LogTrace("TrackingNtuple") << " firstMatchingSimHit=" << simHitIdx
                                     << " simHitPos=" << GlobalPoint(simhit_x[simHitIdx], simhit_y[simHitIdx], simhit_z[simHitIdx])
                                     << " simHitPos=" << GlobalPoint(simhit_x[simHitIdx], simhit_y[simHitIdx], simhit_z[simHitIdx])
                                     << " energyLoss=" << simhit_eloss[simHitIdx]
                                     << " particleType=" << simhit_particle[simHitIdx]
                                     << " processType=" << simhit_process[simHitIdx]
                                     << " bunchCrossing=" << simHitData.bunchCrossing[0]
                                     << " event=" << simHitData.event[0];
        }
      }
    }
  };

  fill(*rphiHits, "stripRPhiHit", false);
  fill(*stereoHits, "stripStereoHit", true);
}

void TrackingNtuple::fillStripMatchedHits(const edm::Event& iEvent,
                                          const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                          const TrackerTopology& tTopo,
                                          std::vector<std::pair<int, int> >& monoStereoClusterList
                                          ) {
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
  iEvent.getByToken(stripMatchedRecHitToken_, matchedHits);
  for (auto it = matchedHits->begin(); it!=matchedHits->end(); it++ ) {
    const DetId hitId = it->detId();
    for (auto hit = it->begin(); hit!=it->end(); hit++ ) {
      TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder.build(&*hit);
      const int lay = tTopo.layer(hitId);
      monoStereoClusterList.emplace_back(hit->monoHit().cluster().key(),hit->stereoHit().cluster().key());
      glu_isBarrel .push_back( (hitId.subdetId()==StripSubdetector::TIB || hitId.subdetId()==StripSubdetector::TOB) );
      glu_det      .push_back( hitId.subdetId() );
      glu_lay      .push_back( tTopo.layer(hitId) );
      glu_detId    .push_back( hitId.rawId() );
      glu_monoIdx  .push_back( hit->monoHit().cluster().key() );
      glu_stereoIdx.push_back( hit->stereoHit().cluster().key() );
      glu_seeIdx   .emplace_back(); // filled in fillSeeds
      glu_x        .push_back( ttrh->globalPosition().x() );
      glu_y        .push_back( ttrh->globalPosition().y() );
      glu_z        .push_back( ttrh->globalPosition().z() );
      glu_xx       .push_back( ttrh->globalPositionError().cxx() );
      glu_xy       .push_back( ttrh->globalPositionError().cyx() );
      glu_yy       .push_back( ttrh->globalPositionError().cyy() );
      glu_yz       .push_back( ttrh->globalPositionError().czy() );
      glu_zz       .push_back( ttrh->globalPositionError().czz() );
      glu_zx       .push_back( ttrh->globalPositionError().czx() );
      glu_radL     .push_back( ttrh->surface()->mediumProperties().radLen() );
      glu_bbxi     .push_back( ttrh->surface()->mediumProperties().xi() );
      LogTrace("TrackingNtuple") << "stripMatchedHit"
                                 << " cluster0=" << hit->stereoHit().cluster().key()
                                 << " cluster1=" << hit->monoHit().cluster().key()
                                 << " subdId=" << hitId.subdetId()
                                 << " lay=" << lay
                                 << " rawId=" << hitId.rawId()
                                 << " pos =" << ttrh->globalPosition();
    }
  }
}

void TrackingNtuple::fillSeeds(const edm::Event& iEvent,
                               const TrackingParticleRefVector& tpCollection,
                               const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                               const reco::BeamSpot& bs,
                               const reco::TrackToTrackingParticleAssociator& associatorByHits,
                               const TransientTrackingRecHitBuilder& theTTRHBuilder,
                               const MagneticField *theMF,
                               const std::vector<std::pair<int, int> >& monoStereoClusterList,
                               const std::set<edm::ProductID>& hitProductIds,
                               std::map<edm::ProductID, size_t>& seedCollToOffset
                               ) {
  TSCBLBuilderNoMaterial tscblBuilder;
  for(const auto& seedToken: seedTokens_) {
    edm::Handle<edm::View<reco::Track> > seedTracksHandle;
    iEvent.getByToken(seedToken, seedTracksHandle);
    const auto& seedTracks = *seedTracksHandle;

    if(seedTracks.empty())
      continue;

    // The associator interfaces really need to be fixed...
    edm::RefToBaseVector<reco::Track> seedTrackRefs;
    for(edm::View<reco::Track>::size_type i=0; i<seedTracks.size(); ++i) {
      seedTrackRefs.push_back(seedTracks.refAt(i));
    }
    reco::RecoToSimCollection recSimColl = associatorByHits.associateRecoToSim(seedTrackRefs, tpCollection);

    edm::EDConsumerBase::Labels labels;
    labelsForToken(seedToken, labels);
    TString label = labels.module;
    //format label to match algoName
    label.ReplaceAll("seedTracks", "");
    label.ReplaceAll("Seeds","");
    label.ReplaceAll("muonSeeded","muonSeededStep");
    int algo = reco::TrackBase::algoByName(label.Data());

    edm::ProductID id = seedTracks[0].seedRef().id();
    auto inserted = seedCollToOffset.emplace(id, see_fitok.size());
    if(!inserted.second)
      throw cms::Exception("Configuration") << "Trying to add seeds with ProductID " << id << " for a second time from collection " << labels.module << ", seed algo " << label << ". Typically this is caused by a configuration problem.";
    see_offset.push_back(see_fitok.size());

    LogTrace("TrackingNtuple") << "NEW SEED LABEL: " << label << " size: " << seedTracks.size() << " algo=" << algo
                               << " ProductID " << id;

    for(const auto& seedTrackRef: seedTrackRefs) {
      const auto& seedTrack = *seedTrackRef;
      const auto& seedRef = seedTrack.seedRef();
      const auto& seed = *seedRef;

      if(seedRef.id() != id)
        throw cms::Exception("LogicError") << "All tracks in 'TracksFromSeeds' collection should point to seeds in the same collection. Now the element 0 had ProductID " << id << " while the element " << seedTrackRef.key() << " had " << seedTrackRef.id() << ". The source collection is " << labels.module << ".";

      std::vector<float> sharedFraction;
      std::vector<int> tpIdx;
      auto foundTPs = recSimColl.find(seedTrackRef);
      if (foundTPs != recSimColl.end()) {
        for(const auto tpQuality: foundTPs->val) {
          sharedFraction.push_back(tpQuality.second);
          tpIdx.push_back( tpKeyToIndex.at( tpQuality.first.key() ) );
        }
      }


      const bool seedFitOk = !trackFromSeedFitFailed(seedTrack);
      const int charge = seedTrack.charge();
      const float pt  = seedFitOk ? seedTrack.pt()  : 0;
      const float eta = seedFitOk ? seedTrack.eta() : 0;
      const float phi = seedFitOk ? seedTrack.phi() : 0;
      const int nHits = seedTrack.numberOfValidHits();

      const auto seedIndex = see_fitok.size();

      see_fitok   .push_back(seedFitOk);

      see_px      .push_back( seedFitOk ? seedTrack.px() : 0 );
      see_py      .push_back( seedFitOk ? seedTrack.py() : 0 );
      see_pz      .push_back( seedFitOk ? seedTrack.pz() : 0 );
      see_pt      .push_back( pt );
      see_eta     .push_back( eta );
      see_phi     .push_back( phi );
      see_q       .push_back( charge );
      see_nValid  .push_back( nHits );

      see_dxy     .push_back( seedFitOk ? seedTrack.dxy(bs.position()) : 0);
      see_dz      .push_back( seedFitOk ? seedTrack.dz(bs.position()) : 0);
      see_ptErr   .push_back( seedFitOk ? seedTrack.ptError() : 0);
      see_etaErr  .push_back( seedFitOk ? seedTrack.etaError() : 0);
      see_phiErr  .push_back( seedFitOk ? seedTrack.phiError() : 0);
      see_dxyErr  .push_back( seedFitOk ? seedTrack.dxyError() : 0);
      see_dzErr   .push_back( seedFitOk ? seedTrack.dzError() : 0);
      see_algo    .push_back( algo );

      see_trkIdx  .push_back(-1); // to be set correctly in fillTracks
      see_shareFrac.push_back( sharedFraction );
      see_simTrkIdx.push_back( tpIdx );

      /// Hmm, the following could make sense instead of plain failing if propagation to beam line fails
      /*
      TransientTrackingRecHit::RecHitPointer lastRecHit = theTTRHBuilder.build(&*(seed.recHits().second-1));
      TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( itSeed->startingState(), lastRecHit->surface(), theMF);
      float pt  = state.globalParameters().momentum().perp();
      float eta = state.globalParameters().momentum().eta();
      float phi = state.globalParameters().momentum().phi();
      see_px      .push_back( state.globalParameters().momentum().x() );
      see_py      .push_back( state.globalParameters().momentum().y() );
      see_pz      .push_back( state.globalParameters().momentum().z() );
      */

      std::vector<int> hitIdx;
      std::vector<int> hitType;

      for (auto hit=seed.recHits().first; hit!=seed.recHits().second; ++hit) {
	TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder.build(&*hit);
	int subid = recHit->geographicalId().subdetId();
	if (subid == (int) PixelSubdetector::PixelBarrel || subid == (int) PixelSubdetector::PixelEndcap) {
	  const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*recHit);
          const auto& clusterRef = bhit->firstClusterRef();
          const auto clusterKey = clusterRef.cluster_pixel().key();
          if(includeAllHits_) {
            checkProductID(hitProductIds, clusterRef.id(), "seed");
            pix_seeIdx[clusterKey].push_back(seedIndex);
          }
          hitIdx.push_back( clusterKey );
          hitType.push_back( static_cast<int>(HitType::Pixel) );
	} else {
	  if (trackerHitRTTI::isMatched(*recHit)) {
	    const SiStripMatchedRecHit2D * matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(&*recHit);
            if(includeAllHits_) {
              checkProductID(hitProductIds, matchedHit->monoClusterRef().id(), "seed");
              checkProductID(hitProductIds, matchedHit->stereoClusterRef().id(), "seed");
            }
	    int monoIdx = matchedHit->monoClusterRef().key();
	    int stereoIdx = matchedHit->stereoClusterRef().key();

            std::vector<std::pair<int,int> >::const_iterator pos = find( monoStereoClusterList.begin(), monoStereoClusterList.end(), std::make_pair(monoIdx,stereoIdx) );
            const auto gluedIndex = std::distance(monoStereoClusterList.begin(), pos);
            if(includeAllHits_) glu_seeIdx[gluedIndex].push_back(seedIndex);
            hitIdx.push_back( gluedIndex );
            hitType.push_back( static_cast<int>(HitType::Glued) );
	  } else {
	    const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*recHit);
            const auto& clusterRef = bhit->firstClusterRef();
            const auto clusterKey = clusterRef.cluster_strip().key();
            if(includeAllHits_) {
              checkProductID(hitProductIds, clusterRef.id(), "seed");
              str_seeIdx[clusterKey].push_back(seedIndex);
            }
            hitIdx.push_back( clusterKey );
            hitType.push_back( static_cast<int>(HitType::Strip) );
	  }
	}
      }
      see_hitIdx  .push_back( hitIdx  );
      see_hitType .push_back( hitType );
      see_nPixel  .push_back( std::count(hitType.begin(), hitType.end(), static_cast<int>(HitType::Pixel)) );
      see_nGlued  .push_back( std::count(hitType.begin(), hitType.end(), static_cast<int>(HitType::Glued)) );
      see_nStrip  .push_back( std::count(hitType.begin(), hitType.end(), static_cast<int>(HitType::Strip)) );
      //the part below is not strictly needed
      float chi2 = -1;
      if (nHits==2) {
	TransientTrackingRecHit::RecHitPointer recHit0 = theTTRHBuilder.build(&*(seed.recHits().first));
	TransientTrackingRecHit::RecHitPointer recHit1 = theTTRHBuilder.build(&*(seed.recHits().first+1));
        std::vector<GlobalPoint> gp(2);
        std::vector<GlobalError> ge(2);
	gp[0] = recHit0->globalPosition();
	ge[0] = recHit0->globalPositionError();
	gp[1] = recHit1->globalPosition();
	ge[1] = recHit1->globalPositionError();
        LogTrace("TrackingNtuple") << "seed " << seedTrackRef.key()
                                   << " pt=" << pt << " eta=" << eta << " phi=" << phi << " q=" << charge
                                   << " - PAIR - ids: " << recHit0->geographicalId().rawId() << " " << recHit1->geographicalId().rawId()
                                   << " hitpos: " << gp[0] << " " << gp[1]
                                   << " trans0: " << (recHit0->transientHits().size()>1 ? recHit0->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
                                   << " " << (recHit0->transientHits().size()>1 ? recHit0->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
                                   << " trans1: " << (recHit1->transientHits().size()>1 ? recHit1->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
                                   << " " << (recHit1->transientHits().size()>1 ? recHit1->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
                                   << " eta,phi: " << gp[0].eta() << "," << gp[0].phi();
      } else if (nHits==3) {
	TransientTrackingRecHit::RecHitPointer recHit0 = theTTRHBuilder.build(&*(seed.recHits().first));
	TransientTrackingRecHit::RecHitPointer recHit1 = theTTRHBuilder.build(&*(seed.recHits().first+1));
	TransientTrackingRecHit::RecHitPointer recHit2 = theTTRHBuilder.build(&*(seed.recHits().first+2));
	declareDynArray(GlobalPoint,4, gp);
	declareDynArray(GlobalError,4, ge);
	declareDynArray(bool,4, bl);
	gp[0] = recHit0->globalPosition();
	ge[0] = recHit0->globalPositionError();
	int subid0 = recHit0->geographicalId().subdetId();
	bl[0] = (subid0 == StripSubdetector::TIB || subid0 == StripSubdetector::TOB || subid0 == (int) PixelSubdetector::PixelBarrel);
	gp[1] = recHit1->globalPosition();
	ge[1] = recHit1->globalPositionError();
	int subid1 = recHit1->geographicalId().subdetId();
	bl[1] = (subid1 == StripSubdetector::TIB || subid1 == StripSubdetector::TOB || subid1 == (int) PixelSubdetector::PixelBarrel);
	gp[2] = recHit2->globalPosition();
	ge[2] = recHit2->globalPositionError();
	int subid2 = recHit2->geographicalId().subdetId();
	bl[2] = (subid2 == StripSubdetector::TIB || subid2 == StripSubdetector::TOB || subid2 == (int) PixelSubdetector::PixelBarrel);
	RZLine rzLine(gp,ge,bl);
	float seed_chi2 = rzLine.chi2();
	//float seed_pt = state.globalParameters().momentum().perp();
        float seed_pt = pt;
	LogTrace("TrackingNtuple") << "seed " << seedTrackRef.key()
                                   << " pt=" << pt << " eta=" << eta << " phi=" << phi << " q=" << charge
                                   << " - TRIPLET - ids: " << recHit0->geographicalId().rawId() << " " << recHit1->geographicalId().rawId() << " " << recHit2->geographicalId().rawId()
                                   << " hitpos: " << gp[0] << " " << gp[1] << " " << gp[2]
                                   << " trans0: " << (recHit0->transientHits().size()>1 ? recHit0->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
                                   << " " << (recHit0->transientHits().size()>1 ? recHit0->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
                                   << " trans1: " << (recHit1->transientHits().size()>1 ? recHit1->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
                                   << " " << (recHit1->transientHits().size()>1 ? recHit1->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
                                   << " trans2: " << (recHit2->transientHits().size()>1 ? recHit2->transientHits()[0]->globalPosition() : GlobalPoint(0,0,0))
                                   << " " << (recHit2->transientHits().size()>1 ? recHit2->transientHits()[1]->globalPosition() : GlobalPoint(0,0,0))
                                   << " local: " << recHit2->localPosition()
          //<< " tsos pos, mom: " << state.globalPosition()<<" "<<state.globalMomentum()
                                   << " eta,phi: " << gp[0].eta() << "," << gp[0].phi()
                                   << " pt,chi2: " << seed_pt << "," << seed_chi2;
	chi2 = seed_chi2;
      }
      see_chi2   .push_back( chi2 );
    }
  }
}

void TrackingNtuple::fillTracks(const edm::RefToBaseVector<reco::Track>& tracks,
                                const TrackingParticleRefVector& tpCollection,
                                const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                const reco::BeamSpot& bs,
                                const reco::TrackToTrackingParticleAssociator& associatorByHits,
                                const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                const TrackerTopology& tTopo,
                                const std::set<edm::ProductID>& hitProductIds,
                                const std::map<edm::ProductID, size_t>& seedCollToOffset
                                ) {
  reco::RecoToSimCollection recSimColl = associatorByHits.associateRecoToSim(tracks, tpCollection);
  edm::EDConsumerBase::Labels labels;
  labelsForToken(trackToken_, labels);
  LogTrace("TrackingNtuple") << "NEW TRACK LABEL: " << labels.module;
  for(size_t iTrack = 0; iTrack<tracks.size(); ++iTrack) {
    const auto& itTrack = tracks[iTrack];
    int nSimHits = 0;
    bool isSimMatched = false;
    std::vector<float> sharedFraction;
    std::vector<int> tpIdx;
    auto foundTPs = recSimColl.find(itTrack);
    if (foundTPs != recSimColl.end()) {
      if (!foundTPs->val.empty()) {
	nSimHits = foundTPs->val[0].first->numberOfTrackerHits();
	isSimMatched = true;
      }
      for(const auto tpQuality: foundTPs->val) {
        sharedFraction.push_back(tpQuality.second);
	tpIdx.push_back( tpKeyToIndex.at( tpQuality.first.key() ) );
      }
    }
    int charge = itTrack->charge();
    float pt = itTrack->pt();
    float eta = itTrack->eta();
    const double lambda = itTrack->lambda();
    float chi2 = itTrack->normalizedChi2();
    float phi = itTrack->phi();
    int nHits = itTrack->numberOfValidHits();
    const reco::HitPattern& hp = itTrack->hitPattern();
    trk_px       .push_back(itTrack->px());
    trk_py       .push_back(itTrack->py());
    trk_pz       .push_back(itTrack->pz());
    trk_pt       .push_back(pt);
    trk_inner_px.push_back(itTrack->innerMomentum().x());
    trk_inner_py.push_back(itTrack->innerMomentum().y());
    trk_inner_pz.push_back(itTrack->innerMomentum().z());
    trk_inner_pt.push_back(itTrack->innerMomentum().rho());
    trk_outer_px.push_back(itTrack->outerMomentum().x());
    trk_outer_py.push_back(itTrack->outerMomentum().y());
    trk_outer_pz.push_back(itTrack->outerMomentum().z());
    trk_outer_pt.push_back(itTrack->outerMomentum().rho());
    trk_eta      .push_back(eta);
    trk_lambda   .push_back(lambda);
    trk_cotTheta .push_back(1/tan(M_PI*0.5-lambda));
    trk_phi      .push_back(phi);
    trk_dxy      .push_back(itTrack->dxy(bs.position()));
    trk_dz       .push_back(itTrack->dz(bs.position()));
    trk_ptErr    .push_back(itTrack->ptError());
    trk_etaErr   .push_back(itTrack->etaError());
    trk_lambdaErr.push_back(itTrack->lambdaError());
    trk_phiErr   .push_back(itTrack->phiError());
    trk_dxyErr   .push_back(itTrack->dxyError());
    trk_dzErr    .push_back(itTrack->dzError());
    trk_refpoint_x.push_back(itTrack->vx());
    trk_refpoint_y.push_back(itTrack->vy());
    trk_refpoint_z.push_back(itTrack->vz());
    trk_nChi2    .push_back( itTrack->normalizedChi2());
    trk_shareFrac.push_back(sharedFraction);
    trk_q        .push_back(charge);
    trk_nValid   .push_back(hp.numberOfValidHits());
    trk_nInvalid .push_back(hp.numberOfLostHits(reco::HitPattern::TRACK_HITS));
    trk_nPixel   .push_back(hp.numberOfValidPixelHits());
    trk_nStrip   .push_back(hp.numberOfValidStripHits());
    trk_nPixelLay.push_back(hp.pixelLayersWithMeasurement());
    trk_nStripLay.push_back(hp.stripLayersWithMeasurement());
    trk_n3DLay   .push_back(hp.numberOfValidStripLayersWithMonoAndStereo()+hp.pixelLayersWithMeasurement());
    trk_nOuterLost.push_back(hp.numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS));
    trk_nInnerLost.push_back(hp.numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS));
    trk_algo     .push_back(itTrack->algo());
    trk_originalAlgo.push_back(itTrack->originalAlgo());
    trk_algoMask .push_back(itTrack->algoMaskUL());
    trk_stopReason.push_back(itTrack->stopReason());
    trk_isHP     .push_back(itTrack->quality(reco::TrackBase::highPurity));
    if(includeSeeds_) {
      auto offset = seedCollToOffset.find(itTrack->seedRef().id());
      if(offset == seedCollToOffset.end()) {
        throw cms::Exception("Configuration") << "Track algo '" << reco::TrackBase::algoName(itTrack->algo())
                                              << "' originalAlgo '" << reco::TrackBase::algoName(itTrack->originalAlgo())
                                              << "' refers to seed collection " << itTrack->seedRef().id()
                                              << ", but that seed collection is not given as an input. The following collections were given as an input " << make_ProductIDMapPrinter(seedCollToOffset);
      }

      const auto seedIndex = offset->second + itTrack->seedRef().key();
      trk_seedIdx  .push_back(seedIndex);
      if(see_trkIdx[seedIndex] != -1) {
        throw cms::Exception("LogicError") << "Track index has already been set for seed " << seedIndex << " to " << see_trkIdx[seedIndex] << "; was trying to set it to " << iTrack;
      }
      see_trkIdx[seedIndex] = iTrack;
    }
    trk_vtxIdx   .push_back(-1); // to be set correctly in fillVertices
    trk_simTrkIdx.push_back(tpIdx);
    LogTrace("TrackingNtuple") << "Track #" << itTrack.key() << " with q=" << charge
                               << ", pT=" << pt << " GeV, eta: " << eta << ", phi: " << phi
                               << ", chi2=" << chi2
                               << ", Nhits=" << nHits
                               << ", algo=" << itTrack->algoName(itTrack->algo()).c_str()
                               << " hp=" << itTrack->quality(reco::TrackBase::highPurity)
                               << " seed#=" << itTrack->seedRef().key()
                               << " simMatch=" << isSimMatched
                               << " nSimHits=" << nSimHits
                               << " sharedFraction=" << (sharedFraction.empty()?-1:sharedFraction[0])
                               << " tpIdx=" << (tpIdx.empty()?-1:tpIdx[0]);
    std::vector<int> hitIdx;
    std::vector<int> hitType;

    for(auto i=itTrack->recHitsBegin(); i!=itTrack->recHitsEnd(); i++) {
      TransientTrackingRecHit::RecHitPointer hit=theTTRHBuilder.build(&**i );
      DetId hitId = hit->geographicalId();
      LogTrace("TrackingNtuple") << "hit #" << std::distance(itTrack->recHitsBegin(), i) << " subdet=" << hitId.subdetId();
      if(hitId.det() != DetId::Tracker)
        continue;

      LogTrace("TrackingNtuple") << " " << subdetstring(hitId.subdetId()) << " " << tTopo.layer(hitId);
      bool isPixel = (hitId.subdetId() == (int) PixelSubdetector::PixelBarrel || hitId.subdetId() == (int) PixelSubdetector::PixelEndcap );

      if (hit->isValid()) {
        //ugly... but works
        const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*hit);
        const auto& clusterRef = bhit->firstClusterRef();
        const auto clusterKey = clusterRef.isPixel() ? clusterRef.cluster_pixel().key() :  clusterRef.cluster_strip().key();

        LogTrace("TrackingNtuple") << " id: " << hitId.rawId() << " - globalPos =" << hit->globalPosition()
                                   << " cluster=" << clusterKey
                                   << " eta,phi: " << hit->globalPosition().eta() << "," << hit->globalPosition().phi();
        if(includeAllHits_) {
          checkProductID(hitProductIds, clusterRef.id(), "track");
          if(isPixel) pix_trkIdx[clusterKey].push_back(iTrack);
          else        str_trkIdx[clusterKey].push_back(iTrack);
        }

        hitIdx.push_back(clusterKey);
        hitType.push_back( static_cast<int>( isPixel ? HitType::Pixel : HitType::Strip ) );
      } else  {
        LogTrace("TrackingNtuple") << " - invalid hit";

        hitIdx.push_back( inv_isBarrel.size() );
        hitType.push_back( static_cast<int>(HitType::Invalid) );

        inv_isBarrel.push_back( hitId.subdetId()==1 );
        inv_det     .push_back( hitId.subdetId() );
        inv_lay     .push_back( tTopo.layer(hitId) );
        inv_detId   .push_back( hitId.rawId() );
        inv_type    .push_back( hit->getType() );

      }
    }

    trk_hitIdx.push_back(hitIdx);
    trk_hitType.push_back(hitType);
  }
}


void TrackingNtuple::fillTrackingParticles(const edm::Event& iEvent, const edm::EventSetup& iSetup,
                                           const edm::RefToBaseVector<reco::Track>& tracks,
                                           const TrackingParticleRefVector& tpCollection,
                                           const TrackingVertexRefKeyToIndex& tvKeyToIndex,
                                           const reco::TrackToTrackingParticleAssociator& associatorByHits,
                                           const std::vector<TPHitIndex>& tpHitList
                                           ) {
  edm::ESHandle<ParametersDefinerForTP> parametersDefinerH;
  iSetup.get<TrackAssociatorRecord>().get(parametersDefinerName_, parametersDefinerH);
  const ParametersDefinerForTP *parametersDefiner = parametersDefinerH.product();

  // Number of 3D layers for TPs
  edm::Handle<edm::ValueMap<unsigned int>> tpNLayersH;
  iEvent.getByToken(tpNLayersToken_, tpNLayersH);
  const auto& nLayers_tPCeff = *tpNLayersH;

  iEvent.getByToken(tpNPixelLayersToken_, tpNLayersH);
  const auto& nPixelLayers_tPCeff = *tpNLayersH;

  iEvent.getByToken(tpNStripStereoLayersToken_, tpNLayersH);
  const auto& nStripMonoAndStereoLayers_tPCeff = *tpNLayersH;

  reco::SimToRecoCollection simRecColl = associatorByHits.associateSimToReco(tracks, tpCollection);

  for(const TrackingParticleRef& tp: tpCollection) {
    LogTrace("TrackingNtuple") << "tracking particle pt=" << tp->pt() << " eta=" << tp->eta() << " phi=" << tp->phi();
    bool isRecoMatched = false;
    std::vector<int> tkIdx;
    std::vector<float> sharedFraction;
    auto foundTracks = simRecColl.find(tp);
    if(foundTracks != simRecColl.end()) {
      isRecoMatched = true;
      for(const auto trackQuality: foundTracks->val) {
        sharedFraction.push_back(trackQuality.second);
        tkIdx.push_back(trackQuality.first.key());
      }
    }
    LogTrace("TrackingNtuple") << "matched to tracks = " << make_VectorPrinter(tkIdx) << " isRecoMatched=" << isRecoMatched;
    sim_event    .push_back(tp->eventId().event());
    sim_bunchCrossing.push_back(tp->eventId().bunchCrossing());
    sim_pdgId    .push_back(tp->pdgId());
    sim_px       .push_back(tp->px());
    sim_py       .push_back(tp->py());
    sim_pz       .push_back(tp->pz());
    sim_pt       .push_back(tp->pt());
    sim_eta      .push_back(tp->eta());
    sim_phi      .push_back(tp->phi());
    sim_q        .push_back(tp->charge());
    sim_trkIdx   .push_back(tkIdx);
    sim_shareFrac.push_back(sharedFraction);
    sim_parentVtxIdx.push_back( tvKeyToIndex.at(tp->parentVertex().key()) );
    std::vector<int> decayIdx;
    for(const auto& v: tp->decayVertices())
      decayIdx.push_back( tvKeyToIndex.at(v.key()) );
    sim_decayVtxIdx.push_back(decayIdx);

    //Calcualte the impact parameters w.r.t. PCA
    TrackingParticle::Vector momentum = parametersDefiner->momentum(iEvent,iSetup,tp);
    TrackingParticle::Point vertex = parametersDefiner->vertex(iEvent,iSetup,tp);
    float dxySim = (-vertex.x()*sin(momentum.phi())+vertex.y()*cos(momentum.phi()));
    float dzSim = vertex.z() - (vertex.x()*momentum.x()+vertex.y()*momentum.y())/sqrt(momentum.perp2())
      * momentum.z()/sqrt(momentum.perp2());
    const double lambdaSim = M_PI/2 - momentum.theta();
    sim_pca_pt       .push_back(std::sqrt(momentum.perp2()));
    sim_pca_eta      .push_back(momentum.Eta());
    sim_pca_lambda   .push_back(lambdaSim);
    sim_pca_cotTheta .push_back(1/tan(M_PI*0.5-lambdaSim));
    sim_pca_phi      .push_back(momentum.phi());
    sim_pca_dxy      .push_back(dxySim);
    sim_pca_dz       .push_back(dzSim);

    std::vector<int> hitIdx;
    int nPixel=0, nStrip=0;
    auto rangeHit = std::equal_range(tpHitList.begin(), tpHitList.end(), TPHitIndex(tp.key()), tpHitIndexListLess);
    for(auto ip = rangeHit.first; ip != rangeHit.second; ++ip) {
      auto type = HitType::Unknown;
      if(!simhit_hitType[ip->simHitIdx].empty())
        type = static_cast<HitType>(simhit_hitType[ip->simHitIdx][0]);
      LogTrace("TrackingNtuple") << "simhit=" << ip->simHitIdx << " type=" << static_cast<int>(type);
      hitIdx.push_back(ip->simHitIdx);
      const auto detid = DetId(simhit_detId[ip->simHitIdx]);
      if(detid.det() != DetId::Tracker) {
        throw cms::Exception("LogicError") << "Encountered SimHit for TP " << tp.key() << " with DetId " << detid.rawId() << " whose det() is not Tracker but " << detid.det();
      }
      const auto subdet = detid.subdetId();
      switch(subdet) {
      case PixelSubdetector::PixelBarrel:
      case PixelSubdetector::PixelEndcap:
        ++nPixel;
        break;
      case StripSubdetector::TIB:
      case StripSubdetector::TID:
      case StripSubdetector::TOB:
      case StripSubdetector::TEC:
        ++nStrip;
        break;
      default:
        throw cms::Exception("LogicError") << "Encountered SimHit for TP " << tp.key() << " with DetId " << detid.rawId() << " whose subdet is not recognized, is " << subdet;
      };
    }
    sim_nValid   .push_back( hitIdx.size() );
    sim_nPixel   .push_back( nPixel );
    sim_nStrip   .push_back( nStrip );

    const auto nSimLayers = nLayers_tPCeff[tp];
    const auto nSimPixelLayers = nPixelLayers_tPCeff[tp];
    const auto nSimStripMonoAndStereoLayers = nStripMonoAndStereoLayers_tPCeff[tp];
    sim_nLay     .push_back( nSimLayers );
    sim_nPixelLay.push_back( nSimPixelLayers );
    sim_n3DLay   .push_back( nSimPixelLayers+nSimStripMonoAndStereoLayers );

    sim_simHitIdx.push_back(hitIdx);
  }
}

void TrackingNtuple::fillVertices(const reco::VertexCollection& vertices,
                                  const edm::RefToBaseVector<reco::Track>& tracks) {
  for(size_t iVertex=0, size=vertices.size(); iVertex<size; ++iVertex) {
    const reco::Vertex& vertex = vertices[iVertex];
    vtx_x.push_back(vertex.x());
    vtx_y.push_back(vertex.y());
    vtx_z.push_back(vertex.z());
    vtx_xErr.push_back(vertex.xError());
    vtx_yErr.push_back(vertex.yError());
    vtx_zErr.push_back(vertex.zError());
    vtx_chi2.push_back(vertex.chi2());
    vtx_ndof.push_back(vertex.ndof());
    vtx_fake.push_back(vertex.isFake());
    vtx_valid.push_back(vertex.isValid());

    std::vector<int> trkIdx;
    for(auto iTrack = vertex.tracks_begin(); iTrack != vertex.tracks_end(); ++iTrack) {
      // Ignore link if vertex was fit from a track collection different from the input
      if(iTrack->id() != tracks.id())
        continue;

      trkIdx.push_back(iTrack->key());

      if(trk_vtxIdx[iTrack->key()] != -1) {
        throw cms::Exception("LogicError") << "Vertex index has already been set for track " << iTrack->key() << " to " << trk_vtxIdx[iTrack->key()] << "; was trying to set it to " << iVertex;
      }
      trk_vtxIdx[iTrack->key()] = iVertex;
    }
    vtx_trkIdx.push_back(trkIdx);
  }
}

void TrackingNtuple::fillTrackingVertices(const TrackingVertexRefVector& trackingVertices,
                                          const TrackingParticleRefKeyToIndex& tpKeyToIndex
                                          ) {
  int current_event = -1;
  for(const auto& ref: trackingVertices) {
    const TrackingVertex v = *ref;
    if(v.eventId().event() != current_event) {
      // next PV
      current_event = v.eventId().event();
      simpv_idx.push_back(simvtx_x.size());
    }

    unsigned int processType = std::numeric_limits<unsigned int>::max();
    if(!v.g4Vertices().empty()) {
      processType = v.g4Vertices()[0].processType();
    }

    simvtx_event.push_back(v.eventId().event());
    simvtx_bunchCrossing.push_back(v.eventId().bunchCrossing());
    simvtx_processType.push_back(processType);

    simvtx_x.push_back(v.position().x());
    simvtx_y.push_back(v.position().y());
    simvtx_z.push_back(v.position().z());

    auto fill = [&](const TrackingParticleRefVector& tps, std::vector<int>& idx) {
      for(const auto& tpRef: tps) {
        auto found = tpKeyToIndex.find(tpRef.key());
        if(found != tpKeyToIndex.end()) {
          idx.push_back(tpRef.key());
        }
      }
    };

    std::vector<int> sourceIdx;
    std::vector<int> daughterIdx;
    fill(v.sourceTracks(), sourceIdx);
    fill(v.daughterTracks(), daughterIdx);

    simvtx_sourceSimIdx.push_back(sourceIdx);
    simvtx_daughterSimIdx.push_back(daughterIdx);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackingNtuple::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<edm::InputTag> >("seedTracks", std::vector<edm::InputTag>{
      edm::InputTag("seedTracksinitialStepSeeds"),
      edm::InputTag("seedTracksdetachedTripletStepSeeds"),
      edm::InputTag("seedTrackspixelPairStepSeeds"),
      edm::InputTag("seedTrackslowPtTripletStepSeeds"),
      edm::InputTag("seedTracksmixedTripletStepSeeds"),
      edm::InputTag("seedTrackspixelLessStepSeeds"),
      edm::InputTag("seedTrackstobTecStepSeeds"),
      edm::InputTag("seedTracksjetCoreRegionalStepSeeds"),
      edm::InputTag("seedTracksmuonSeededSeedsInOut"),
      edm::InputTag("seedTracksmuonSeededSeedsOutIn")
  });
  desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.addUntracked<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.addUntracked<bool>("trackingParticlesRef", false);
  desc.addUntracked<edm::InputTag>("clusterTPMap", edm::InputTag("tpClusterProducer"));
  desc.addUntracked<edm::InputTag>("simHitTPMap", edm::InputTag("simHitTPAssocProducer"));
  desc.addUntracked<edm::InputTag>("trackAssociator", edm::InputTag("quickTrackAssociatorByHits"));
  desc.addUntracked<edm::InputTag>("pixelDigiSimLink", edm::InputTag("simSiPixelDigis"));
  desc.addUntracked<edm::InputTag>("stripDigiSimLink", edm::InputTag("simSiStripDigis"));
  desc.addUntracked<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<edm::InputTag>("pixelRecHits", edm::InputTag("siPixelRecHits"));
  desc.addUntracked<edm::InputTag>("stripRphiRecHits", edm::InputTag("siStripMatchedRecHits", "rphiRecHit"));
  desc.addUntracked<edm::InputTag>("stripStereoRecHits", edm::InputTag("siStripMatchedRecHits", "stereoRecHit"));
  desc.addUntracked<edm::InputTag>("stripMatchedRecHits", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));
  desc.addUntracked<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.addUntracked<edm::InputTag>("trackingVertices", edm::InputTag("mix", "MergedTrackTruth"));
  desc.addUntracked<edm::InputTag>("trackingParticleNlayers", edm::InputTag("trackingParticleNumberOfLayersProducer", "trackerLayers"));
  desc.addUntracked<edm::InputTag>("trackingParticleNpixellayers", edm::InputTag("trackingParticleNumberOfLayersProducer", "pixelLayers"));
  desc.addUntracked<edm::InputTag>("trackingParticleNstripstereolayers", edm::InputTag("trackingParticleNumberOfLayersProducer", "stripStereoLayers"));
  desc.addUntracked<std::string>("TTRHBuilder", "WithTrackAngle");
  desc.addUntracked<std::string>("parametersDefiner", "LhcParametersDefinerForTP");
  desc.addUntracked<bool>("includeSeeds", false);
  desc.addUntracked<bool>("includeAllHits", false);
  descriptions.add("trackingNtuple",desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackingNtuple);
