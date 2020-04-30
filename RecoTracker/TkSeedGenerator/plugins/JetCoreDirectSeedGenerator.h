#ifndef RecoTracker_TkSeedGenerator_JetCoreDirectSeedGenerator_H
#define RecoTracker_TkSeedGenerator_JetCoreDirectSeedGenerator_H

#define jetDimX 30
#define jetDimY 30
#define Nlayer 4
#define Nover 3
#define Npar 5


#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"


#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"



#include <boost/range.hpp>
#include <boost/foreach.hpp>
#include "boost/multi_array.hpp"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// #include "SimG4Core/Application/interface/G4SimTrack.h"
// #include "SimDataFormats/Track/interface/SimTrack.h"

#include "SimDataFormats/Vertex/interface/SimVertex.h"


#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"



#include "TTree.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"


namespace edm { class Event; class EventSetup; }


class JetCoreDirectSeedGenerator : public edm::one::EDProducer<edm::one::SharedResources>  {
   public:
      explicit JetCoreDirectSeedGenerator(const edm::ParameterSet&);
      ~JetCoreDirectSeedGenerator();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  // A pointer to a cluster and a list of tracks on it
  struct TrackAndState
  {
    TrackAndState(const reco::Track *aTrack, TrajectoryStateOnSurface aState) :
      track(aTrack), state(aState) {}
    const reco::Track*      track;
    TrajectoryStateOnSurface state;
  };


  template<typename Cluster>
  struct ClusterWithTracks
  {
    ClusterWithTracks(const Cluster &c) : cluster(&c) {}
    const Cluster* cluster;
    std::vector<TrackAndState> tracks;
  };

  typedef ClusterWithTracks<SiPixelCluster> SiPixelClusterWithTracks;

  typedef boost::sub_range<std::vector<SiPixelClusterWithTracks> > SiPixelClustersWithTracks;

  TFile* JetCoreDirectSeedGenerator_out;
  TTree* JetCoreDirectSeedGeneratorTree;
  // static const int jetDimX =30;
  // static const int jetDimY =30;
  // static const int Nlayer =4;
  // static const int Nover = 3;
  // static const int Npar = 4;

  // double clusterMeas[jetDimX][jetDimY][Nlayer];
  double jet_pt;
  double jet_eta;
  double pitchX = 0.01;
  double pitchY = 0.015;
  bool print = false;
  int evt_counter =0;


   private:
      virtual void beginJob() override;
      virtual void produce( edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;


      // ----------member data ---------------------------
  std::string propagatorName_;
  edm::ESHandle<MagneticField>          magfield_;
  edm::ESHandle<GlobalTrackingGeometry> geometry_;
  edm::ESHandle<Propagator>             propagator_;

  edm::EDGetTokenT<std::vector<reco::Vertex> > vertices_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
  std::vector<SiPixelClusterWithTracks> allSiPixelClusters;
  std::map<uint32_t, SiPixelClustersWithTracks> siPixelDetsWithClusters;
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> > pixeldigisimlink;
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClusters;
  edm::EDGetTokenT< edm::DetSetVector<PixelDigiSimLink> > pixeldigisimlinkToken;
  edm::EDGetTokenT<edm::View<reco::Candidate> > cores_;
  // edm::EDGetTokenT<std::vector<SimTrack> > simtracksToken;
  // edm::EDGetTokenT<std::vector<SimVertex> > simvertexToken;

  double ptMin_;
  double deltaR_;
  double chargeFracMin_;
  double centralMIPCharge_;

  std::string pixelCPE_;
  std::string weightfilename_;
  std::vector<std::string> inputTensorName_;
  std::vector<std::string> outputTensorName_;
  size_t nThreads;
  std::string singleThreadPool;

  double probThr;


  tensorflow::GraphDef* graph_;
  tensorflow::Session* session_;



  std::pair<bool, Basic3DVector<float>> findIntersection(const GlobalVector & , const reco::Candidate::Point & ,const GeomDet*);

  void fillPixelMatrix(const SiPixelCluster &, int, auto, const GeomDet*, tensorflow::NamedTensorList);

  std::pair<int,int> local2Pixel(double, double, const GeomDet*);

  LocalPoint pixel2Local(int, int, const GeomDet*);

  int pixelFlipper(const GeomDet*);

  const GeomDet* DetectorSelector(int ,const reco::Candidate& jet, GlobalVector,  const reco::Vertex& jetVertex, const TrackerTopology* const);

  std::vector<GlobalVector> splittedClusterDirectionsOld(const reco::Candidate&, const TrackerTopology* const, auto pp, const reco::Vertex& jetVertex );
  std::vector<GlobalVector> splittedClusterDirections(const reco::Candidate&, const TrackerTopology* const, auto pp, const reco::Vertex& jetVertex, int );
  
  std::pair<double[jetDimX][jetDimY][Nover][Npar],double[jetDimX][jetDimY][Nover]> SeedEvaluation(tensorflow::NamedTensorList);



};
#endif
