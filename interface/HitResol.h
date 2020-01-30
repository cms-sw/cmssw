#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h" 
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "TROOT.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include <vector>
#include "TTree.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "Riostream.h"
#include "TRandom2.h"

class TrackerTopology;

class HitResol : public edm::EDAnalyzer {
 public:  
  explicit HitResol(const edm::ParameterSet& conf);
  double checkConsistency(const StripClusterParameterEstimator::LocalValues& parameters, double xx, double xerr);
  bool isDoubleSided(unsigned int iidd, const TrackerTopology* tTopo) const;
  bool check2DPartner(unsigned int iidd, const std::vector<TrajectoryMeasurement>& traj);
  ~HitResol() override;
  unsigned int checkLayer(unsigned int iidd, const TrackerTopology* tTopo);
//  float getSimHitRes(const GeomDetUnit * det, const LocalVector& trackdirection, const TrackingRecHit& recHit, float & trackWidth,   float* pitch, LocalVector& drift);
  void getSimHitRes(const GeomDetUnit * det, const LocalVector& trackdirection, const TrackingRecHit& recHit, float & trackWidth,   float* pitch, LocalVector& drift);
  double getSimpleRes(const TrajectoryMeasurement* traj1);
  bool getPairParameters(const MagneticField* magField_, AnalyticalPropagator& propagator,const TrajectoryMeasurement* traj1, const TrajectoryMeasurement* traj2, float & pairPath, float & hitDX, float & trackDX,  float & trackDXE, float & trackParamX, float &trackParamY , float & trackParamDXDZ, float &trackParamDYDZ , float & trackParamXE, float &trackParamYE, float & trackParamDXDZE, float &trackParamDYDZE);
  typedef std::vector<Trajectory> TrajectoryCollection;

 private:
  void beginJob() override;
  void endJob() override; 
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

        // ----------member data ---------------------------

  const edm::EDGetTokenT<LumiScalersCollection> scalerToken_;
  const edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > commonModeToken_;
  
  bool addLumi_;
  bool addCommonMode_;
  bool cutOnTracks_;
  unsigned int trackMultiplicityCut_;
  bool useFirstMeas_;
  bool useLastMeas_;
  bool useAllHitsFromTracksWithMissingHits_; 
  double MomentumCut_; 
  unsigned int UsePairsOnly_;
  
  const edm::EDGetTokenT< reco::TrackCollection > combinatorialTracks_token_;
  const edm::EDGetTokenT< std::vector<Trajectory> > trajectories_token_;
  const edm::EDGetTokenT< TrajTrackAssociationCollection > trajTrackAsso_token_;
  const edm::EDGetTokenT< std::vector<Trajectory> > tjToken_;
  const edm::EDGetTokenT<reco::TrackCollection> tkToken_;
  const edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  const edm::EDGetTokenT<DetIdCollection> digis_token_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> trackerEvent_token_;

  edm::ParameterSet conf_;
  
  TTree* reso;
  TTree* treso;

  int events,EventTrackCKF;
  
  int compSettings;
  unsigned int layers;
  bool DEBUG;
  unsigned int whatlayer;
  
  double  MomentumCut;
  
  
// Tree declarations
//Hit Resolution Ntuple Content
  float        mymom            ;
  int          numHits          ;
  float        ProbTrackChi2    ;
  unsigned int iidd1            ;
  float        mypitch1         ;
  unsigned int clusterWidth     ;
  float        expWidth         ;
  float        atEdge           ;
  float        simpleRes        ;
  unsigned int iidd2            ;
  unsigned int clusterWidth_2   ;
  float        expWidth_2       ;
  float        atEdge_2         ;
  
  float        pairPath 	;
  float        hitDX		;
  float        trackDX  	;
  float        trackDXE 	;
  float        trackParamX	;
  float        trackParamY	;
  float        trackParamDXDZ	;
  float        trackParamDYDZ	;
  float        trackParamXE	;
  float        trackParamYE	;
  float        trackParamDXDZE  ;
  float        trackParamDYDZE  ;
  unsigned int pairsOnly        ;
  float        track_momentum   ;
  float        track_eta        ;
  float        track_trackChi2  ;

};


//#endif
