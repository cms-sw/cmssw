#ifndef RecoTracker_SingleTrackPattern_AnalyzeMTCCTracks_h
#define RecoTracker_SingleTrackPattern_AnalyzeMTCCTracks_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include <TROOT.h>
#include <TFile.h>
#include <TH1F.h>

class AnalyzeMTCCTracks : public edm::EDAnalyzer
{
typedef TrajectoryStateOnSurface     TSOS;
 public:
  
  explicit AnalyzeMTCCTracks(const edm::ParameterSet& conf);
  
  virtual ~AnalyzeMTCCTracks();
  virtual void beginRun(edm::Run & run, const edm::EventSetup& c);
  virtual void endJob(); 
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  void makeResiduals(const Trajectory traj);
  void makeResiduals(const TrajectorySeed& seed,
		     const TrackingRecHitCollection &hits,
		     const edm::Event& e, 
		     const edm::EventSetup& es);
  void AnalHits(const TrackingRecHitCollection &hits);
  std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;
  TrajectoryStateOnSurface startingTSOS(const TrajectorySeed& seed)const;
  Trajectory createStartingTrajectory( const TrajectorySeed& seed) const;
 private:
  edm::ParameterSet conf_;
  bool ltib1,ltib2,ltob1,ltob2;
  unsigned int nlay;
  TFile* hFile;
  TH1F  *hphi, *hnhit,*hchi,*hresTOB,*hresTIB,*hresTOB1,*hresTIB1,*hresTOB2,*hresTIB2,*heta,*hpt,*hq,*hpx,*hpy,*hpz,*hresTOB_4l,*hresTIB_4l,*hresTOB1_4l,*hresTIB1_4l,*hresTOB2_4l,*hresTIB2_4l;
  TH1F *hresTIBL1_int_str1_mod1_ste,*hresTIBL1_int_str1_mod2_ste,*hresTIBL1_int_str1_mod3_ste;
  TH1F *hresTIBL1_int_str2_mod1_ste,*hresTIBL1_int_str2_mod2_ste,*hresTIBL1_int_str2_mod3_ste;
  TH1F *hresTIBL1_est_str1_mod1_ste,*hresTIBL1_est_str1_mod2_ste,*hresTIBL1_est_str1_mod3_ste;
  TH1F *hresTIBL1_est_str2_mod1_ste,*hresTIBL1_est_str2_mod2_ste,*hresTIBL1_est_str2_mod3_ste;
  TH1F *hresTIBL1_est_str3_mod1_ste,*hresTIBL1_est_str3_mod2_ste,*hresTIBL1_est_str3_mod3_ste;
  TH1F *hresTIBL1_int_str1_mod1_rphi,*hresTIBL1_int_str1_mod2_rphi,*hresTIBL1_int_str1_mod3_rphi;
  TH1F *hresTIBL1_int_str2_mod1_rphi,*hresTIBL1_int_str2_mod2_rphi,*hresTIBL1_int_str2_mod3_rphi;
  TH1F *hresTIBL1_est_str1_mod1_rphi,*hresTIBL1_est_str1_mod2_rphi,*hresTIBL1_est_str1_mod3_rphi;
  TH1F *hresTIBL1_est_str2_mod1_rphi,*hresTIBL1_est_str2_mod2_rphi,*hresTIBL1_est_str2_mod3_rphi;
  TH1F *hresTIBL1_est_str3_mod1_rphi,*hresTIBL1_est_str3_mod2_rphi,*hresTIBL1_est_str3_mod3_rphi;
  TH1F *hresTIBL2_int_str1_mod1,*hresTIBL2_int_str1_mod2,*hresTIBL2_int_str1_mod3;
  TH1F *hresTIBL2_int_str2_mod1,*hresTIBL2_int_str2_mod2,*hresTIBL2_int_str2_mod3;
  TH1F *hresTIBL2_int_str3_mod1,*hresTIBL2_int_str3_mod2,*hresTIBL2_int_str3_mod3;
  TH1F *hresTIBL2_int_str4_mod1,*hresTIBL2_int_str4_mod2,*hresTIBL2_int_str4_mod3;
  TH1F *hresTIBL2_int_str5_mod1,*hresTIBL2_int_str5_mod2,*hresTIBL2_int_str5_mod3;
  TH1F *hresTIBL2_int_str6_mod1,*hresTIBL2_int_str6_mod2,*hresTIBL2_int_str6_mod3;
  TH1F *hresTIBL2_int_str7_mod1,*hresTIBL2_int_str7_mod2,*hresTIBL2_int_str7_mod3;
  TH1F *hresTIBL2_int_str8_mod1,*hresTIBL2_int_str8_mod2,*hresTIBL2_int_str8_mod3;
  TH1F *hresTIBL2_est_str1_mod1,*hresTIBL2_est_str1_mod2,*hresTIBL2_est_str1_mod3;
  TH1F *hresTIBL2_est_str2_mod1,*hresTIBL2_est_str2_mod2,*hresTIBL2_est_str2_mod3;
  TH1F *hresTIBL2_est_str3_mod1,*hresTIBL2_est_str3_mod2,*hresTIBL2_est_str3_mod3;
  TH1F *hresTIBL2_est_str4_mod1,*hresTIBL2_est_str4_mod2,*hresTIBL2_est_str4_mod3;
  TH1F *hresTIBL2_est_str5_mod1,*hresTIBL2_est_str5_mod2,*hresTIBL2_est_str5_mod3;
  TH1F *hresTIBL2_est_str6_mod1,*hresTIBL2_est_str6_mod2,*hresTIBL2_est_str6_mod3;
  TH1F *hresTIBL2_est_str7_mod1,*hresTIBL2_est_str7_mod2,*hresTIBL2_est_str7_mod3;
  TH1F *hresTOBL1_rod1_mod1,*hresTOBL1_rod1_mod2,*hresTOBL1_rod1_mod3,*hresTOBL1_rod1_mod4,*hresTOBL1_rod1_mod5,*hresTOBL1_rod1_mod6;
  TH1F *hresTOBL1_rod2_mod1,*hresTOBL1_rod2_mod2,*hresTOBL1_rod2_mod3,*hresTOBL1_rod2_mod4,*hresTOBL1_rod2_mod5,*hresTOBL1_rod2_mod6;
  TH1F *hresTOBL2_rod1_mod1,*hresTOBL2_rod1_mod2,*hresTOBL2_rod1_mod3,*hresTOBL2_rod1_mod4,*hresTOBL2_rod1_mod5,*hresTOBL2_rod1_mod6;
  TH1F *hresTOBL2_rod2_mod1,*hresTOBL2_rod2_mod2,*hresTOBL2_rod2_mod3,*hresTOBL2_rod2_mod4,*hresTOBL2_rod2_mod5,*hresTOBL2_rod2_mod6;
 

  bool seed_plus;
  PropagatorWithMaterial  *thePropagator;
  PropagatorWithMaterial  *thePropagatorOp;
  KFUpdator *theUpdator;
  Chi2MeasurementEstimator *theEstimator;
  const TransientTrackingRecHitBuilder *RHBuilder;
  const KFTrajectorySmoother * theSmoother;
  const KFTrajectoryFitter * theFitter;
  TrajectoryStateTransform tsTransform;
  edm::ESHandle<TrackerGeometry> tracker;
  edm::ESHandle<MagneticField> magfield;
  const  CosmicTrajectoryBuilder *pippo;
  bool trinevents;
};


#endif
