#ifndef RecoTracker_SingleTrackPattern_ReadCosmicTracks_h
#define RecoTracker_SingleTrackPattern_ReadCosmicTracks_h

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include <TROOT.h>
#include <TFile.h>
#include <TH1F.h>

class CompareTOF {
 public:
  CompareTOF(){};
   bool operator()( PSimHit ps1,
		    PSimHit ps2){

     return ps1.tof()<ps2.tof();};
 };
class ReadCosmicTracks : public edm::EDAnalyzer
{
 typedef TrajectoryStateOnSurface     TSOS;
 public:
  
  explicit ReadCosmicTracks(const edm::ParameterSet& conf);
  
  virtual ~ReadCosmicTracks();
  virtual void beginRun(edm::Run & run, const edm::EventSetup& c);
  virtual void endJob(); 
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);  
  void makeResiduals(const Trajectory traj);  
 private:
  edm::ParameterSet conf_;
  std::vector<PSimHit> theStripHits;
  edm::ESHandle<TrackerGeometry> tracker;
  edm::ESHandle<MagneticField> magfield;
  TFile* hFile;
  TH1F  *hptres;
  TH1F *hptres1,*hptres2,*hptres3,*hptres4,*hptres5,*hptres6,*hptres7,*hptres8,*hptres9,*hptres10;
  TH1F  *hchiSq;
  TH1F *hchiSq1,*hchiSq2,*hchiSq3,*hchiSq4,*hchiSq5,*hchiSq6,*hchiSq7,*hchiSq8,*hchiSq9,*hchiSq10;
  TH1F  *heffpt,*hrespt,*hchipt,*hchiR;
  TH1F  *heffhit,*hcharge;
  TH1F  *hresTIB,*hresTOB,*hresTID,*hresTEC;
  unsigned int inum[10];
  unsigned int iden[10];
  unsigned int inum2[9];
  unsigned int iden2[9];
  float ichiR[10];
  unsigned int iden3[10];
  bool seed_plus;
  const TransientTrackingRecHitBuilder *RHBuilder;
  const PSimHit *isim;
  bool trinevents;
  bool trackable_cosmic;
  unsigned int GlobDen;
  unsigned int GlobNum;
};


#endif
