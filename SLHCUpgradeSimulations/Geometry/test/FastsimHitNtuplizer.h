#ifndef FastsimHitNtuplizer_h
#define FastsimHitNtuplizer_h

/** \class FastsimHitNtuplizer
 * 
 *
 ************************************************************/

#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h" 

class TTree;
class TFile;

class TransientInitialStateEstimator;
class MagneticField;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class PTrajectoryStateOnDet;

class FastsimHitNtuplizer : public edm::EDAnalyzer
{
 public:
  
  explicit FastsimHitNtuplizer(const edm::ParameterSet& conf);
  virtual ~FastsimHitNtuplizer();
  virtual void beginJob(const edm::EventSetup& es);
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

 protected:

  void fillEvt(const edm::Event& );
  void fillSRecHit(const int subid, const FastTrackerRecHit & recHit,
                   const GeomDet* theGeom);
  void fillPRecHit(const int subid, const FastTrackerRecHit & recHit,
                   const GeomDet* PixGeom);

 private:
  edm::ParameterSet conf_;
  const TrackerGeometry*  theGeometry;

  void init();
  
  //--- Structures for ntupling:
  struct evt
  {
    int run;
    int evtnum;
    
    void init();
  } evt_,stripevt_;
  
  struct RecHit 
  {
    float x;
    float y;
    float xx;
    float xy;
    float yy;
    float row;
    float col;
    float gx;
    float gy;
    float gz;
    int subid;

    void init();
  } recHit_, striprecHit_;

  TFile * tfile_;
  TTree * pixeltree_;
  TTree * striptree_;
};

#endif
