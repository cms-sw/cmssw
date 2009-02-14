#ifndef StdHitNtuplizer_h
#define StdHitNtuplizer_h

/** \class StdHitNtuplizer
 * 
 *
 ************************************************************/

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"

//#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h" 
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

class TTree;
class TFile;
class RectangularPixelTopology;

class TransientInitialStateEstimator;
class MagneticField;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class PTrajectoryStateOnDet;

class StdHitNtuplizer : public edm::EDAnalyzer
{
 public:
  
  explicit StdHitNtuplizer(const edm::ParameterSet& conf);
  virtual ~StdHitNtuplizer();
  virtual void beginJob(const edm::EventSetup& es);
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

 protected:

  void fillEvt(const edm::Event& );
  void fillSRecHit(const int subid, SiTrackerGSRecHit2DCollection::const_iterator pixeliter,
                   const GeomDet* theGeom);
  void fillPRecHit(const int subid, SiTrackerGSRecHit2DCollection::const_iterator pixeliter,
                   const GeomDet* PixGeom);
  void fillPRecHit(const int subid, SiPixelRecHitCollection::const_iterator pixeliter,
                   const GeomDet* PixGeom);
  void fillPRecHit(const int subid, trackingRecHit_iterator pixeliter,
                   const GeomDet* PixGeom);

 private:
  edm::ParameterSet conf_;
  const TrackerGeometry*  theGeometry;
  edm::InputTag src_;

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
  TTree * pixeltree2_;
};

#endif
