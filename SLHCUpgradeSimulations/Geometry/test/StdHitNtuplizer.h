#ifndef StdHitNtuplizer_h
#define StdHitNtuplizer_h

/** \class StdHitNtuplizer
 * 
 *
 ************************************************************/

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h" 
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
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
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

 protected:

  void fillEvt(const edm::Event& );
  void fillSRecHit(const int subid, SiStripRecHit2DCollection::DetSet::const_iterator pixeliter,
                   const GeomDet* theGeom);
  void fillSRecHit(const int subid, SiStripMatchedRecHit2DCollection::DetSet::const_iterator pixeliter,
                   const GeomDet* theGeom);
  void fillSRecHit(const int subid, SiTrackerGSRecHit2DCollection::const_iterator pixeliter,
                   const GeomDet* theGeom);
  //void fillPRecHit(const int subid, SiPixelRecHitCollection::const_iterator pixeliter,
  //                 const GeomDet* PixGeom);
  void fillPRecHit(const int subid, 
                   const int layer_num,const int ladder_num,const int module_num,
                   const int disk_num,const int blade_num,const int panel_num,const int side_num,
                   SiPixelRecHitCollection::DetSet::const_iterator pixeliter,
                   const int num_simhit,
                   std::vector<PSimHit>::const_iterator closest_simhit,
                   const GeomDet* PixGeom);
  void fillPRecHit(const int subid, trackingRecHit_iterator pixeliter,
                   const GeomDet* PixGeom);

 private:
  edm::ParameterSet conf_;
  edm::InputTag src_;
  edm::InputTag rphiRecHits_;
  edm::InputTag stereoRecHits_;
  edm::InputTag matchedRecHits_;

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
    int subid,module;
    int layer,ladder;		// BPix
    int disk,blade,panel,side;	// FPix
    int nsimhit;
    int spreadx,spready;
    float hx, hy;
    float tx, ty;
    float theta, phi;
    void init();
  } recHit_, striprecHit_;

  TFile * tfile_;
  TTree * pixeltree_;
  TTree * striptree_;
  TTree * pixeltree2_;
};

#endif
