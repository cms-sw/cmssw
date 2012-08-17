#ifndef TrackRecoDeDx_HSCPDeDxInfoProducer_H
#define TrackRecoDeDx_HSCPDeDxInfoProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h" 

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPDeDxInfo.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

#include "RecoTracker/DeDx/interface/DeDxDiscriminatorTools.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"


#include "TH3F.h"
#include "TChain.h"

#include <ext/hash_map>


class HSCPDeDxInfoProducer : public edm::EDProducer {

public:

  explicit HSCPDeDxInfoProducer(const edm::ParameterSet&);
  ~HSCPDeDxInfoProducer();

private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void FillInfo(const SiStripCluster*   cluster, TrajectoryStateOnSurface trajState,const uint32_t &, susybsm::HSCPDeDxInfo& hscpDeDxInfo);
  void FillPosition(TrajectoryStateOnSurface trajState,  const uint32_t &  detId, susybsm::HSCPDeDxInfo& hscpDeDxInfo);
  void   MakeCalibrationMap();


  // ----------member data ---------------------------
  edm::InputTag                     m_trajTrackAssociationTag;
  edm::InputTag                     m_tracksTag;

  bool usePixel;
  bool useStrip;
  double MeVperADCPixel;
  double MeVperADCStrip;

  std::string                       m_calibrationPath;
  bool                              useCalibration;
  bool				    shapetest;

  const TrackerGeometry* m_tracker;

  PhysicsTools::Calibration::HistogramD3D DeDxMap_;

  double       MinTrackMomentum;
  double       MaxTrackMomentum;
  double       MinTrackEta;
  double       MaxTrackEta;
  unsigned int MaxNrStrips;
  unsigned int MinTrackHits;
  double       MaxTrackChiOverNdf;

  unsigned int Formula;
  std::string       Reccord;
  std::string       ProbabilityMode;


  TH3D*        Prob_ChargePath;



   private :
      struct stModInfo{int DetId; int SubDet; float Eta; float R; float Thickness; int NAPV; double Gain; double Width; double Length; std::vector<float> trapezoParams;};

      class isEqual{
         public:
                 template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
      };

  __gnu_cxx::hash_map<unsigned int, stModInfo*,  __gnu_cxx::hash<unsigned int>, isEqual > MODsColl;
};

#endif

