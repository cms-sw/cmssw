#ifndef Validation_RecoTrack_SiStripTrackingRecHitsValid_h
#define Validation_RecoTrack_SiStripTrackingRecHitsValid_h

//DQM services for histogram
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/RecoTrack/interface/TrackLocalAngle.h"
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TProfile.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//--- for SimHit association
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"  
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h" 
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h" 
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include <string>

using namespace std;
using namespace edm;

class SiStripTrackingRecHitsValid : public edm::EDAnalyzer
{
 public:
  
  explicit SiStripTrackingRecHitsValid(const edm::ParameterSet& conf);
  
  virtual ~SiStripTrackingRecHitsValid();

  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  std::pair<LocalPoint,LocalVector> projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,const BoundPlane& plane);

 private:

  edm::ParameterSet conf_;
  TrackLocalAngle *anglefinder_;
  DaqMonitorBEInterface* dbe_;
  string outputFile_;
  string src_;
  string builderName_;
  bool MTCCtrack_;

  //TIB
  MonitorElement* meNstpRphiTIB[4];
  MonitorElement* meAdcRphiTIB[4];
  MonitorElement* mePosxRphiTIB[4];
  MonitorElement* meErrxRphiTIB[4];
  MonitorElement* meResRphiTIB[4];
  MonitorElement* mePullRphiTIB[4];
  MonitorElement* meTrackangleRphiTIB[4];
  MonitorElement* mePullTrackangleProfileRphiTIB[4];
  MonitorElement* mePullTrackangle2DRphiTIB[4];
  MonitorElement* meNstpSasTIB[4];
  MonitorElement* meAdcSasTIB[4];
  MonitorElement* mePosxSasTIB[4];
  MonitorElement* meErrxSasTIB[4];
  MonitorElement* meResSasTIB[4];
  MonitorElement* mePullSasTIB[4];
  MonitorElement* meTrackangleSasTIB[4];
  MonitorElement* mePullTrackangleProfileSasTIB[4];
  MonitorElement* mePosxMatchedTIB[2];
  MonitorElement* mePosyMatchedTIB[2];
  MonitorElement* meErrxMatchedTIB[2];
  MonitorElement* meErryMatchedTIB[2];
  MonitorElement* meResxMatchedTIB[2];
  MonitorElement* meResyMatchedTIB[2];
  MonitorElement* mePullxMatchedTIB[2];
  MonitorElement* mePullyMatchedTIB[2];
  //TOB
  MonitorElement* meNstpRphiTOB[6];
  MonitorElement* meAdcRphiTOB[6];
  MonitorElement* mePosxRphiTOB[6];
  MonitorElement* meErrxRphiTOB[6];
  MonitorElement* meResRphiTOB[6];
  MonitorElement* mePullRphiTOB[6];
  MonitorElement* meTrackangleRphiTOB[6];
  MonitorElement* mePullTrackangleProfileRphiTOB[6];
  MonitorElement* meNstpSasTOB[2];
  MonitorElement* meAdcSasTOB[2];
  MonitorElement* mePosxSasTOB[2];
  MonitorElement* meErrxSasTOB[2];
  MonitorElement* meResSasTOB[2];
  MonitorElement* mePullSasTOB[2];
  MonitorElement* meTrackangleSasTOB[2];
  MonitorElement* mePullTrackangleProfileSasTOB[2];
  MonitorElement* mePosxMatchedTOB[2];
  MonitorElement* mePosyMatchedTOB[2];
  MonitorElement* meErrxMatchedTOB[2];
  MonitorElement* meErryMatchedTOB[2];
  MonitorElement* meResxMatchedTOB[2];
  MonitorElement* meResyMatchedTOB[2];
  MonitorElement* mePullxMatchedTOB[2];
  MonitorElement* mePullyMatchedTOB[2];
  //TID
  MonitorElement* meNstpRphiTID[3];
  MonitorElement* meAdcRphiTID[3];
  MonitorElement* mePosxRphiTID[3];
  MonitorElement* meErrxRphiTID[3];
  MonitorElement* meResRphiTID[3];
  MonitorElement* mePullRphiTID[3];
  MonitorElement* meTrackangleRphiTID[3];
  MonitorElement* mePullTrackangleProfileRphiTID[3];
  MonitorElement* meNstpSasTID[2];
  MonitorElement* meAdcSasTID[2];
  MonitorElement* mePosxSasTID[2];
  MonitorElement* meErrxSasTID[2];
  MonitorElement* meResSasTID[2];
  MonitorElement* mePullSasTID[2];
  MonitorElement* meTrackangleSasTID[2];
  MonitorElement* mePullTrackangleProfileSasTID[2];
  MonitorElement* mePosxMatchedTID[2];
  MonitorElement* mePosyMatchedTID[2];
  MonitorElement* meErrxMatchedTID[2];
  MonitorElement* meErryMatchedTID[2];
  MonitorElement* meResxMatchedTID[2];
  MonitorElement* meResyMatchedTID[2];
  MonitorElement* mePullxMatchedTID[2];
  MonitorElement* mePullyMatchedTID[2];
 //TEC
  MonitorElement* meNstpRphiTEC[7];
  MonitorElement* meAdcRphiTEC[7];
  MonitorElement* mePosxRphiTEC[7];
  MonitorElement* meErrxRphiTEC[7];
  MonitorElement* meResRphiTEC[7];
  MonitorElement* mePullRphiTEC[7];
  MonitorElement* meTrackangleRphiTEC[7];
  MonitorElement* mePullTrackangleProfileRphiTEC[7];
  MonitorElement* meNstpSasTEC[5];
  MonitorElement* meAdcSasTEC[5];
  MonitorElement* mePosxSasTEC[5];
  MonitorElement* meErrxSasTEC[5];
  MonitorElement* meResSasTEC[5];
  MonitorElement* mePullSasTEC[5];
  MonitorElement* meTrackangleSasTEC[5];
  MonitorElement* mePullTrackangleProfileSasTEC[5];
  MonitorElement* mePosxMatchedTEC[5];
  MonitorElement* mePosyMatchedTEC[5];
  MonitorElement* meErrxMatchedTEC[5];
  MonitorElement* meErryMatchedTEC[5];
  MonitorElement* meResxMatchedTEC[5];
  MonitorElement* meResyMatchedTEC[5];
  MonitorElement* mePullxMatchedTEC[5];
  MonitorElement* mePullyMatchedTEC[5];

  const StripTopology* topol;

  float rechitrphix;
  float rechitrphierrx;
  float rechitrphiy;
  float rechitrphiz;
  float rechitrphiphi;
  float rechitrphires;
  float rechitrphipull;
  float rechitrphitrackangle;
  int   clusizrphi;
  float cluchgrphi;
  float rechitsasx;
  float rechitsaserrx;
  float rechitsasy;
  float rechitsasz;
  float rechitsasphi;
  float rechitsasres;
  float rechitsaspull;
  float rechitsastrackangle;
  int   clusizsas;
  float cluchgsas;
  float rechitmatchedx;
  float rechitmatchedy;
  float rechitmatchedz;
  float rechitmatchederrxx;
  float rechitmatchederrxy;
  float rechitmatchederryy;
  float rechitmatchedphi;
  float rechitmatchedresx;
  float rechitmatchedresy;
  float rechitmatchedpullx;
  float rechitmatchedpully;
  float rechitmatchedtrackangle;

};


#endif
