#ifndef Validation_RecoTrack_SiStripTrackingRecHitsValid_h
#define Validation_RecoTrack_SiStripTrackingRecHitsValid_h

//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"

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

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include <string>

class SiStripTrackingRecHitsValid : public edm::EDAnalyzer
{
 public:
  
  explicit SiStripTrackingRecHitsValid(const edm::ParameterSet& conf);
  
  virtual ~SiStripTrackingRecHitsValid();

  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  void endJob();

  std::pair<LocalPoint,LocalVector> projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,const BoundPlane& plane);

  LocalVector driftDirection(const StripGeomDetUnit* det)const;

  MonitorElement* Fit_SliceY(TH2F * Histo2D);

 private:

  edm::ParameterSet conf_;

  DQMStore* dbe_;
  std::string outputFile_;

  MonitorElement* PullRMSvsTrackwidth;
  MonitorElement* PullRMSvsExpectedwidth;
  MonitorElement* PullRMSvsClusterwidth;
  MonitorElement* PullRMSvsTrackangle;
  MonitorElement* PullRMSvsTrackanglebeta;

  MonitorElement* PullRMSvsTrackwidthTIB;
  MonitorElement* PullRMSvsExpectedwidthTIB;
  MonitorElement* PullRMSvsClusterwidthTIB;
  MonitorElement* PullRMSvsTrackangleTIB;
  MonitorElement* PullRMSvsTrackanglebetaTIB;

  MonitorElement* PullRMSvsTrackwidthTOB;
  MonitorElement* PullRMSvsExpectedwidthTOB;
  MonitorElement* PullRMSvsClusterwidthTOB;
  MonitorElement* PullRMSvsTrackangleTOB;
  MonitorElement* PullRMSvsTrackanglebetaTOB;

  MonitorElement* PullRMSvsTrackwidthTID;
  MonitorElement* PullRMSvsExpectedwidthTID;
  MonitorElement* PullRMSvsClusterwidthTID;
  MonitorElement* PullRMSvsTrackangleTID;
  MonitorElement* PullRMSvsTrackanglebetaTID;

  MonitorElement* PullRMSvsTrackwidthTEC;
  MonitorElement* PullRMSvsExpectedwidthTEC;
  MonitorElement* PullRMSvsClusterwidthTEC;
  MonitorElement* PullRMSvsTrackangleTEC;
  MonitorElement* PullRMSvsTrackanglebetaTEC;


  // ALL

  MonitorElement* meCategory;
  MonitorElement* meTrackwidth;
  MonitorElement* meExpectedwidth;
  MonitorElement* meClusterwidth;
  MonitorElement* meTrackanglealpha;
  MonitorElement* meTrackanglebeta;
  MonitorElement* meErrxMFTrackwidthProfile;
  MonitorElement* meErrxMFTrackwidthProfileWClus1;
  MonitorElement* meErrxMFTrackwidthProfileWClus2;
  MonitorElement* meErrxMFTrackwidthProfileWClus3;
  MonitorElement* meErrxMFTrackwidthProfileWClus4;
  MonitorElement* meResMFTrackwidthProfileWClus1;

  MonitorElement* meResMFTrackwidthProfileWClus2;
  MonitorElement* meResMFTrackwidthProfileWClus21;
  MonitorElement* meResMFTrackwidthProfileWClus22;
  MonitorElement* meResMFTrackwidthProfileWClus23;

  MonitorElement* meResMFTrackwidthProfileWClus3;
  MonitorElement* meResMFTrackwidthProfileWClus4;
  MonitorElement* meErrxMFTrackwidthProfileCategory1;
  MonitorElement* meErrxMFTrackwidthProfileCategory2;
  MonitorElement* meErrxMFTrackwidthProfileCategory3;
  MonitorElement* meErrxMFTrackwidthProfileCategory4;
  MonitorElement* meErrxMFClusterwidthProfileCategory1;
  MonitorElement* meErrxMFAngleProfile;
  MonitorElement* meErrxLF;
  MonitorElement* meResLF;
  MonitorElement* mePullLF;
  MonitorElement* meErrxMF;
  MonitorElement* meResMF;
  MonitorElement* mePullMF;


  //TIB
  MonitorElement* meNstpRphiTIB[4];
  MonitorElement* meAdcRphiTIB[4];
  MonitorElement* mePosxRphiTIB[4];
  MonitorElement* meErrxLFRphiTIB[4];
  MonitorElement* meErrxMFRphiTIB[4];
  MonitorElement* meResLFRphiTIB[4];
  MonitorElement* meResMFRphiTIB[4];
  MonitorElement* mePullLFRphiTIB[4];
  MonitorElement* mePullMFRphiTIB[4];
  MonitorElement* meTrackangleRphiTIB[4];
  MonitorElement* meTrackanglebetaRphiTIB[4];
  MonitorElement* meTrackangle2RphiTIB[4];
  MonitorElement* mePullTrackangleProfileRphiTIB[4];
  MonitorElement* mePullTrackangle2DRphiTIB[4];
  MonitorElement* meTrackwidthRphiTIB[4];
  MonitorElement* meExpectedwidthRphiTIB[4];
  MonitorElement* meClusterwidthRphiTIB[4];
  MonitorElement* meCategoryRphiTIB[4];
  MonitorElement* mePullTrackwidthProfileRphiTIB[4];
  MonitorElement* mePullTrackwidthProfileCategory1RphiTIB[4];
  MonitorElement* mePullTrackwidthProfileCategory2RphiTIB[4];
  MonitorElement* mePullTrackwidthProfileCategory3RphiTIB[4];
  MonitorElement* mePullTrackwidthProfileCategory4RphiTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileRphiTIB[4];

  MonitorElement* meErrxMFTrackwidthProfileWclus1RphiTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileWclus2RphiTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileWclus3RphiTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileWclus4RphiTIB[4];
  MonitorElement* meResMFTrackwidthProfileWclus1RphiTIB[4];
  MonitorElement* meResMFTrackwidthProfileWclus2RphiTIB[4];
  MonitorElement* meResMFTrackwidthProfileWclus3RphiTIB[4];
  MonitorElement* meResMFTrackwidthProfileWclus4RphiTIB[4];

  MonitorElement* meErrxMFTrackwidthProfileCategory1RphiTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileCategory2RphiTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileCategory3RphiTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileCategory4RphiTIB[4];
  MonitorElement* meErrxMFClusterwidthProfileCategory1RphiTIB[4];
  MonitorElement* meErrxMFAngleProfileRphiTIB[4];

  MonitorElement* meNstpSasTIB[4];
  MonitorElement* meAdcSasTIB[4];
  MonitorElement* mePosxSasTIB[4];
  MonitorElement* meErrxLFSasTIB[4];
  MonitorElement* meErrxMFSasTIB[4];
  MonitorElement* meResLFSasTIB[4];
  MonitorElement* meResMFSasTIB[4];
  MonitorElement* mePullLFSasTIB[4];
  MonitorElement* mePullMFSasTIB[4];
  MonitorElement* meTrackangleSasTIB[4];
  MonitorElement* meTrackanglebetaSasTIB[4];
  MonitorElement* mePullTrackangleProfileSasTIB[4];
  MonitorElement* meTrackwidthSasTIB[4];
  MonitorElement* meExpectedwidthSasTIB[4];
  MonitorElement* meClusterwidthSasTIB[4];
  MonitorElement* meCategorySasTIB[4];
  MonitorElement* mePullTrackwidthProfileSasTIB[4];
  MonitorElement* mePullTrackwidthProfileCategory1SasTIB[4];
  MonitorElement* mePullTrackwidthProfileCategory2SasTIB[4];
  MonitorElement* mePullTrackwidthProfileCategory3SasTIB[4];
  MonitorElement* mePullTrackwidthProfileCategory4SasTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileSasTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileCategory1SasTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileCategory2SasTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileCategory3SasTIB[4];
  MonitorElement* meErrxMFTrackwidthProfileCategory4SasTIB[4];
  MonitorElement* meErrxMFClusterwidthProfileCategory1SasTIB[4];
  MonitorElement* meErrxMFAngleProfileSasTIB[4];

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
  MonitorElement* meErrxLFRphiTOB[6];
  MonitorElement* meResLFRphiTOB[6];
  MonitorElement* mePullLFRphiTOB[6];
  MonitorElement* meErrxMFRphiTOB[6];
  MonitorElement* meResMFRphiTOB[6];
  MonitorElement* mePullMFRphiTOB[6];
  MonitorElement* meTrackangleRphiTOB[6];
  MonitorElement* meTrackanglebetaRphiTOB[6];
  MonitorElement* mePullTrackangleProfileRphiTOB[6];
  MonitorElement* meTrackwidthRphiTOB[6];
  MonitorElement* meExpectedwidthRphiTOB[6];
  MonitorElement* meClusterwidthRphiTOB[6];
  MonitorElement* meCategoryRphiTOB[6];
  MonitorElement* mePullTrackwidthProfileRphiTOB[6];
  MonitorElement* mePullTrackwidthProfileCategory1RphiTOB[6];
  MonitorElement* mePullTrackwidthProfileCategory2RphiTOB[6];
  MonitorElement* mePullTrackwidthProfileCategory3RphiTOB[6];
  MonitorElement* mePullTrackwidthProfileCategory4RphiTOB[6];
  MonitorElement* meErrxMFTrackwidthProfileRphiTOB[6];

  MonitorElement* meErrxMFTrackwidthProfileWclus1RphiTOB[6];
  MonitorElement* meErrxMFTrackwidthProfileWclus2RphiTOB[6];
  MonitorElement* meErrxMFTrackwidthProfileWclus3RphiTOB[6];
  MonitorElement* meErrxMFTrackwidthProfileWclus4RphiTOB[6];
  MonitorElement* meResMFTrackwidthProfileWclus1RphiTOB[6];
  MonitorElement* meResMFTrackwidthProfileWclus2RphiTOB[6];
  MonitorElement* meResMFTrackwidthProfileWclus3RphiTOB[6];
  MonitorElement* meResMFTrackwidthProfileWclus4RphiTOB[6];

  MonitorElement* meErrxMFTrackwidthProfileCategory1RphiTOB[6];
  MonitorElement* meErrxMFTrackwidthProfileCategory2RphiTOB[6];
  MonitorElement* meErrxMFTrackwidthProfileCategory3RphiTOB[6];
  MonitorElement* meErrxMFTrackwidthProfileCategory4RphiTOB[6];
  MonitorElement* meErrxMFClusterwidthProfileCategory1RphiTOB[6];
  MonitorElement* meErrxMFAngleProfileRphiTOB[6];

  MonitorElement* meNstpSasTOB[2];
  MonitorElement* meAdcSasTOB[2];
  MonitorElement* mePosxSasTOB[2];
  MonitorElement* meErrxLFSasTOB[2];
  MonitorElement* meResLFSasTOB[2];
  MonitorElement* mePullLFSasTOB[2];
  MonitorElement* meErrxMFSasTOB[2];
  MonitorElement* meResMFSasTOB[2];
  MonitorElement* mePullMFSasTOB[2];
  MonitorElement* meTrackangleSasTOB[2];
  MonitorElement* meTrackanglebetaSasTOB[2];
  MonitorElement* mePullTrackangleProfileSasTOB[2];
  MonitorElement* meTrackwidthSasTOB[2];
  MonitorElement* meExpectedwidthSasTOB[2];
  MonitorElement* meClusterwidthSasTOB[2];
  MonitorElement* meCategorySasTOB[2];
  MonitorElement* mePullTrackwidthProfileSasTOB[2];
  MonitorElement* mePullTrackwidthProfileCategory1SasTOB[2];
  MonitorElement* mePullTrackwidthProfileCategory2SasTOB[2];
  MonitorElement* mePullTrackwidthProfileCategory3SasTOB[2];
  MonitorElement* mePullTrackwidthProfileCategory4SasTOB[2];
  MonitorElement* meErrxMFTrackwidthProfileSasTOB[2];
  MonitorElement* meErrxMFTrackwidthProfileCategory1SasTOB[2];
  MonitorElement* meErrxMFTrackwidthProfileCategory2SasTOB[2];
  MonitorElement* meErrxMFTrackwidthProfileCategory3SasTOB[2];
  MonitorElement* meErrxMFTrackwidthProfileCategory4SasTOB[2];
  MonitorElement* meErrxMFClusterwidthProfileCategory1SasTOB[2];
  MonitorElement* meErrxMFAngleProfileSasTOB[2];

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
  MonitorElement* meErrxLFRphiTID[3];
  MonitorElement* meResLFRphiTID[3];
  MonitorElement* mePullLFRphiTID[3];
  MonitorElement* meErrxMFRphiTID[3];
  MonitorElement* meResMFRphiTID[3];
  MonitorElement* mePullMFRphiTID[3];
  MonitorElement* meTrackangleRphiTID[3];
  MonitorElement* meTrackanglebetaRphiTID[3];
  MonitorElement* mePullTrackangleProfileRphiTID[3];
  MonitorElement* meTrackwidthRphiTID[3];
  MonitorElement* meExpectedwidthRphiTID[3];
  MonitorElement* meClusterwidthRphiTID[3];
  MonitorElement* meCategoryRphiTID[3];
  MonitorElement* mePullTrackwidthProfileRphiTID[3];
  MonitorElement* mePullTrackwidthProfileCategory1RphiTID[3];
  MonitorElement* mePullTrackwidthProfileCategory2RphiTID[3];
  MonitorElement* mePullTrackwidthProfileCategory3RphiTID[3];
  MonitorElement* mePullTrackwidthProfileCategory4RphiTID[3];
  MonitorElement* meErrxMFTrackwidthProfileRphiTID[3];
  MonitorElement* meErrxMFTrackwidthProfileCategory1RphiTID[3];
  MonitorElement* meErrxMFTrackwidthProfileCategory2RphiTID[3];
  MonitorElement* meErrxMFTrackwidthProfileCategory3RphiTID[3];
  MonitorElement* meErrxMFTrackwidthProfileCategory4RphiTID[3];
  MonitorElement* meErrxMFClusterwidthProfileCategory1RphiTID[3];
  MonitorElement* meErrxMFAngleProfileRphiTID[3];

  MonitorElement* meNstpSasTID[2];
  MonitorElement* meAdcSasTID[2];
  MonitorElement* mePosxSasTID[2];
  MonitorElement* meErrxLFSasTID[2];
  MonitorElement* meResLFSasTID[2];
  MonitorElement* mePullLFSasTID[2];
  MonitorElement* meErrxMFSasTID[2];
  MonitorElement* meResMFSasTID[2];
  MonitorElement* mePullMFSasTID[2];
  MonitorElement* meTrackangleSasTID[2];
  MonitorElement* meTrackanglebetaSasTID[2];
  MonitorElement* mePullTrackangleProfileSasTID[2];
  MonitorElement* meTrackwidthSasTID[2];
  MonitorElement* meExpectedwidthSasTID[2];
  MonitorElement* meClusterwidthSasTID[2];
  MonitorElement* meCategorySasTID[2];
  MonitorElement* mePullTrackwidthProfileSasTID[2];
  MonitorElement* mePullTrackwidthProfileCategory1SasTID[2];
  MonitorElement* mePullTrackwidthProfileCategory2SasTID[2];
  MonitorElement* mePullTrackwidthProfileCategory3SasTID[2];
  MonitorElement* mePullTrackwidthProfileCategory4SasTID[2];
  MonitorElement* meErrxMFTrackwidthProfileSasTID[2];
  MonitorElement* meErrxMFTrackwidthProfileCategory1SasTID[2];
  MonitorElement* meErrxMFTrackwidthProfileCategory2SasTID[2];
  MonitorElement* meErrxMFTrackwidthProfileCategory3SasTID[2];
  MonitorElement* meErrxMFTrackwidthProfileCategory4SasTID[2];
  MonitorElement* meErrxMFClusterwidthProfileCategory1SasTID[2];
  MonitorElement* meErrxMFAngleProfileSasTID[2];

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
  MonitorElement* meErrxLFRphiTEC[7];
  MonitorElement* meResLFRphiTEC[7];
  MonitorElement* mePullLFRphiTEC[7];
  MonitorElement* meErrxMFRphiTEC[7];
  MonitorElement* meResMFRphiTEC[7];
  MonitorElement* mePullMFRphiTEC[7];
  MonitorElement* meTrackangleRphiTEC[7];
  MonitorElement* meTrackanglebetaRphiTEC[7];
  MonitorElement* mePullTrackangleProfileRphiTEC[7];
  MonitorElement* meTrackwidthRphiTEC[7];
  MonitorElement* meExpectedwidthRphiTEC[7];
  MonitorElement* meClusterwidthRphiTEC[7];
  MonitorElement* meCategoryRphiTEC[7];
  MonitorElement* mePullTrackwidthProfileRphiTEC[7];
  MonitorElement* mePullTrackwidthProfileCategory1RphiTEC[7];
  MonitorElement* mePullTrackwidthProfileCategory2RphiTEC[7];
  MonitorElement* mePullTrackwidthProfileCategory3RphiTEC[7];
  MonitorElement* mePullTrackwidthProfileCategory4RphiTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileSasTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileCategory1SasTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileCategory2SasTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileCategory3SasTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileCategory4SasTEC[7];
  MonitorElement* meErrxMFClusterwidthProfileCategory1SasTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileRphiTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileCategory1RphiTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileCategory2RphiTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileCategory3RphiTEC[7];
  MonitorElement* meErrxMFTrackwidthProfileCategory4RphiTEC[7];
  MonitorElement* meErrxMFClusterwidthProfileCategory1RphiTEC[7];
  MonitorElement* meErrxMFAngleProfileRphiTEC[7];

  MonitorElement* meNstpSasTEC[5];
  MonitorElement* meAdcSasTEC[5];
  MonitorElement* mePosxSasTEC[5];
  MonitorElement* meErrxLFSasTEC[5];
  MonitorElement* meResLFSasTEC[5];
  MonitorElement* mePullLFSasTEC[5];
  MonitorElement* meErrxMFSasTEC[5];
  MonitorElement* meResMFSasTEC[5];
  MonitorElement* mePullMFSasTEC[5];
  MonitorElement* meTrackangleSasTEC[5];
  MonitorElement* meTrackanglebetaSasTEC[5];
  MonitorElement* mePullTrackangleProfileSasTEC[5];
  MonitorElement* meTrackwidthSasTEC[5];
  MonitorElement* meExpectedwidthSasTEC[5];
  MonitorElement* meClusterwidthSasTEC[5];
  MonitorElement* meCategorySasTEC[5];
  MonitorElement* mePullTrackwidthProfileSasTEC[5];
  MonitorElement* mePullTrackwidthProfileCategory1SasTEC[5];
  MonitorElement* mePullTrackwidthProfileCategory2SasTEC[5];
  MonitorElement* mePullTrackwidthProfileCategory3SasTEC[5];
  MonitorElement* mePullTrackwidthProfileCategory4SasTEC[5];
  MonitorElement* meErrxMFAngleProfileSasTEC[5];

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
  float rechitrphierrxLF;
  float rechitrphierrxMF;
  float rechitrphiy;
  float rechitrphiz;
  float rechitrphiphi;
  float rechitrphires;
  float rechitrphiresLF;
  float rechitrphiresMF;
  float rechitrphipull;
  float rechitrphipullLF;
  float rechitrphipullMF;
  float rechitrphitrackangle;
  float rechitrphitrackanglebeta;
  float rechitrphitrackangle2;
  float rechitrphitrackwidth;
  int rechitrphiexpectedwidth;
  int rechitrphicategory;
  int   clusizrphi;
  float cluchgrphi;
  float rechitsasx;
  float rechitsaserrx;
  float rechitsaserrxLF;
  float rechitsaserrxMF;
  float rechitsasy;
  float rechitsasz;
  float rechitsasphi;
  float rechitsasres;
  float rechitsasresLF;
  float rechitsasresMF;
  float rechitsaspull;
  float rechitsaspullLF;
  float rechitsaspullMF;
  float rechitsastrackangle;
  float rechitsastrackanglebeta;
  float rechitsastrackwidth;
  int rechitsasexpectedwidth;
  int rechitsascategory;
  float  rechitrphithickness;
  float  rechitsasthickness;

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

 protected:  
    const MagneticField * magfield2_ ;

};


#endif
