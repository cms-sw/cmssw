#ifndef Validation_RecoTrack_SiStripTrackingRecHitsValid_h
#define Validation_RecoTrack_SiStripTrackingRecHitsValid_h

//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

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

class SiStripDetCabling;
class SiStripDCSStatus;

class SiStripTrackingRecHitsValid : public thread_unsafe::DQMEDAnalyzer
{
 public:
  
  SiStripTrackingRecHitsValid(const edm::ParameterSet& conf);
  
  ~SiStripTrackingRecHitsValid();

  // ALL
  //Simple hits MEs either from matched either 
  //from hit1D, hit2D in all subdetectors.
  struct SimpleHitsMEs{ 
    MonitorElement* meCategory;
    MonitorElement* meTrackwidth;
    MonitorElement* meExpectedwidth;
    MonitorElement* meClusterwidth;
    MonitorElement* meTrackanglealpha;
    MonitorElement* meTrackanglebeta;
    MonitorElement* meResolxMFTrackwidthProfile;
    MonitorElement* meResolxMFTrackwidthProfileWClus1;
    MonitorElement* meResolxMFTrackwidthProfileWClus2;
    MonitorElement* meResolxMFTrackwidthProfileWClus3;
    MonitorElement* meResolxMFTrackwidthProfileWClus4;
    MonitorElement* meResMFTrackwidthProfileWClus1;

    MonitorElement* meResMFTrackwidthProfileWClus2;
    MonitorElement* meResMFTrackwidthProfileWClus21;
    MonitorElement* meResMFTrackwidthProfileWClus22;
    MonitorElement* meResMFTrackwidthProfileWClus23;

    MonitorElement* meResMFTrackwidthProfileWClus3;
    MonitorElement* meResMFTrackwidthProfileWClus4;
    MonitorElement* meResolxMFTrackwidthProfileCategory1;
    MonitorElement* meResolxMFTrackwidthProfileCategory2;
    MonitorElement* meResolxMFTrackwidthProfileCategory3;
    MonitorElement* meResolxMFTrackwidthProfileCategory4;
    MonitorElement* meResolxMFClusterwidthProfileCategory1;
    MonitorElement* meResolxMFAngleProfile;
    MonitorElement* meResolxLF;
    MonitorElement* meResLF;
    MonitorElement* mePullLF;
    MonitorElement* meResolxMF;
    MonitorElement* meResMF;
    MonitorElement* mePullMF;

  };

  struct LayerMEs{ // MEs for Layer Level
    MonitorElement* meWclusRphi;
    MonitorElement* meAdcRphi;
    MonitorElement* meResolxLFRphi;
    MonitorElement* meResolxMFRphi;
    MonitorElement* meResolxMFRphiwclus1;
    MonitorElement* meResolxMFRphiwclus2;
    MonitorElement* meResolxMFRphiwclus3;
    MonitorElement* meResolxMFRphiwclus4;
    MonitorElement* meResLFRphi;
    MonitorElement* meResMFRphi;
    MonitorElement* meResMFRphiwclus1;
    MonitorElement* meResMFRphiwclus2;
    MonitorElement* meResMFRphiwclus3;
    MonitorElement* meResMFRphiwclus4;
    MonitorElement* mePullLFRphi;
    MonitorElement* mePullMFRphi;
    MonitorElement* mePullMFRphiwclus1;
    MonitorElement* mePullMFRphiwclus2;
    MonitorElement* mePullMFRphiwclus3;
    MonitorElement* mePullMFRphiwclus4;
    MonitorElement* meTrackangleRphi;
    MonitorElement* meTrackanglebetaRphi;
    MonitorElement* meTrackangle2Rphi;
    MonitorElement* mePullTrackangleProfileRphi;
    MonitorElement* mePullTrackangle2DRphi;
    MonitorElement* meTrackwidthRphi;
    MonitorElement* meExpectedwidthRphi;
    MonitorElement* meClusterwidthRphi;
    MonitorElement* meCategoryRphi;
    MonitorElement* mePullTrackwidthProfileRphi;
    MonitorElement* mePullTrackwidthProfileRphiwclus1;
    MonitorElement* mePullTrackwidthProfileRphiwclus2;
    MonitorElement* mePullTrackwidthProfileRphiwclus3;
    MonitorElement* mePullTrackwidthProfileRphiwclus4;
    MonitorElement* mePullTrackwidthProfileCategory1Rphi;
    MonitorElement* mePullTrackwidthProfileCategory2Rphi;
    MonitorElement* mePullTrackwidthProfileCategory3Rphi;
    MonitorElement* mePullTrackwidthProfileCategory4Rphi;
    MonitorElement* meResolxMFTrackwidthProfileRphi;

    MonitorElement* meResolxMFTrackwidthProfileWclus1Rphi;
    MonitorElement* meResolxMFTrackwidthProfileWclus2Rphi;
    MonitorElement* meResolxMFTrackwidthProfileWclus3Rphi;
    MonitorElement* meResolxMFTrackwidthProfileWclus4Rphi;
    MonitorElement* meResMFTrackwidthProfileWclus1Rphi;
    MonitorElement* meResMFTrackwidthProfileWclus2Rphi;
    MonitorElement* meResMFTrackwidthProfileWclus3Rphi;
    MonitorElement* meResMFTrackwidthProfileWclus4Rphi;

    MonitorElement* meResolxMFTrackwidthProfileCategory1Rphi;
    MonitorElement* meResolxMFTrackwidthProfileCategory2Rphi;
    MonitorElement* meResolxMFTrackwidthProfileCategory3Rphi;
    MonitorElement* meResolxMFTrackwidthProfileCategory4Rphi;
    MonitorElement* meResolxMFClusterwidthProfileCategory1Rphi;
    MonitorElement* meResolxMFAngleProfileRphi;
    MonitorElement* merapidityResProfilewclus1;
    MonitorElement* merapidityResProfilewclus2;
    MonitorElement* merapidityResProfilewclus3;
    MonitorElement* merapidityResProfilewclus4;
    

  };

  struct StereoAndMatchedMEs{ // MEs for stereo and matched hits
      
    MonitorElement* meWclusSas;
    MonitorElement* meAdcSas;
    MonitorElement* meResolxLFSas;
    MonitorElement* meResolxMFSas;
    MonitorElement* meResLFSas;
    MonitorElement* meResMFSas;
    MonitorElement* mePullLFSas;
    MonitorElement* mePullMFSas;
    MonitorElement* meTrackangleSas;
    MonitorElement* meTrackanglebetaSas;
    MonitorElement* mePullTrackangleProfileSas;
    MonitorElement* meTrackwidthSas;
    MonitorElement* meExpectedwidthSas;
    MonitorElement* meClusterwidthSas;
    MonitorElement* meCategorySas;
    MonitorElement* mePullTrackwidthProfileSas;
    MonitorElement* mePullTrackwidthProfileCategory1Sas;
    MonitorElement* mePullTrackwidthProfileCategory2Sas;
    MonitorElement* mePullTrackwidthProfileCategory3Sas;
    MonitorElement* mePullTrackwidthProfileCategory4Sas;
    MonitorElement* meResolxMFTrackwidthProfileSas;
    MonitorElement* meResolxMFTrackwidthProfileCategory1Sas;
    MonitorElement* meResolxMFTrackwidthProfileCategory2Sas;
    MonitorElement* meResolxMFTrackwidthProfileCategory3Sas;
    MonitorElement* meResolxMFTrackwidthProfileCategory4Sas;
    MonitorElement* meResolxMFClusterwidthProfileCategory1Sas;
    MonitorElement* meResolxMFAngleProfileSas;

    MonitorElement* mePosxMatched;
    MonitorElement* mePosyMatched;
    MonitorElement* meResolxMatched;
    MonitorElement* meResolyMatched;
    MonitorElement* meResxMatched;
    MonitorElement* meResyMatched;
    MonitorElement* mePullxMatched;
    MonitorElement* mePullyMatched;

  };

  struct RecHitProperties{ 
    float x;
    float y;
    float z;
    float resolxx; 
    float resolxy; 
    float resolyy; 
    float resolxxMF; // in Measurement Frame
    float phi;
    float resx;
    float resy;
    float resxMF;// in Measurement Frame
    float pullx;
    float pully;
    float pullxMF;// in Measurement Frame
    float trackangle;
    float trackanglebeta;
    float trackangle2;
    float trackwidth;
    int   expectedwidth;
    int   category;
    float  thickness;
    int   clusiz;
    float cluchg;
  };

 protected:

  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  void bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es);
  const MagneticField * magfield2_ ;
  void beginJob(const edm::EventSetup& es);
  void endJob();

 private:
  
  DQMStore* dbe_;
  bool runStandalone;
  bool outputMEsInRootFile;
  std::string outputFileName;
  
  std::string topFolderName_;
  
 
  bool layerswitchResolx_LF;
  bool layerswitchResolx_MF;
  bool layerswitchRes_LF;
  bool layerswitchRes_MF;
  bool layerswitchPull_LF;
  bool layerswitchPull_MF;
  bool layerswitchCategory;
  bool layerswitchTrackwidth;
  bool layerswitchExpectedwidth;
  bool layerswitchClusterwidth;
  bool layerswitchTrackanglealpha;
  bool layerswitchTrackanglebeta;
  bool layerswitchResolxMFTrackwidthProfile_WClus1;
  bool layerswitchResolxMFTrackwidthProfile_WClus2;
  bool layerswitchResolxMFTrackwidthProfile_WClus3;
  bool layerswitchResolxMFTrackwidthProfile_WClus4;
  bool layerswitchResMFTrackwidthProfile_WClus1;
  bool layerswitchResMFTrackwidthProfile_WClus2;
  bool layerswitchResMFTrackwidthProfile_WClus21;
  bool layerswitchResMFTrackwidthProfile_WClus22;
  bool layerswitchResMFTrackwidthProfile_WClus23;
  bool layerswitchResMFTrackwidthProfile_WClus3;
  bool layerswitchResMFTrackwidthProfile_WClus4;
  bool layerswitchResolxMFTrackwidthProfile;
  bool layerswitchResolxMFTrackwidthProfile_Category1;
  bool layerswitchResolxMFTrackwidthProfile_Category2;
  bool layerswitchResolxMFTrackwidthProfile_Category3;
  bool layerswitchResolxMFTrackwidthProfile_Category4;
  bool layerswitchResolxMFClusterwidthProfile_Category1;
  bool layerswitchResolxMFAngleProfile;
  bool layerswitchWclusRphi;
  bool layerswitchAdcRphi;
  bool layerswitchResolxLFRphi;
  bool layerswitchResolxMFRphi;
  bool layerswitchResolxMFRphiwclus1 ;
  bool layerswitchResolxMFRphiwclus2 ;
  bool layerswitchResolxMFRphiwclus3 ;
  bool layerswitchResolxMFRphiwclus4 ;
  bool layerswitchResLFRphi;
  bool layerswitchResMFRphi;
  bool layerswitchResMFRphiwclus1;
  bool layerswitchResMFRphiwclus2;
  bool layerswitchResMFRphiwclus3;
  bool layerswitchResMFRphiwclus4;
  bool layerswitchPullLFRphi;
  bool layerswitchPullMFRphi;
  bool layerswitchPullMFRphiwclus1;
  bool layerswitchPullMFRphiwclus2;
  bool layerswitchPullMFRphiwclus3;
  bool layerswitchPullMFRphiwclus4;
  bool layerswitchTrackangleRphi;
  bool layerswitchTrackanglebetaRphi;
  bool layerswitchTrackangle2Rphi;
  bool layerswitchPullTrackangleProfileRphi;
  bool layerswitchPullTrackangle2DRphi;
  bool layerswitchTrackwidthRphi;
  bool layerswitchExpectedwidthRphi;
  bool layerswitchClusterwidthRphi;
  bool layerswitchCategoryRphi;
  bool layerswitchPullTrackwidthProfileRphi;
  bool layerswitchPullTrackwidthProfileRphiwclus1;
  bool layerswitchPullTrackwidthProfileRphiwclus2;
  bool layerswitchPullTrackwidthProfileRphiwclus3;
  bool layerswitchPullTrackwidthProfileRphiwclus4;
  bool layerswitchPullTrackwidthProfileCategory1Rphi;
  bool layerswitchPullTrackwidthProfileCategory2Rphi;
  bool layerswitchPullTrackwidthProfileCategory3Rphi;
  bool layerswitchPullTrackwidthProfileCategory4Rphi;
  bool layerswitchResolxMFTrackwidthProfileRphi;
  bool layerswitchResolxMFTrackwidthProfileWclus1Rphi;
  bool layerswitchResolxMFTrackwidthProfileWclus2Rphi;
  bool layerswitchResolxMFTrackwidthProfileWclus3Rphi;
  bool layerswitchResolxMFTrackwidthProfileWclus4Rphi;
  bool layerswitchResMFTrackwidthProfileWclus1Rphi;
  bool layerswitchResMFTrackwidthProfileWclus2Rphi;
  bool layerswitchResMFTrackwidthProfileWclus3Rphi;
  bool layerswitchResMFTrackwidthProfileWclus4Rphi;
  bool layerswitchResolxMFTrackwidthProfileCategory1Rphi;
  bool layerswitchResolxMFTrackwidthProfileCategory2Rphi;
  bool layerswitchResolxMFTrackwidthProfileCategory3Rphi;
  bool layerswitchResolxMFTrackwidthProfileCategory4Rphi;
  bool layerswitchResolxMFAngleProfileRphi;
  bool layerswitchResolxMFClusterwidthProfileCategory1Rphi;
  bool layerswitchrapidityResProfilewclus1;
  bool layerswitchrapidityResProfilewclus2;
  bool layerswitchrapidityResProfilewclus3;
  bool layerswitchrapidityResProfilewclus4;
  bool layerswitchWclusSas;
  bool layerswitchAdcSas;
  bool layerswitchResolxLFSas;
  bool layerswitchResolxMFSas;
  bool layerswitchResLFSas;
  bool layerswitchResMFSas;
  bool layerswitchPullLFSas;
  bool layerswitchPullMFSas;
  bool layerswitchTrackangleSas;
  bool layerswitchTrackanglebetaSas;
  bool layerswitchPullTrackangleProfileSas;
  bool layerswitchTrackwidthSas;
  bool layerswitchExpectedwidthSas;
  bool layerswitchClusterwidthSas;
  bool layerswitchCategorySas;
  bool layerswitchPullTrackwidthProfileSas;
  bool layerswitchPullTrackwidthProfileCategory1Sas;
  bool layerswitchPullTrackwidthProfileCategory2Sas;
  bool layerswitchPullTrackwidthProfileCategory3Sas;
  bool layerswitchPullTrackwidthProfileCategory4Sas;
  bool layerswitchResolxMFTrackwidthProfileSas;
  bool layerswitchResolxMFTrackwidthProfileCategory1Sas;
  bool layerswitchResolxMFTrackwidthProfileCategory2Sas;
  bool layerswitchResolxMFTrackwidthProfileCategory3Sas;
  bool layerswitchResolxMFTrackwidthProfileCategory4Sas;
  bool layerswitchResolxMFAngleProfileSas;
  bool layerswitchResolxMFClusterwidthProfileCategory1Sas;
  bool layerswitchPosxMatched;
  bool layerswitchPosyMatched;
  bool layerswitchResolxMatched;
  bool layerswitchResolyMatched;
  bool layerswitchResxMatched;
  bool layerswitchResyMatched;
  bool layerswitchPullxMatched;
  bool layerswitchPullyMatched;

  SimpleHitsMEs simplehitsMEs;
  std::vector<PSimHit> matched;
  std::map<std::string, LayerMEs> LayerMEsMap;
  std::map<std::string, StereoAndMatchedMEs> StereoAndMatchedMEsMap;
  std::map<std::string, std::vector< uint32_t > > LayerDetMap;
  std::map<std::string, std::vector< uint32_t > > StereoAndMatchedDetMap;

  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;

  std::pair<LocalPoint,LocalVector> projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,const BoundPlane& plane);

  LocalVector driftDirection(const StripGeomDetUnit* det)const;

  MonitorElement* Fit_SliceY(TH2F * Histo2D);

  void createMEs(DQMStore::IBooker & ibooker,const edm::EventSetup& es);
  void createSimpleHitsMEs(DQMStore::IBooker & ibooker); 
  void createLayerMEs(DQMStore::IBooker & ibooker,std::string label);
  void createStereoAndMatchedMEs(DQMStore::IBooker & ibooker,std::string label);
  
  MonitorElement* bookME1D(DQMStore::IBooker & ibooker,const char* ParameterSetLabel, const char* HistoName, const char* HistoTitle);
  MonitorElement* bookMEProfile(DQMStore::IBooker & ibooker,const char* ParameterSetLabel, const char* HistoName, const char* HistoTitle);

  inline void fillME(MonitorElement* ME,float value1){if (ME!=0)ME->Fill(value1);}
  inline void fillME(MonitorElement* ME,float value1,float value2){if (ME!=0)ME->Fill(value1,value2);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3){if (ME!=0)ME->Fill(value1,value2,value3);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3,float value4){if (ME!=0)ME->Fill(value1,value2,value3,value4);}
  

  edm::ParameterSet conf_;
  unsigned long long m_cacheID_;
  edm::ParameterSet Parameters;

  //const StripTopology* topol;
  std::vector<RecHitProperties> rechitrphi;
  std::vector<RecHitProperties> rechitstereo;
  std::vector<RecHitProperties> rechitmatched;
  RecHitProperties rechitpro;

  void rechitanalysis(TrajectoryStateOnSurface tsos, const TransientTrackingRecHit::ConstRecHitPointer thit, const StripGeomDetUnit *stripdet, edm::ESHandle < StripClusterParameterEstimator > stripcpe, TrackerHitAssociator& associate,  bool simplehit1or2D);
  
  void rechitanalysis_matched(TrajectoryStateOnSurface tsos, const TransientTrackingRecHit::ConstRecHitPointer thit, const GluedGeomDet* gluedDet,TrackerHitAssociator& associate, edm::ESHandle < StripClusterParameterEstimator > stripcpe, std::string matchedmonorstereo);
 

  float track_rapidity;
  //edm::InputTag trajectoryInput_;
  edm::EDGetTokenT<std::vector<Trajectory> > trajectoryInputToken_;

};


#endif
