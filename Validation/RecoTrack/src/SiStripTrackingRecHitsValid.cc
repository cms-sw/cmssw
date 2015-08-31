#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "Validation/RecoTrack/interface/SiStripTrackingRecHitsValid.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"

using namespace std;
using namespace edm;

// ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1.h"
#include "TH2.h"
class TFile;

//Constructor
SiStripTrackingRecHitsValid::SiStripTrackingRecHitsValid(const edm::ParameterSet& ps) : 
  dbe_(edm::Service<DQMStore>().operator->()),	
  conf_(ps),
  trackerHitAssociatorConfig_(ps, consumesCollector()),
  m_cacheID_(0)
  // trajectoryInput_( ps.getParameter<edm::InputTag>("trajectoryInput") )
{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");

  runStandalone = conf_.getParameter<bool>("runStandalone");

  outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");

  outputFileName = conf_.getParameter<std::string>("outputFile");
  
  trajectoryInputToken_ = consumes<std::vector<Trajectory> >( conf_.getParameter<edm::InputTag>("trajectoryInput") ); 

  edm::ParameterSet ParametersResolx_LF =  conf_.getParameter<edm::ParameterSet>("TH1Resolx_LF");
  layerswitchResolx_LF = ParametersResolx_LF.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolx_MF =  conf_.getParameter<edm::ParameterSet>("TH1Resolx_MF");
  layerswitchResolx_MF = ParametersResolx_MF.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersRes_LF =  conf_.getParameter<edm::ParameterSet>("TH1Res_LF");
  layerswitchRes_LF = ParametersRes_LF.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersRes_MF =  conf_.getParameter<edm::ParameterSet>("TH1Res_MF");
  layerswitchRes_MF = ParametersRes_MF.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPull_LF =  conf_.getParameter<edm::ParameterSet>("TH1Pull_LF");
  layerswitchPull_LF = ParametersPull_LF.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPull_MF =  conf_.getParameter<edm::ParameterSet>("TH1Pull_MF");
  layerswitchPull_MF = ParametersPull_MF.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersCategory =  conf_.getParameter<edm::ParameterSet>("TH1Category");
  layerswitchCategory = ParametersCategory.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackwidth =  conf_.getParameter<edm::ParameterSet>("TH1Trackwidth");
  layerswitchTrackwidth = ParametersTrackwidth.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersExpectedwidth =  conf_.getParameter<edm::ParameterSet>("TH1Expectedwidth");
  layerswitchExpectedwidth = ParametersExpectedwidth.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterwidth =  conf_.getParameter<edm::ParameterSet>("TH1Clusterwidth");
  layerswitchClusterwidth = ParametersClusterwidth.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackanglealpha =  conf_.getParameter<edm::ParameterSet>("TH1Trackanglealpha");
  layerswitchTrackanglealpha = ParametersTrackanglealpha.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackanglebeta =  conf_.getParameter<edm::ParameterSet>("TH1Trackanglebeta");
  layerswitchTrackanglebeta = ParametersTrackanglebeta.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile_WClus1 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile_WClus1");
  layerswitchResolxMFTrackwidthProfile_WClus1 = ParametersResolxMFTrackwidthProfile_WClus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile_WClus2 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile_WClus2");
  layerswitchResolxMFTrackwidthProfile_WClus2 = ParametersResolxMFTrackwidthProfile_WClus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile_WClus3 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile_WClus3");
  layerswitchResolxMFTrackwidthProfile_WClus3 = ParametersResolxMFTrackwidthProfile_WClus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile_WClus4 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile_WClus4");
  layerswitchResolxMFTrackwidthProfile_WClus4 = ParametersResolxMFTrackwidthProfile_WClus4.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfile_WClus1 =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfile_WClus1");
  layerswitchResMFTrackwidthProfile_WClus1 = ParametersResMFTrackwidthProfile_WClus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfile_WClus2 =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfile_WClus2");
  layerswitchResMFTrackwidthProfile_WClus2 = ParametersResMFTrackwidthProfile_WClus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfile_WClus21 =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfile_WClus21");
  layerswitchResMFTrackwidthProfile_WClus21 = ParametersResMFTrackwidthProfile_WClus21.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfile_WClus22 =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfile_WClus22");
  layerswitchResMFTrackwidthProfile_WClus22 = ParametersResMFTrackwidthProfile_WClus22.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfile_WClus23 =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfile_WClus23");
  layerswitchResMFTrackwidthProfile_WClus23 = ParametersResMFTrackwidthProfile_WClus23.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfile_WClus3 =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfile_WClus3");
  layerswitchResMFTrackwidthProfile_WClus3 = ParametersResMFTrackwidthProfile_WClus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfile_WClus4 =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfile_WClus4");
  layerswitchResMFTrackwidthProfile_WClus4 = ParametersResMFTrackwidthProfile_WClus4.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile");
  layerswitchResolxMFTrackwidthProfile = ParametersResolxMFTrackwidthProfile.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile_Category1 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile_Category1");
  layerswitchResolxMFTrackwidthProfile_Category1 = ParametersResolxMFTrackwidthProfile_Category1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile_Category2 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile_Category2");
  layerswitchResolxMFTrackwidthProfile_Category2 = ParametersResolxMFTrackwidthProfile_Category2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile_Category3 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile_Category3");
  layerswitchResolxMFTrackwidthProfile_Category3 = ParametersResolxMFTrackwidthProfile_Category3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfile_Category4 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfile_Category4");
  layerswitchResolxMFTrackwidthProfile_Category4 = ParametersResolxMFTrackwidthProfile_Category4.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFClusterwidthProfile_Category1 =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFClusterwidthProfile_Category1");
  layerswitchResolxMFClusterwidthProfile_Category1 = ParametersResolxMFClusterwidthProfile_Category1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFAngleProfile =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFAngleProfile");
  layerswitchResolxMFAngleProfile = ParametersResolxMFAngleProfile.getParameter<bool>("layerswitchon");
  
  edm::ParameterSet ParametersWclusRphi =  conf_.getParameter<edm::ParameterSet>("TH1WclusRphi");
  layerswitchWclusRphi = ParametersWclusRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersAdcRphi =  conf_.getParameter<edm::ParameterSet>("TH1AdcRphi");
  layerswitchAdcRphi = ParametersAdcRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxLFRphi =  conf_.getParameter<edm::ParameterSet>("TH1ResolxLFRphi");
  layerswitchResolxLFRphi = ParametersResolxLFRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFRphi =  conf_.getParameter<edm::ParameterSet>("TH1ResolxMFRphi");
  layerswitchResolxMFRphi = ParametersResolxMFRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFRphiwclus1 =  conf_.getParameter<edm::ParameterSet>("TH1ResolxMFRphiwclus1");
  layerswitchResolxMFRphiwclus1 = ParametersResolxMFRphiwclus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFRphiwclus2 =  conf_.getParameter<edm::ParameterSet>("TH1ResolxMFRphiwclus2");
  layerswitchResolxMFRphiwclus2 = ParametersResolxMFRphiwclus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFRphiwclus3 =  conf_.getParameter<edm::ParameterSet>("TH1ResolxMFRphiwclus3");
  layerswitchResolxMFRphiwclus3 = ParametersResolxMFRphiwclus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFRphiwclus4 =  conf_.getParameter<edm::ParameterSet>("TH1ResolxMFRphiwclus4");
  layerswitchResolxMFRphiwclus4 = ParametersResolxMFRphiwclus4.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResLFRphi =  conf_.getParameter<edm::ParameterSet>("TH1ResLFRphi");
  layerswitchResLFRphi = ParametersResLFRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFRphi =  conf_.getParameter<edm::ParameterSet>("TH1ResMFRphi");
  layerswitchResMFRphi = ParametersResMFRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFRphiwclus1 =  conf_.getParameter<edm::ParameterSet>("TH1ResMFRphiwclus1");
  layerswitchResMFRphiwclus1 = ParametersResMFRphiwclus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFRphiwclus2 =  conf_.getParameter<edm::ParameterSet>("TH1ResMFRphiwclus2");
  layerswitchResMFRphiwclus2 = ParametersResMFRphiwclus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFRphiwclus3 =  conf_.getParameter<edm::ParameterSet>("TH1ResMFRphiwclus3");
  layerswitchResMFRphiwclus3 = ParametersResMFRphiwclus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFRphiwclus4 =  conf_.getParameter<edm::ParameterSet>("TH1ResMFRphiwclus4");
  layerswitchResMFRphiwclus4 = ParametersResMFRphiwclus4.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullLFRphi =  conf_.getParameter<edm::ParameterSet>("TH1PullLFRphi");
  layerswitchPullLFRphi = ParametersPullLFRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullMFRphi =  conf_.getParameter<edm::ParameterSet>("TH1PullMFRphi");
  layerswitchPullMFRphi = ParametersPullMFRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullMFRphiwclus1 =  conf_.getParameter<edm::ParameterSet>("TH1PullMFRphiwclus1");
  layerswitchPullMFRphiwclus1 = ParametersPullMFRphiwclus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullMFRphiwclus2 =  conf_.getParameter<edm::ParameterSet>("TH1PullMFRphiwclus2");
  layerswitchPullMFRphiwclus2 = ParametersPullMFRphiwclus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullMFRphiwclus3 =  conf_.getParameter<edm::ParameterSet>("TH1PullMFRphiwclus3");
  layerswitchPullMFRphiwclus3 = ParametersPullMFRphiwclus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullMFRphiwclus4 =  conf_.getParameter<edm::ParameterSet>("TH1PullMFRphiwclus4");
  layerswitchPullMFRphiwclus4 = ParametersPullMFRphiwclus4.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackangleRphi =  conf_.getParameter<edm::ParameterSet>("TH1TrackangleRphi");
  layerswitchTrackangleRphi = ParametersTrackangleRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackanglebetaRphi =  conf_.getParameter<edm::ParameterSet>("TH1TrackanglebetaRphi");
  layerswitchTrackanglebetaRphi = ParametersTrackanglebetaRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackangle2Rphi =  conf_.getParameter<edm::ParameterSet>("TH1Trackangle2Rphi");
  layerswitchTrackangle2Rphi = ParametersTrackangle2Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackangleProfileRphi =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackangleProfileRphi");
  layerswitchPullTrackangleProfileRphi = ParametersPullTrackangleProfileRphi.getParameter<bool>("layerswitchon");
  
  edm::ParameterSet ParametersPullTrackangle2DRphi =  conf_.getParameter<edm::ParameterSet>("TH1PullTrackangle2DRphi");
  layerswitchPullTrackangle2DRphi = ParametersPullTrackangle2DRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackwidthRphi =  conf_.getParameter<edm::ParameterSet>("TH1TrackwidthRphi");
  layerswitchTrackwidthRphi = ParametersTrackwidthRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersExpectedwidthRphi =  conf_.getParameter<edm::ParameterSet>("TH1ExpectedwidthRphi");
  layerswitchExpectedwidthRphi = ParametersExpectedwidthRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterwidthRphi =  conf_.getParameter<edm::ParameterSet>("TH1ClusterwidthRphi");
  layerswitchClusterwidthRphi = ParametersClusterwidthRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersCategoryRphi =  conf_.getParameter<edm::ParameterSet>("TH1CategoryRphi");
  layerswitchCategoryRphi = ParametersCategoryRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileRphi =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileRphi");
  layerswitchPullTrackwidthProfileRphi = ParametersPullTrackwidthProfileRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileRphiwclus1 =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileRphiwclus1");
  layerswitchPullTrackwidthProfileRphiwclus1 = ParametersPullTrackwidthProfileRphiwclus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileRphiwclus2 =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileRphiwclus2");
  layerswitchPullTrackwidthProfileRphiwclus2 = ParametersPullTrackwidthProfileRphiwclus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileRphiwclus3 =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileRphiwclus3");
  layerswitchPullTrackwidthProfileRphiwclus3 = ParametersPullTrackwidthProfileRphiwclus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileRphiwclus4 =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileRphiwclus4");
  layerswitchPullTrackwidthProfileRphiwclus4 = ParametersPullTrackwidthProfileRphiwclus4.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileCategory1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileCategory1Rphi");
  layerswitchPullTrackwidthProfileCategory1Rphi = ParametersPullTrackwidthProfileCategory1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileCategory2Rphi =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileCategory2Rphi");
  layerswitchPullTrackwidthProfileCategory2Rphi = ParametersPullTrackwidthProfileCategory2Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileCategory3Rphi =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileCategory3Rphi");
  layerswitchPullTrackwidthProfileCategory3Rphi = ParametersPullTrackwidthProfileCategory3Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileCategory4Rphi =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileCategory4Rphi");
  layerswitchPullTrackwidthProfileCategory4Rphi = ParametersPullTrackwidthProfileCategory4Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileRphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileRphi");
  layerswitchResolxMFTrackwidthProfileRphi = ParametersResolxMFTrackwidthProfileRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileWclus1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileWclus1Rphi");
  layerswitchResolxMFTrackwidthProfileWclus1Rphi = ParametersResolxMFTrackwidthProfileWclus1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileWclus2Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileWclus2Rphi");
  layerswitchResolxMFTrackwidthProfileWclus2Rphi = ParametersResolxMFTrackwidthProfileWclus2Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileWclus3Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileWclus3Rphi");
  layerswitchResolxMFTrackwidthProfileWclus3Rphi = ParametersResolxMFTrackwidthProfileWclus3Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileWclus4Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileWclus4Rphi");
  layerswitchResolxMFTrackwidthProfileWclus4Rphi = ParametersResolxMFTrackwidthProfileWclus4Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfileWclus1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfileWclus1Rphi");
  layerswitchResMFTrackwidthProfileWclus1Rphi = ParametersResMFTrackwidthProfileWclus1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfileWclus2Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfileWclus2Rphi");
  layerswitchResMFTrackwidthProfileWclus2Rphi = ParametersResMFTrackwidthProfileWclus2Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfileWclus3Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfileWclus3Rphi");
  layerswitchResMFTrackwidthProfileWclus3Rphi = ParametersResMFTrackwidthProfileWclus3Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfileWclus4Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfileWclus4Rphi");
  layerswitchResMFTrackwidthProfileWclus4Rphi = ParametersResMFTrackwidthProfileWclus4Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileCategory1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileCategory1Rphi");
  layerswitchResolxMFTrackwidthProfileCategory1Rphi = ParametersResolxMFTrackwidthProfileCategory1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileCategory2Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileCategory2Rphi");
  layerswitchResolxMFTrackwidthProfileCategory2Rphi = ParametersResolxMFTrackwidthProfileCategory2Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileCategory3Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileCategory3Rphi");
  layerswitchResolxMFTrackwidthProfileCategory3Rphi = ParametersResolxMFTrackwidthProfileCategory3Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileCategory4Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileCategory4Rphi");
  layerswitchResolxMFTrackwidthProfileCategory4Rphi = ParametersResolxMFTrackwidthProfileCategory4Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFAngleProfileRphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFAngleProfileRphi");
  layerswitchResolxMFAngleProfileRphi = ParametersResolxMFAngleProfileRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFClusterwidthProfileCategory1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFClusterwidthProfileCategory1Rphi");
  layerswitchResolxMFClusterwidthProfileCategory1Rphi = ParametersResolxMFClusterwidthProfileCategory1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersrapidityResProfilewclus1 =  conf_.getParameter<edm::ParameterSet>("TProfrapidityResProfilewclus1");
  layerswitchrapidityResProfilewclus1 = ParametersrapidityResProfilewclus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersrapidityResProfilewclus2 =  conf_.getParameter<edm::ParameterSet>("TProfrapidityResProfilewclus2");
  layerswitchrapidityResProfilewclus2 = ParametersrapidityResProfilewclus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersrapidityResProfilewclus3 =  conf_.getParameter<edm::ParameterSet>("TProfrapidityResProfilewclus3");
  layerswitchrapidityResProfilewclus3 = ParametersrapidityResProfilewclus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersrapidityResProfilewclus4 =  conf_.getParameter<edm::ParameterSet>("TProfrapidityResProfilewclus4");
  layerswitchrapidityResProfilewclus4 = ParametersrapidityResProfilewclus4.getParameter<bool>("layerswitchon");
  
  edm::ParameterSet ParametersWclusSas =  conf_.getParameter<edm::ParameterSet>("TH1WclusSas");
  layerswitchWclusSas = ParametersWclusSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersAdcSas =  conf_.getParameter<edm::ParameterSet>("TH1AdcSas");
  layerswitchAdcSas = ParametersAdcSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxLFSas =  conf_.getParameter<edm::ParameterSet>("TH1ResolxLFSas");
  layerswitchResolxLFSas = ParametersResolxLFSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFSas =  conf_.getParameter<edm::ParameterSet>("TH1ResolxMFSas");
  layerswitchResolxMFSas = ParametersResolxMFSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResLFSas =  conf_.getParameter<edm::ParameterSet>("TH1ResLFSas");
  layerswitchResLFSas = ParametersResLFSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFSas =  conf_.getParameter<edm::ParameterSet>("TH1ResMFSas");
  layerswitchResMFSas = ParametersResMFSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullLFSas =  conf_.getParameter<edm::ParameterSet>("TH1PullLFSas");
  layerswitchPullLFSas = ParametersPullLFSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullMFSas =  conf_.getParameter<edm::ParameterSet>("TH1PullMFSas");
  layerswitchPullMFSas = ParametersPullMFSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackangleSas =  conf_.getParameter<edm::ParameterSet>("TH1TrackangleSas");
  layerswitchTrackangleSas = ParametersTrackangleSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackanglebetaSas =  conf_.getParameter<edm::ParameterSet>("TH1TrackanglebetaSas");
  layerswitchTrackanglebetaSas = ParametersTrackanglebetaSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackangleProfileSas =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackangleProfileSas");
  layerswitchPullTrackangleProfileSas = ParametersPullTrackangleProfileSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTrackwidthSas =  conf_.getParameter<edm::ParameterSet>("TH1TrackwidthSas");
  layerswitchTrackwidthSas = ParametersTrackwidthSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersExpectedwidthSas =  conf_.getParameter<edm::ParameterSet>("TH1ExpectedwidthSas");
  layerswitchExpectedwidthSas = ParametersExpectedwidthSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterwidthSas =  conf_.getParameter<edm::ParameterSet>("TH1ClusterwidthSas");
  layerswitchClusterwidthSas = ParametersClusterwidthSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersCategorySas =  conf_.getParameter<edm::ParameterSet>("TH1CategorySas");
  layerswitchCategorySas = ParametersCategorySas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileSas =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileSas");
  layerswitchPullTrackwidthProfileSas = ParametersPullTrackwidthProfileSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileCategory1Sas =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileCategory1Sas");
  layerswitchPullTrackwidthProfileCategory1Sas = ParametersPullTrackwidthProfileCategory1Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileCategory2Sas =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileCategory2Sas");
  layerswitchPullTrackwidthProfileCategory2Sas = ParametersPullTrackwidthProfileCategory2Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileCategory3Sas =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileCategory3Sas");
  layerswitchPullTrackwidthProfileCategory3Sas = ParametersPullTrackwidthProfileCategory3Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullTrackwidthProfileCategory4Sas =  conf_.getParameter<edm::ParameterSet>("TProfPullTrackwidthProfileCategory4Sas");
  layerswitchPullTrackwidthProfileCategory4Sas = ParametersPullTrackwidthProfileCategory4Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileSas =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileSas");
  layerswitchResolxMFTrackwidthProfileSas = ParametersResolxMFTrackwidthProfileSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileCategory1Sas =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileCategory1Sas");
  layerswitchResolxMFTrackwidthProfileCategory1Sas = ParametersResolxMFTrackwidthProfileCategory1Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileCategory2Sas =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileCategory2Sas");
  layerswitchResolxMFTrackwidthProfileCategory2Sas = ParametersResolxMFTrackwidthProfileCategory2Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileCategory3Sas =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileCategory3Sas");
  layerswitchResolxMFTrackwidthProfileCategory3Sas = ParametersResolxMFTrackwidthProfileCategory3Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFTrackwidthProfileCategory4Sas =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFTrackwidthProfileCategory4Sas");
  layerswitchResolxMFTrackwidthProfileCategory4Sas = ParametersResolxMFTrackwidthProfileCategory4Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFAngleProfileSas =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFAngleProfileSas");
  layerswitchResolxMFAngleProfileSas = ParametersResolxMFAngleProfileSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMFClusterwidthProfileCategory1Sas =  conf_.getParameter<edm::ParameterSet>("TProfResolxMFClusterwidthProfileCategory1Sas");
  layerswitchResolxMFClusterwidthProfileCategory1Sas = ParametersResolxMFClusterwidthProfileCategory1Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPosxMatched =  conf_.getParameter<edm::ParameterSet>("TH1PosxMatched");
  layerswitchPosxMatched = ParametersPosxMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPosyMatched =  conf_.getParameter<edm::ParameterSet>("TH1PosyMatched");
  layerswitchPosyMatched = ParametersPosyMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolxMatched =  conf_.getParameter<edm::ParameterSet>("TH1ResolxMatched");
  layerswitchResolxMatched = ParametersResolxMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResolyMatched =  conf_.getParameter<edm::ParameterSet>("TH1ResolyMatched");
  layerswitchResolyMatched = ParametersResolyMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResxMatched =  conf_.getParameter<edm::ParameterSet>("TH1ResxMatched");
  layerswitchResxMatched = ParametersResxMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResyMatched =  conf_.getParameter<edm::ParameterSet>("TH1ResyMatched");
  layerswitchResyMatched = ParametersResyMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullxMatched =  conf_.getParameter<edm::ParameterSet>("TH1PullxMatched");
  layerswitchPullxMatched = ParametersPullxMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPullyMatched =  conf_.getParameter<edm::ParameterSet>("TH1PullyMatched");
  layerswitchPullyMatched = ParametersPullyMatched.getParameter<bool>("layerswitchon");

}

//Destructor
SiStripTrackingRecHitsValid::~SiStripTrackingRecHitsValid(){
}
//--------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es){

  unsigned long long cacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
    edm::LogInfo("SiStripRecHitsValid") <<"SiStripRecHitsValid::beginRun: " 
					<< " Creating MEs for new Cabling ";     
    
    createMEs(ibooker,es);
  }
}


void SiStripTrackingRecHitsValid::beginJob(const edm::EventSetup& es){
  
}

void SiStripTrackingRecHitsValid::endJob() {

  //Only in standalone mode save local root file 
  if(runStandalone && outputMEsInRootFile){dbe_->save(outputFileName);}

}

// Functions that gets called by framework every event
void SiStripTrackingRecHitsValid::analyze(const edm::Event & e, const edm::EventSetup & es)
{

  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();  
  //cout  << " Run = " << e.id().run() << " Event = " << e.id().event() << endl;  

  int isrechitrphi = 0;
  int isrechitsas = 0;
  int isrechitmatched = 0;

  DetId detid;
  uint32_t myid;

  TrackerHitAssociator associate(e, trackerHitAssociatorConfig_);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::ESHandle < TrackerGeometry > pDD;
  es.get < TrackerDigiGeometryRecord > ().get(pDD);
  const TrackerGeometry & tracker(*pDD);

  const TrackerGeometry *tracker2;
  edm::ESHandle < TrackerGeometry > estracker;
  es.get < TrackerDigiGeometryRecord > ().get(estracker);
  tracker2 = &(*estracker);

  edm::ESHandle < MagneticField > magfield;
  es.get < IdealMagneticFieldRecord > ().get(magfield);

  const MagneticField & magfield_(*magfield);
  magfield2_ = &magfield_;

  edm::ESHandle < StripClusterParameterEstimator > stripcpe;
  es.get < TkStripCPERecord > ().get("SimpleStripCPE", stripcpe);

  // Mangano's

  edm::Handle < std::vector<Trajectory> > trajCollectionHandle;
  e.getByToken(trajectoryInputToken_, trajCollectionHandle);

  edm::LogVerbatim("TrajectoryAnalyzer") << "trajColl->size(): " << trajCollectionHandle->size();

  for (vector < Trajectory >::const_iterator it = trajCollectionHandle->begin(), itEnd = trajCollectionHandle->end(); it !=itEnd ; ++it) {

    edm::LogVerbatim("TrajectoryAnalyzer") << "this traj has " << it->foundHits() << " valid hits" << " , " << "isValid: " << it->isValid();

    vector < TrajectoryMeasurement > tmColl = it->measurements();
 
    for (auto const &itTraj : tmColl ) {
      if (itTraj.updatedState().isValid()) {
        //edm::LogVerbatim("TrajectoryAnalyzer") << "tm number: " <<
        //   (itTraj - tmColl.begin()) + 1<< " , " << "tm.backwardState.pt: " <<
        //   itTraj->backwardPredictedState().globalMomentum().perp() << " , " <<
        //   "tm.forwardState.pt:  " << itTraj->forwardPredictedState().globalMomentum().perp() <<
        //   " , " << "tm.updatedState.pt:  " << itTraj->updatedState().globalMomentum().perp() <<
        //   " , " << "tm.globalPos.perp: "   << itTraj->updatedState().globalPosition().perp();

        if (itTraj.updatedState().globalMomentum().perp() < 0.5)	continue;
      
        TrajectoryStateOnSurface tsos = itTraj.updatedState();

        const TransientTrackingRecHit::ConstRecHitPointer thit2 = itTraj.recHit();
        DetId detid2 = thit2->geographicalId();
        const SiStripMatchedRecHit2D *matchedhit = dynamic_cast < const SiStripMatchedRecHit2D * >((*thit2).hit());
        const SiStripRecHit2D *hit2d = dynamic_cast < const SiStripRecHit2D * >((*thit2).hit());
        const SiStripRecHit1D *hit1d = dynamic_cast < const SiStripRecHit1D * >((*thit2).hit());
        //if(matchedhit) cout<<"manganomatchedhit"<<endl;
        //if(hit) cout<<"manganosimplehit"<<endl;
        //if (hit && matchedhit) cout<<"manganosimpleandmatchedhit"<<endl;
        const TrackingRecHit *thit = (*thit2).hit();

        detid = (thit)->geographicalId();
        myid = detid.rawId();
        //Here due to the fact that the SiStripHistoId::getSubdetid complains when 
        //a subdet of 1 or 2 appears we add an if statement. 
        if(detid.subdetId()==1 ||detid.subdetId()==2 ){
	  continue;
        }
        SiStripHistoId hidmanager;
        std::string label = hidmanager.getSubdetid(myid,tTopo,true);
        // std::cout<< "label " << label << " and id " << detid.subdetId() << std::endl;

        StripSubdetector StripSubdet = (StripSubdetector) detid;
        //Variable to define the case we are dealing with
        std::string matchedmonorstereo;

        isrechitmatched = 0;
  
        if (matchedhit) {
  	  isrechitmatched = 1;
	  const GluedGeomDet *gluedDet = (const GluedGeomDet *) tracker.idToDet(matchedhit->geographicalId());
	  //Analysis
	  rechitanalysis_matched(tsos, thit2, gluedDet, associate, stripcpe, MatchStatus::matched);
	  // rechitmatched.push_back(rechitpro);
        }

        std::map<std::string, StereoAndMatchedMEs>::iterator iStereoAndMatchedME  = StereoAndMatchedMEsMap.find(label);

        //Filling Histograms for Matched hits

        if (isrechitmatched) {

	  if(iStereoAndMatchedME != StereoAndMatchedMEsMap.end()){
	    fillME(iStereoAndMatchedME->second.mePosxMatched,rechitpro.x);
	    fillME(iStereoAndMatchedME->second.mePosyMatched,rechitpro.y);
	    fillME(iStereoAndMatchedME->second.meResolxMatched,sqrt(rechitpro.resolxx));
	    fillME(iStereoAndMatchedME->second.meResolyMatched,sqrt(rechitpro.resolyy));
	    fillME(iStereoAndMatchedME->second.meResxMatched,rechitpro.resx);
	    fillME(iStereoAndMatchedME->second.meResyMatched,rechitpro.resy);
	    fillME(iStereoAndMatchedME->second.mePullxMatched,rechitpro.pullx);
	    fillME(iStereoAndMatchedME->second.mePullyMatched,rechitpro.pully);
	  }
        }
    
        //Reset Variables here for the current event
        isrechitrphi    = 0;
        isrechitsas     = 0;
     
        ///////////////////////////////////////////////////////
        // simple hits from matched hits
        ///////////////////////////////////////////////////////
 
        if (tsos.globalDirection().transverse() != 0) {
	  track_rapidity = tsos.globalDirection().eta();
        } else {
	  track_rapidity = -999.0;
        }

        if (matchedhit) {
          auto hm = matchedhit->monoHit();
	  const SiStripRecHit2D *monohit = &hm;
	  //      const GeomDetUnit * monodet=gdet->monoDet();
	  GluedGeomDet *gdet = (GluedGeomDet *) tracker2->idToDet(matchedhit->geographicalId());
	  
	  if (monohit) {

	    isrechitrphi = 1;
	  
	    //Analysis
	    rechitanalysis_matched(tsos, thit2, gdet, associate, stripcpe, MatchStatus::monoHit);

	  }

	  auto s = matchedhit->stereoHit();
	  const SiStripRecHit2D *stereohit = &s;
	
	  if (stereohit) {
	
	    isrechitsas = 1;
	  
	    //Analysis
	    rechitanalysis_matched(tsos, thit2, gdet, associate, stripcpe, MatchStatus::stereoHit);
	  }
        }
      
        if (hit1d) {
	  // simple hits are mono or stereo
	  //      cout<<"simple hit"<<endl; 
	  if (StripSubdet.stereo()) {

	    //cout<<"simple hit stereo"<<endl;
	    isrechitsas = 1;

	    const GeomDetUnit *det = tracker.idToDetUnit(detid2);
	    const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (det);

	    //Analysis for hit1d stereo
	    rechitanalysis(tsos, thit2, stripdet, stripcpe, associate, true);
	  } else {
	    isrechitrphi = 1;
	    //      cout<<"simple hit mono"<<endl;

	    const GeomDetUnit *det = tracker.idToDetUnit(detid2);
	    const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (det);
	  
	    //Analysis for hit1d mono
	    rechitanalysis(tsos, thit2, stripdet, stripcpe, associate, true);

	  }
        }
    
        if (hit2d) {
	  // simple hits are mono or stereo
	  //      cout<<"simple hit"<<endl;
	  if (StripSubdet.stereo()) {

	    //cout<<"simple hit stereo"<<endl;
	    isrechitsas = 1;

	    const GeomDetUnit *det = tracker.idToDetUnit(detid2);
	    const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (det);

	    //Analysis for hit2d stereo
	    rechitanalysis(tsos, thit2, stripdet, stripcpe, associate, false);

	  } else {
	    isrechitrphi = 1;
	    //      cout<<"simple hit mono"<<endl;

	    const GeomDetUnit *det = tracker.idToDetUnit(detid2);
	    const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (det);

	    //Analysis for hit2d mono
	    rechitanalysis(tsos, thit2, stripdet, stripcpe, associate, false);

	  }
        }

//------------------------------------------------------------------------------------------------------------------------------------------------------

        //Filling Histograms for simple hits
        //cout<<"isrechitrphi,isrechitsas = "<<isrechitrphi<<","<<isrechitsas<<endl;

        std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(label);
        if (isrechitrphi) {
          fillME(simplehitsMEs.meCategory,rechitpro.category);
	  fillME(simplehitsMEs.meTrackwidth,rechitpro.trackwidth);
	  fillME(simplehitsMEs.meExpectedwidth,rechitpro.expectedwidth);
	  fillME(simplehitsMEs.meClusterwidth,rechitpro.clusiz);
	  fillME(simplehitsMEs.meTrackanglealpha,rechitpro.trackangle);
	  fillME(simplehitsMEs.meTrackanglebeta,rechitpro.trackanglebeta);

	  fillME(simplehitsMEs.meResolxMFAngleProfile,rechitpro.trackangle, sqrt(rechitpro.resolxxMF));
	  fillME(simplehitsMEs.meResolxMFTrackwidthProfile,rechitpro.trackwidth, sqrt(rechitpro.resolxxMF));

	  if (rechitpro.clusiz == 1) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileWClus1,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus1,rechitpro.trackwidth, fabs(rechitpro.resxMF));
	  } else if (rechitpro.clusiz == 2) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileWClus2,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus2,rechitpro.trackwidth, fabs(rechitpro.resxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus21,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus22,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus23,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	  } else if (rechitpro.clusiz == 3) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileWClus3,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus3,rechitpro.trackwidth, fabs(rechitpro.resxMF));
	  } else if (rechitpro.clusiz == 4) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileWClus4,rechitpro.trackwidth, sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus4,rechitpro.trackwidth, fabs(rechitpro.resxMF));
	  }

	  if (rechitpro.category == 1) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileCategory1,rechitpro.trackwidth, sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResolxMFClusterwidthProfileCategory1,rechitpro.clusiz, sqrt(rechitpro.resolxxMF));
	  } else if (rechitpro.category == 2) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileCategory2,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	  } else if (rechitpro.category == 3) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileCategory3,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	  } else if (rechitpro.category == 4) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileCategory4,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	  }

	  fillME(simplehitsMEs.meResolxMF,sqrt(rechitpro.resolxxMF));
	  fillME(simplehitsMEs.meResolxLF,sqrt(rechitpro.resolxx));
	  fillME(simplehitsMEs.meResMF,rechitpro.resxMF);
	  fillME(simplehitsMEs.meResLF,rechitpro.resx);
	  fillME(simplehitsMEs.mePullMF,rechitpro.pullxMF);
	  fillME(simplehitsMEs.mePullLF,rechitpro.pullx);

          if(iLayerME != LayerMEsMap.end()){
	    fillME(iLayerME->second.meWclusRphi,rechitpro.clusiz);
	    fillME(iLayerME->second.meAdcRphi,rechitpro.cluchg);
	    fillME(iLayerME->second.meResolxLFRphi,sqrt(rechitpro.resolxx));
	    fillME(iLayerME->second.meResolxMFRphi,sqrt(rechitpro.resolxxMF));

	    if( (min(rechitpro.clusiz, 4) - 1) == 1 ){fillME(iLayerME->second.meResolxMFRphiwclus1,sqrt(rechitpro.resolxxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 2 ){fillME(iLayerME->second.meResolxMFRphiwclus2,sqrt(rechitpro.resolxxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 3 ){fillME(iLayerME->second.meResolxMFRphiwclus3,sqrt(rechitpro.resolxxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 4 ){fillME(iLayerME->second.meResolxMFRphiwclus4,sqrt(rechitpro.resolxxMF));}


	    fillME(iLayerME->second.meResLFRphi,rechitpro.resx);
	    fillME(iLayerME->second.meResMFRphi,rechitpro.resxMF);

	    if( (min(rechitpro.clusiz, 4) - 1) == 1 ){fillME(iLayerME->second.merapidityResProfilewclus1,track_rapidity, fabs(rechitpro.resxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 2 ){fillME(iLayerME->second.merapidityResProfilewclus2,track_rapidity, fabs(rechitpro.resxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 3 ){fillME(iLayerME->second.merapidityResProfilewclus3,track_rapidity, fabs(rechitpro.resxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 4 ){fillME(iLayerME->second.merapidityResProfilewclus4,track_rapidity, fabs(rechitpro.resxMF));}

	    if( (min(rechitpro.clusiz, 4) - 1) == 1 ){fillME(iLayerME->second.meResMFRphiwclus1,rechitpro.resxMF);}
	    if( (min(rechitpro.clusiz, 4) - 1) == 2 ){fillME(iLayerME->second.meResMFRphiwclus2,rechitpro.resxMF);}
	    if( (min(rechitpro.clusiz, 4) - 1) == 3 ){fillME(iLayerME->second.meResMFRphiwclus3,rechitpro.resxMF);}
	    if( (min(rechitpro.clusiz, 4) - 1) == 4 ){fillME(iLayerME->second.meResMFRphiwclus4,rechitpro.resxMF);}

	    fillME(iLayerME->second.mePullLFRphi,rechitpro.pullx);
	    fillME(iLayerME->second.mePullMFRphi,rechitpro.pullxMF);
	    
	    if( (min(rechitpro.clusiz, 4) - 1) == 1 ){fillME(iLayerME->second.mePullMFRphiwclus1,rechitpro.pullxMF);}
	    if( (min(rechitpro.clusiz, 4) - 1) == 2 ){fillME(iLayerME->second.mePullMFRphiwclus2,rechitpro.pullxMF);}
	    if( (min(rechitpro.clusiz, 4) - 1) == 3 ){fillME(iLayerME->second.mePullMFRphiwclus3,rechitpro.pullxMF);}
	    if( (min(rechitpro.clusiz, 4) - 1) == 4 ){fillME(iLayerME->second.mePullMFRphiwclus4,rechitpro.pullxMF);}


	    fillME(iLayerME->second.meTrackangleRphi,rechitpro.trackangle);
	    fillME(iLayerME->second.mePullTrackangleProfileRphi,rechitpro.trackangle,fabs(rechitpro.pullxMF));

	    if( (min(rechitpro.clusiz, 4) - 1) == 1 ){fillME(iLayerME->second.mePullTrackwidthProfileRphiwclus1,rechitpro.trackwidth, fabs(rechitpro.pullxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 2 ){fillME(iLayerME->second.mePullTrackwidthProfileRphiwclus2,rechitpro.trackwidth, fabs(rechitpro.pullxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 3 ){fillME(iLayerME->second.mePullTrackwidthProfileRphiwclus3,rechitpro.trackwidth, fabs(rechitpro.pullxMF));}
	    if( (min(rechitpro.clusiz, 4) - 1) == 4 ){fillME(iLayerME->second.mePullTrackwidthProfileRphiwclus4,rechitpro.trackwidth, fabs(rechitpro.pullxMF));}


	    if (rechitpro.clusiz == 1) {
	      fillME(iLayerME->second.meResolxMFTrackwidthProfileWclus1Rphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	      fillME(iLayerME->second.meResMFTrackwidthProfileWclus1Rphi,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	    }
	    if (rechitpro.clusiz == 2) {
	      fillME(iLayerME->second.meResolxMFTrackwidthProfileWclus2Rphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	      fillME(iLayerME->second.meResMFTrackwidthProfileWclus2Rphi,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	    }
	    if (rechitpro.clusiz == 3) {
	      fillME(iLayerME->second.meResolxMFTrackwidthProfileWclus3Rphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	      fillME(iLayerME->second.meResMFTrackwidthProfileWclus3Rphi,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	    }
	    if (rechitpro.clusiz == 4) {
	      fillME(iLayerME->second.meResolxMFTrackwidthProfileWclus4Rphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	      fillME(iLayerME->second.meResMFTrackwidthProfileWclus4Rphi,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	    }


	    if (rechitpro.category == 1) {
	      fillME(iLayerME->second.mePullTrackwidthProfileCategory1Rphi,rechitpro.trackwidth,fabs(rechitpro.pullxMF));
	      fillME(iLayerME->second.meResolxMFTrackwidthProfileCategory1Rphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	      fillME(iLayerME->second.meResolxMFClusterwidthProfileCategory1Rphi,rechitpro.clusiz,sqrt(rechitpro.resolxxMF));
	    } else if(rechitpro.category == 2) {
	      fillME(iLayerME->second.mePullTrackwidthProfileCategory2Rphi,rechitpro.trackwidth,fabs(rechitpro.pullxMF));
	      fillME(iLayerME->second.meResolxMFTrackwidthProfileCategory2Rphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    } else if (rechitpro.category == 3) {
	      fillME(iLayerME->second.mePullTrackwidthProfileCategory3Rphi,rechitpro.trackwidth,fabs(rechitpro.pullxMF));
	      fillME(iLayerME->second.meResolxMFTrackwidthProfileCategory3Rphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    } else if (rechitpro.category == 4) {
	      fillME(iLayerME->second.mePullTrackwidthProfileCategory4Rphi,rechitpro.trackwidth,fabs(rechitpro.pullxMF));
	      fillME(iLayerME->second.meResolxMFTrackwidthProfileCategory4Rphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    }
	    
	    fillME(iLayerME->second.meTrackwidthRphi,rechitpro.trackwidth);
	    fillME(iLayerME->second.meExpectedwidthRphi,rechitpro.expectedwidth);
	    fillME(iLayerME->second.meClusterwidthRphi,rechitpro.clusiz);
	    fillME(iLayerME->second.meCategoryRphi,rechitpro.category);
	    fillME(iLayerME->second.meResolxMFTrackwidthProfileRphi,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    fillME(iLayerME->second.meResolxMFAngleProfileRphi,rechitpro.trackangle,sqrt(rechitpro.resolxxMF));
	  }
        }

        if (isrechitsas > 0) {
          fillME(simplehitsMEs.meCategory,rechitpro.category);
	  fillME(simplehitsMEs.meTrackwidth,rechitpro.trackwidth);
	  fillME(simplehitsMEs.meExpectedwidth,rechitpro.expectedwidth);
	  fillME(simplehitsMEs.meClusterwidth,rechitpro.clusiz);
	  fillME(simplehitsMEs.meTrackanglealpha,rechitpro.trackangle);
	  fillME(simplehitsMEs.meTrackanglebeta,rechitpro.trackanglebeta);

	  fillME(simplehitsMEs.meResolxMFAngleProfile,rechitpro.trackangle, sqrt(rechitpro.resolxxMF));
	  fillME(simplehitsMEs.meResolxMFTrackwidthProfile,rechitpro.trackwidth, sqrt(rechitpro.resolxxMF));

	  if (rechitpro.clusiz == 1) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileWClus1,rechitpro.trackwidth, sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus1,rechitpro.trackwidth, rechitpro.resxMF);
	  } else if (rechitpro.clusiz == 2) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileWClus2,rechitpro.trackwidth, sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus2,rechitpro.trackwidth, rechitpro.resxMF);
	  } else if (rechitpro.clusiz == 3) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileWClus3,rechitpro.trackwidth, sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus3,rechitpro.trackwidth, rechitpro.resxMF);
	  } else if (rechitpro.clusiz == 4) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileWClus4,rechitpro.trackwidth, sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus4,rechitpro.trackwidth, rechitpro.resxMF);
	  } if (rechitpro.category == 1) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileCategory1,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    fillME(simplehitsMEs.meResolxMFClusterwidthProfileCategory1,rechitpro.clusiz, sqrt(rechitpro.resolxxMF));
	  } else if (rechitpro.category == 2) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileCategory2,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	  } else if (rechitpro.category == 3) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileCategory3,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	  } else if (rechitpro.category == 4) {
	    fillME(simplehitsMEs.meResolxMFTrackwidthProfileCategory4,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	  }

	  fillME(simplehitsMEs.meResolxMF,sqrt(rechitpro.resolxxMF));
	  fillME(simplehitsMEs.meResolxLF,sqrt(rechitpro.resolxx));
	  fillME(simplehitsMEs.meResMF,rechitpro.resxMF);
	  fillME(simplehitsMEs.meResLF,rechitpro.resx);
	  fillME(simplehitsMEs.mePullMF,rechitpro.pullxMF);
	  fillME(simplehitsMEs.mePullLF,rechitpro.pullx);

	  if(iStereoAndMatchedME != StereoAndMatchedMEsMap.end()){
	    fillME(iStereoAndMatchedME->second.meWclusSas,rechitpro.clusiz);
	    fillME(iStereoAndMatchedME->second.meAdcSas,rechitpro.cluchg);
	    fillME(iStereoAndMatchedME->second.meResolxLFSas,sqrt(rechitpro.resolxx));
	    fillME(iStereoAndMatchedME->second.meResLFSas,rechitpro.resx);
	    fillME(iStereoAndMatchedME->second.mePullLFSas,rechitpro.pullx);
	    fillME(iStereoAndMatchedME->second.meResolxMFSas,sqrt(rechitpro.resolxxMF));
	    fillME(iStereoAndMatchedME->second.meResMFSas,rechitpro.resxMF);
	    fillME(iStereoAndMatchedME->second.mePullMFSas,rechitpro.pullxMF);
	    fillME(iStereoAndMatchedME->second.meTrackangleSas,rechitpro.trackangle);
	    fillME(iStereoAndMatchedME->second.mePullTrackangleProfileSas,rechitpro.trackangle, rechitpro.pullxMF);
	    fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileSas,rechitpro.trackwidth, rechitpro.pullxMF);
	    if (rechitpro.category == 1) {
	      fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileCategory1Sas,rechitpro.trackwidth,rechitpro.pullxMF);
	      fillME(iStereoAndMatchedME->second.meResolxMFTrackwidthProfileCategory1Sas,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	      fillME(iStereoAndMatchedME->second.meResolxMFClusterwidthProfileCategory1Sas,rechitpro.clusiz,sqrt(rechitpro.resolxxMF));
	    } else if (rechitpro.category == 2) {
	      fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileCategory2Sas,rechitpro.trackwidth,rechitpro.pullxMF);
	      fillME(iStereoAndMatchedME->second.meResolxMFTrackwidthProfileCategory2Sas,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    } else if (rechitpro.category == 3) {
	      fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileCategory3Sas,rechitpro.trackwidth,rechitpro.pullxMF);
	      fillME(iStereoAndMatchedME->second.meResolxMFTrackwidthProfileCategory3Sas,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    } else if (rechitpro.category == 4) {
	      fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileCategory4Sas,rechitpro.trackwidth,rechitpro.pullxMF);
	      fillME(iStereoAndMatchedME->second.meResolxMFTrackwidthProfileCategory4Sas,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    }
	    fillME(iStereoAndMatchedME->second.meTrackwidthSas,rechitpro.trackwidth);
	    fillME(iStereoAndMatchedME->second.meExpectedwidthSas,rechitpro.expectedwidth);
	    fillME(iStereoAndMatchedME->second.meClusterwidthSas,rechitpro.clusiz);
	    fillME(iStereoAndMatchedME->second.meCategorySas,rechitpro.category);
	    fillME(iStereoAndMatchedME->second.meResolxMFTrackwidthProfileSas,rechitpro.trackwidth,sqrt(rechitpro.resolxxMF));
	    fillME(iStereoAndMatchedME->second.meResolxMFAngleProfileSas,rechitpro.trackangle, rechitpro.resolxxMF);
	  }
        } 
      }
    }
  }
}



//needed by to do the residual for matched hits
std::pair < LocalPoint, LocalVector > SiStripTrackingRecHitsValid::projectHit(const PSimHit & hit,
                                                                              const StripGeomDetUnit
                                                                              * stripDet,
                                                                              const BoundPlane &
                                                                              plane)
{
  //  const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(hit.det());
  //if (stripDet == 0) throw MeasurementDetException("HitMatcher hit is not on StripGeomDetUnit");

  const StripTopology & topol = stripDet->specificTopology();
  GlobalPoint globalpos = stripDet->surface().toGlobal(hit.localPosition());
  LocalPoint localHit = plane.toLocal(globalpos);
  //track direction
  LocalVector locdir = hit.localDirection();
  //rotate track in new frame

  GlobalVector globaldir = stripDet->surface().toGlobal(locdir);
  LocalVector dir = plane.toLocal(globaldir);
  float scale = -localHit.z() / dir.z();

  LocalPoint projectedPos = localHit + scale * dir;

  //  std::cout << "projectedPos " << projectedPos << std::endl;

  float selfAngle = topol.stripAngle(topol.strip(hit.localPosition()));

  LocalVector stripDir(sin(selfAngle), cos(selfAngle), 0);     // vector along strip in hit frame

  LocalVector localStripDir(plane.toLocal(stripDet->surface().toGlobal(stripDir)));

  return std::pair < LocalPoint, LocalVector > (projectedPos, localStripDir);
}
//--------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::rechitanalysis_matched(TrajectoryStateOnSurface tsos, const TransientTrackingRecHit::ConstRecHitPointer thit, const GluedGeomDet* gluedDet, TrackerHitAssociator&  associate, edm::ESHandle<StripClusterParameterEstimator> stripcpe, const MatchStatus matchedmonorstereo){
  
  rechitpro.resx = -999999.; rechitpro.resy = -999999.; rechitpro.resxMF = -999999.; 
  rechitpro.pullx = -999999.; rechitpro.pully = -999999.; rechitpro.pullxMF = -999999.; rechitpro.trackangle = -999999.; rechitpro.trackanglebeta = -999999.;

  const GeomDetUnit *monodet = gluedDet->monoDet(); 
  const GeomDetUnit *stereodet = gluedDet->stereoDet();
  //We initialized it to monoHit case because it complains that it may be uninitialized
  //and it will change value in the stereoHit case. The matched case do not use this
  const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (monodet) ; 

  const TrackingRecHit *rechit = (*thit).hit();
  const SiStripMatchedRecHit2D *matchedhit = dynamic_cast < const SiStripMatchedRecHit2D * > (rechit);
  const SiStripRecHit2D *monohit = nullptr;
  const SiStripRecHit2D *stereohit = nullptr;
  SiStripRecHit2D::ClusterRef clust;
 
  if (matchedmonorstereo == MatchStatus::monoHit){
    auto hm = matchedhit->monoHit();
    monohit = &hm;
    stripdet = (const StripGeomDetUnit *) (monodet);
  } else if (matchedmonorstereo == MatchStatus::stereoHit){
    auto s = matchedhit->stereoHit();
    stereohit = &s;
    stripdet = (const StripGeomDetUnit *) (stereodet);
  }
  //if(matchedhit) cout<<"manganomatchedhit"<<endl;
  //if(hit) cout<<"manganosimplehit"<<endl;
  //if (hit && matchedhit) cout<<"manganosimpleandmatchedhit"<<endl;
  const StripTopology & topol = (const StripTopology &) stripdet->topology();

  LocalVector trackdirection = tsos.localDirection();

  GlobalVector gtrkdir = gluedDet->toGlobal(trackdirection);
  LocalVector monotkdir = monodet->toLocal(gtrkdir);
  LocalVector stereotkdir = stereodet->toLocal(gtrkdir);
 
  LocalPoint position;
  LocalError error;
  MeasurementPoint Mposition;
  MeasurementError Merror;

  if (matchedmonorstereo == MatchStatus::matched){
    position=rechit->localPosition();
    error=rechit->localPositionError();
  } else if(matchedmonorstereo == MatchStatus::monoHit){
    position = monohit->localPosition();
    error = monohit->localPositionError();
    Mposition = topol.measurementPosition(position);
    Merror = topol.measurementError(position, error);
    if (monotkdir.z()) {
      rechitpro.trackangle = atan(monotkdir.x() / monotkdir.z()) * TMath::RadToDeg();
      rechitpro.trackanglebeta = atan(monotkdir.y() / monotkdir.z()) * TMath::RadToDeg();
    }
    clust = monohit->cluster(); 
  } else if (matchedmonorstereo == MatchStatus::stereoHit){
    position = stereohit->localPosition();
    error = stereohit->localPositionError();
    Mposition = topol.measurementPosition(position);
    Merror = topol.measurementError(position, error);
    if (stereotkdir.z()) {
      rechitpro.trackangle = atan(stereotkdir.x() / stereotkdir.z()) * TMath::RadToDeg();
      rechitpro.trackanglebeta = atan(stereotkdir.y() / stereotkdir.z()) * TMath::RadToDeg();
    }
    clust = stereohit->cluster();
  }

  LocalVector drift = stripcpe->driftDirection(stripdet);
  rechitpro.thickness = stripdet->surface().bounds().thickness();
  float pitch = topol.localPitch(position);
  float tanalpha = tan(rechitpro.trackangle * TMath::DegToRad());
  float tanalphaL = drift.x() / drift.z();
  rechitpro.trackwidth = fabs((rechitpro.thickness / pitch) * tanalpha - (rechitpro.thickness / pitch) * tanalphaL);
  float SLorentz = 0.5 * (rechitpro.thickness / pitch) * tanalphaL;
  int Sp = int (position.x() / pitch + SLorentz + 0.5 * rechitpro.trackwidth);
  int Sm = int (position.x() / pitch + SLorentz - 0.5 * rechitpro.trackwidth);
  rechitpro.expectedwidth = 1 + Sp - Sm;

  const auto & amplitudes=clust->amplitudes();
  rechitpro.clusiz = amplitudes.size();
  int totcharge=0;
  for(size_t ia=0; ia<amplitudes.size();++ia){
    totcharge+=amplitudes[ia];
  }

  rechitpro.x = position.x();
  rechitpro.y = position.y();
  rechitpro.z = position.z();
  rechitpro.resolxx = error.xx();
  rechitpro.resolxy = error.xy();
  rechitpro.resolyy = error.yy();
  rechitpro.resolxxMF = Merror.uu();
  rechitpro.cluchg = totcharge;
 
  if (rechitpro.clusiz > rechitpro.expectedwidth + 2) {
    rechitpro.category = 1;
  } else if (rechitpro.expectedwidth == 1) {
    rechitpro.category = 2;
  } else if (rechitpro.clusiz <= rechitpro.expectedwidth) {
    rechitpro.category = 3;
  } else {
    rechitpro.category = 4;
  }

  if(matchedmonorstereo == MatchStatus::matched){matched.clear();matched = associate.associateHit(*matchedhit);}
  else if(matchedmonorstereo == MatchStatus::monoHit){matched.clear();matched = associate.associateHit(*monohit);}
  else if(matchedmonorstereo == MatchStatus::stereoHit){matched.clear();matched = associate.associateHit(*stereohit);}

  if(!matched.empty()){
    float mindist = std::numeric_limits<float>::max();
    float dist = std::numeric_limits<float>::max();
    float distx = std::numeric_limits<float>::max();
    float disty = std::numeric_limits<float>::max();
    std::pair<LocalPoint,LocalVector> closestPair;
    PSimHit* closest = NULL;

    const StripGeomDetUnit* partnerstripdet = static_cast<const StripGeomDetUnit*>(gluedDet->stereoDet());
    std::pair<LocalPoint,LocalVector> hitPair;
   
    if (matchedmonorstereo ==MatchStatus::matched) {
      for(auto &m : matched){
        //project simhit;
 	hitPair= projectHit(m,partnerstripdet,gluedDet->surface());
	distx = fabs(rechitpro.x - hitPair.first.x());
	disty = fabs(rechitpro.y - hitPair.first.y());
	dist = sqrt(distx*distx+disty*disty);
	if(dist<mindist){
	  mindist = dist;
	  closestPair = hitPair;
	  closest = &m;
        }
      }
      float closestX = closestPair.first.x();
      float closestY = closestPair.first.y();
      rechitpro.resx = rechitpro.x - closestX;
      rechitpro.resy = rechitpro.y - closestPair.first.y();
      rechitpro.pullx = ((rechit)->localPosition().x() - closestX) / sqrt(error.xx());
      rechitpro.pully = ((rechit)->localPosition().y() - closestY) / sqrt(error.yy());
    } else if(matchedmonorstereo == MatchStatus::monoHit){
      for(auto &m : matched ){
        //project simhit;
        dist = abs((monohit)->localPosition().x() - m.localPosition().x());
        if(dist<mindist){
	  mindist = dist;
	  closestPair = hitPair;
	  closest = &m;
        }
      }
      float closestX = closest->localPosition().x();
      rechitpro.resx = rechitpro.x - closestX;
      rechitpro.resxMF = Mposition.x() - (topol.measurementPosition(closest->localPosition())).x();
      rechitpro.pullx = (rechit->localPosition().x() - closestX) / sqrt(error.xx());
      rechitpro.pullxMF = (rechitpro.resxMF)/sqrt(Merror.uu());
     } else if(matchedmonorstereo == MatchStatus::stereoHit){
      for(auto &m : matched){
        //project simhit;
        dist = abs((stereohit)->localPosition().x() - m.localPosition().x());
        if(dist<mindist){
	  mindist = dist;
	  closestPair = hitPair;
	  closest = &m;
        }
      }
      float closestX =  closest->localPosition().x();
      rechitpro.resx = rechitpro.x - closestX;
      rechitpro.resxMF = Mposition.x() - (topol.measurementPosition(closest->localPosition())).x();
      rechitpro.pullx = (rechit->localPosition().x() - closestX) / sqrt(error.xx());
      rechitpro.pullxMF = (rechitpro.resxMF)/sqrt(Merror.uu());
    }
  }
}
//--------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::rechitanalysis(TrajectoryStateOnSurface tsos, const TransientTrackingRecHit::ConstRecHitPointer thit, const StripGeomDetUnit *stripdet,edm::ESHandle<StripClusterParameterEstimator> stripcpe, TrackerHitAssociator& associate,  bool simplehit1or2D){

  rechitpro.resx = -999999.; rechitpro.resy = -999999.; rechitpro.resxMF = -999999.; 
  rechitpro.pullx = -999999.; rechitpro.pully = -999999.; rechitpro.pullxMF = -999999.;
  
  //If simplehit1or2D is true we are dealing with hit1d, false is for hit2d
  const SiStripRecHit2D *hit2d = dynamic_cast < const SiStripRecHit2D * >((*thit).hit());;
  const SiStripRecHit1D *hit1d = dynamic_cast < const SiStripRecHit1D * >((*thit).hit());;

  const StripTopology & topol = (const StripTopology &) stripdet->topology();
  const TrackingRecHit *rechit = (*thit).hit();

  LocalPoint position = rechit->localPosition();
  LocalError error = rechit->localPositionError();
  MeasurementPoint Mposition = topol.measurementPosition(position);
  MeasurementError Merror = topol.measurementError(position,error);
 
  LocalVector trackdirection = tsos.localDirection();
  rechitpro.trackangle = atan(trackdirection.x() / trackdirection.z()) * TMath::RadToDeg();
  rechitpro.trackanglebeta = atan(trackdirection.y() / trackdirection.z()) * TMath::RadToDeg();

  LocalVector drift = stripcpe->driftDirection(stripdet);
  rechitpro.thickness = stripdet->surface().bounds().thickness();
  float pitch = topol.localPitch(position);
  float tanalpha = tan(rechitpro.trackangle * TMath::DegToRad());
  float tanalphaL = drift.x() / drift.z();
  rechitpro.trackwidth = fabs((rechitpro.thickness / pitch) * tanalpha - (rechitpro.thickness / pitch) * tanalphaL);
  float SLorentz = 0.5 * (rechitpro.thickness / pitch) * tanalphaL;
  int Sp = int (position.x() / pitch + SLorentz + 0.5 * rechitpro.trackwidth);
  int Sm = int (position.x() / pitch + SLorentz - 0.5 * rechitpro.trackwidth);
  rechitpro.expectedwidth = 1 + Sp - Sm;

  int totcharge=0;
  if(simplehit1or2D){ 
    SiStripRecHit1D::ClusterRef clust1d;
    clust1d = hit1d->cluster();
    const auto & amplitudes1d = clust1d->amplitudes();
    rechitpro.clusiz = amplitudes1d.size();
    for(size_t ia=0; ia<amplitudes1d.size();++ia){
      totcharge+=amplitudes1d[ia];
    }
  } else {    
    SiStripRecHit2D::ClusterRef clust2d;
    clust2d = hit2d->cluster();
    const auto & amplitudes2d = clust2d->amplitudes();
    rechitpro.clusiz = amplitudes2d.size();
    for(size_t ia=0; ia<amplitudes2d.size();++ia){
      totcharge+=amplitudes2d[ia];
    }
  }

  rechitpro.x = position.x();
  rechitpro.y = position.y();
  rechitpro.z = position.z();
  rechitpro.resolxx = error.xx();
  rechitpro.resolxy = error.xy();
  rechitpro.resolyy = error.yy();
  rechitpro.resolxxMF = Merror.uu();
  rechitpro.cluchg = totcharge;

  if (rechitpro.clusiz > rechitpro.expectedwidth + 2) {
    rechitpro.category = 1;
  } else if (rechitpro.expectedwidth == 1) {
    rechitpro.category = 2;
  } else if (rechitpro.clusiz <= rechitpro.expectedwidth) {
    rechitpro.category = 3;
  } else {
    rechitpro.category = 4;
  }

  matched.clear();
  float mindist = std::numeric_limits<float>::max();
  float dist = std::numeric_limits<float>::max();
  PSimHit* closest = NULL;
 
  if(simplehit1or2D){
    matched = associate.associateHit(*hit1d);
    if(!matched.empty()){
      for(auto &m : matched ){
        dist = abs((hit1d)->localPosition().x() - m.localPosition().x());
        if(dist<mindist){
	  mindist = dist;
	  closest = &m;
        }
      } 
      float closestX = closest->localPosition().x();
      rechitpro.resx = rechitpro.x - closestX;
      rechitpro.resxMF = Mposition.x() - (topol.measurementPosition(closest->localPosition())).x();
      rechitpro.pullx = (rechit->localPosition().x() - closestX) / sqrt(error.xx());
      rechitpro.pullxMF = (rechitpro.resxMF)/sqrt(Merror.uu());
    }
  } else {
    matched = associate.associateHit(*hit2d);
    if(!matched.empty()){
      for(auto &m : matched ){
        dist = abs((hit2d)->localPosition().x() - m.localPosition().x());
        if(dist<mindist){
	  mindist = dist;
	  closest = &m;
        }
      }
      float closestX = closest->localPosition().x(); 
      rechitpro.resx = rechitpro.x - closestX;
      rechitpro.resxMF = Mposition.x() - (topol.measurementPosition(closest->localPosition())).x();
      rechitpro.pullx = (rechit->localPosition().x() - closestX) / sqrt(error.xx());
      rechitpro.pullxMF = (rechitpro.resxMF)/sqrt(Merror.uu());
    }
  }
}
//--------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::createMEs(DQMStore::IBooker & ibooker,const edm::EventSetup& es){

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  
  // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
  es.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);
    
  // get list of active detectors from SiStripDetCabling 
  std::vector<uint32_t> activeDets;
  SiStripDetCabling_->addActiveDetectorsRawIds(activeDets);
    
  SiStripSubStructure substructure;

  SiStripFolderOrganizer folder_organizer;
  // folder_organizer.setSiStripFolderName(topFolderName_);
  std::string curfold = topFolderName_;
  folder_organizer.setSiStripFolderName(curfold);
  folder_organizer.setSiStripFolder();

  // std::cout << "curfold " << curfold << std::endl;

  createSimpleHitsMEs(ibooker);

  // loop over detectors and book MEs
  edm::LogInfo("SiStripTrackingRecHitsValid|SiStripTrackingRecHitsValid")<<"nr. of activeDets:  "<<activeDets.size();
  const std::string& tec = "TEC", tid = "TID", tob = "TOB", tib = "TIB";
  for(std::vector<uint32_t>::iterator detid_iterator = activeDets.begin(), detid_end = activeDets.end(); detid_iterator!=detid_end; ++detid_iterator){
    uint32_t detid = (*detid_iterator);
    // remove any eventual zero elements - there should be none, but just in case
    if(detid == 0) {
      activeDets.erase(detid_iterator);
      continue;
    }
    
    // Create Layer Level MEs
    std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid,tTopo,true);
    SiStripHistoId hidmanager;
    std::string label = hidmanager.getSubdetid(detid,tTopo,true);
    // std::cout << "label " << label << endl;
      
    std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(label);
    if(iLayerME==LayerMEsMap.end()) {
	
      // get detids for the layer
      // Keep in mind that when we are on the TID or TEC we deal with rings not wheel 
      int32_t lnumber = det_layer_pair.second;
      const std::string& lname = det_layer_pair.first;
      std::vector<uint32_t> layerDetIds;        
      if (lname.compare(tec) == 0) { 
        if (lnumber > 0) {
	  substructure.getTECDetectors(activeDets,layerDetIds,2,0,0,0,abs(lnumber),0);
        } else if (lnumber < 0) {
	  substructure.getTECDetectors(activeDets,layerDetIds,1,0,0,0,abs(lnumber),0);
        }
      } else if (lname.compare(tid) == 0) {
        if (lnumber > 0) {
	  substructure.getTIDDetectors(activeDets,layerDetIds,2,0,abs(lnumber),0);
        } else if (lnumber < 0) {
	  substructure.getTIDDetectors(activeDets,layerDetIds,1,0,abs(lnumber),0);
        }
      } else if (lname.compare(tob) == 0) {
	substructure.getTOBDetectors(activeDets,layerDetIds,lnumber,0,0);
      } else if (lname.compare(tib) == 0) {
	substructure.getTIBDetectors(activeDets,layerDetIds,lnumber,0,0,0);
      }
      LayerDetMap[label] = layerDetIds;

      // book Layer MEs 
      folder_organizer.setLayerFolder(detid,tTopo,det_layer_pair.second,true);
      // std::stringstream ss;
      // folder_organizer.getLayerFolderName(ss, detid, tTopo, true); 
      // std::cout << "Folder Name " << ss.str().c_str() << std::endl;
      // folder_organizer.setLayerFolder(detid,det_layer_pair.second,true);
      createLayerMEs(ibooker,label);
    }
    //Create StereoAndMatchedMEs
    std::map<std::string, StereoAndMatchedMEs>::iterator iStereoAndMatchedME  = StereoAndMatchedMEsMap.find(label);
    if(iStereoAndMatchedME==StereoAndMatchedMEsMap.end()) {
	
      // get detids for the stereo and matched layer. We are going to need a bool for these layers
      bool isStereo = false;
      // Keep in mind that when we are on the TID or TEC we deal with rings not wheel 
      std::vector<uint32_t> stereoandmatchedDetIds;        
      int32_t stereolnumber = det_layer_pair.second;
      const std::string& stereolname = det_layer_pair.first;
      if ( stereolname.compare(tec) == 0 && (tTopo->tecIsStereo(detid)) ) {
        if ( stereolnumber > 0 ) {
          substructure.getTECDetectors(activeDets,stereoandmatchedDetIds,2,0,0,0,abs(stereolnumber),1);
          isStereo = true;
        } else if ( stereolnumber < 0 ) {
          substructure.getTECDetectors(activeDets,stereoandmatchedDetIds,1,0,0,0,abs(stereolnumber),1);
          isStereo = true;
        }
      } else if ( stereolname.compare(tid) == 0 && (tTopo->tidIsStereo(detid)) ) {
        if ( stereolnumber > 0 ) {
          substructure.getTIDDetectors(activeDets,stereoandmatchedDetIds,2,0,abs(stereolnumber),1);
          isStereo = true;
        } else if ( stereolnumber < 0 ) {
          substructure.getTIDDetectors(activeDets,stereoandmatchedDetIds,1,0,abs(stereolnumber),1);
          isStereo = true;
        }
      } else if ( stereolname.compare(tob) == 0 && (tTopo->tobIsStereo(detid)) ) {
        substructure.getTOBDetectors(activeDets,stereoandmatchedDetIds,stereolnumber,0,0);
        isStereo = true;
      } else if ( stereolname.compare(tib) == 0 && (tTopo->tibIsStereo(detid)) ) {
        substructure.getTIBDetectors(activeDets,stereoandmatchedDetIds,stereolnumber,0,0,0);
        isStereo = true;
      }

      StereoAndMatchedDetMap[label] = stereoandmatchedDetIds;

      // book StereoAndMatched MEs 
      if(isStereo){
	folder_organizer.setLayerFolder(detid,tTopo,det_layer_pair.second,true);
	// std::stringstream ss1;
	// folder_organizer.getLayerFolderName(ss1, detid, tTopo, true);  
	// std::cout << "Folder Name stereo " <<  ss1.str().c_str() << std::endl;
	//Create the Monitor Elements only when we have a stereo module
	createStereoAndMatchedMEs(ibooker,label);
      }
    }
 

  }//end of loop over detectors
}
//------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::createSimpleHitsMEs(DQMStore::IBooker & ibooker) 
{
  simplehitsMEs.meCategory = 0;
  simplehitsMEs.meTrackwidth = 0;
  simplehitsMEs.meExpectedwidth = 0;
  simplehitsMEs.meClusterwidth = 0;
  simplehitsMEs.meTrackanglealpha = 0;
  simplehitsMEs.meTrackanglebeta = 0;
  simplehitsMEs.meResolxMFTrackwidthProfile = 0;
  simplehitsMEs.meResolxMFTrackwidthProfileWClus1 = 0;
  simplehitsMEs.meResolxMFTrackwidthProfileWClus2 = 0;
  simplehitsMEs.meResolxMFTrackwidthProfileWClus3 = 0;
  simplehitsMEs.meResolxMFTrackwidthProfileWClus4 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus1 = 0;

  simplehitsMEs.meResMFTrackwidthProfileWClus2 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus21 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus22 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus23 = 0;

  simplehitsMEs.meResMFTrackwidthProfileWClus3 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus4 = 0;
  simplehitsMEs.meResolxMFTrackwidthProfileCategory1 = 0;
  simplehitsMEs.meResolxMFTrackwidthProfileCategory2 = 0;
  simplehitsMEs.meResolxMFTrackwidthProfileCategory3 = 0;
  simplehitsMEs.meResolxMFTrackwidthProfileCategory4 = 0;
  simplehitsMEs.meResolxMFClusterwidthProfileCategory1 = 0;
  simplehitsMEs.meResolxMFAngleProfile = 0;
  simplehitsMEs.meResolxLF = 0;
  simplehitsMEs.meResLF = 0;
  simplehitsMEs.mePullLF = 0;
  simplehitsMEs.meResolxMF = 0;
  simplehitsMEs.meResMF = 0;
  simplehitsMEs.mePullMF = 0;
    

  if(layerswitchResolx_LF) { 
    simplehitsMEs.meResolxLF = bookME1D(ibooker,"TH1Resolx_LF", "TH1Resolx_LF" ,"RecHit resol(x) coord. (local frame)");
    simplehitsMEs.meResolxLF->setAxisTitle("resol(x) RecHit coord. (local frame)");
  }
  if(layerswitchResolx_MF) { 
    simplehitsMEs.meResolxMF = bookME1D(ibooker,"TH1Resolx_MF", "TH1Resolx_MF" ,"RecHit resol(x) coord. (measurement frame)");
    simplehitsMEs.meResolxMF->setAxisTitle("resol(x) RecHit coord. (measurement frame)");
  }
  if(layerswitchRes_LF) { 
    simplehitsMEs.meResLF = bookME1D(ibooker,"TH1Res_LF", "TH1Res_LF" ,"Residual of the hit x coordinate (local frame)");
    simplehitsMEs.meResLF->setAxisTitle("Hit Res(x) (local frame)");
  }
  if(layerswitchRes_MF) {
    simplehitsMEs.meResMF = bookME1D(ibooker,"TH1Res_MF", "TH1Res_MF" ,"Residual of the hit x coordinate (measurement frame)");
    simplehitsMEs.meResMF->setAxisTitle("Hit Res(x) (measurement frame)");
  }
  if(layerswitchPull_LF) {
    simplehitsMEs.mePullLF = bookME1D(ibooker,"TH1Pull_LF", "TH1Pull_LF" ,"Pull distribution (local frame)");
    simplehitsMEs.mePullLF->setAxisTitle("Pull distribution (local frame)");
  } 
  if(layerswitchPull_MF) { 
    simplehitsMEs.mePullMF = bookME1D(ibooker,"TH1Pull_MF", "TH1Pull_MF" ,"Pull distribution (measurement frame)");
    simplehitsMEs.mePullMF->setAxisTitle("Pull distribution (measurement frame)");
  }
  if(layerswitchCategory) {
    simplehitsMEs.meCategory = bookME1D(ibooker,"TH1Category", "TH1Category" ,"Category");
    simplehitsMEs.meCategory->setAxisTitle("Category");
  } 
  if(layerswitchTrackwidth) { 
    simplehitsMEs.meTrackwidth = bookME1D(ibooker,"TH1Trackwidth", "TH1Trackwidth" ,"Track width");
    simplehitsMEs.meTrackwidth->setAxisTitle("Track width");
  }
  if(layerswitchExpectedwidth) { 
    simplehitsMEs.meExpectedwidth = bookME1D(ibooker,"TH1Expectedwidth", "TH1Expectedwidth" ,"Expected width");
    simplehitsMEs.meExpectedwidth->setAxisTitle("Expected width");
  }
  if(layerswitchClusterwidth) { 
    simplehitsMEs.meClusterwidth = bookME1D(ibooker,"TH1Clusterwidth", "TH1Clusterwidth" ,"Cluster width");
    simplehitsMEs.meClusterwidth->setAxisTitle("Cluster width");
  } 
  if(layerswitchTrackanglealpha) { 
    simplehitsMEs.meTrackanglealpha = bookME1D(ibooker,"TH1Trackanglealpha", "TH1Trackanglealpha" ,"Track angle alpha");
    simplehitsMEs.meTrackanglealpha->setAxisTitle("Track angle alpha");
  } 
  if(layerswitchTrackanglebeta) { 
    simplehitsMEs.meTrackanglebeta = bookME1D(ibooker,"TH1Trackanglebeta", "TH1Trackanglebeta" ,"Track angle beta");
    simplehitsMEs.meTrackanglebeta->setAxisTitle("Track angle beta");
  } 
  if(layerswitchResolxMFTrackwidthProfile_WClus1) { 
    simplehitsMEs.meResolxMFTrackwidthProfileWClus1 = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile_WClus1","TProfResolxMFTrackwidthProfile_WClus1","Profile of Resolution in MF vs track width for w=1");
    simplehitsMEs.meResolxMFTrackwidthProfileWClus1->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfileWClus1->setAxisTitle("Resolution (measurement frame) w=1",2);
  }
  if(layerswitchResolxMFTrackwidthProfile_WClus2) { 
    simplehitsMEs.meResolxMFTrackwidthProfileWClus2 = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile_WClus2","TProfResolxMFTrackwidthProfile_WClus2","Profile of Resolution in MF vs track width for w=2");
    simplehitsMEs.meResolxMFTrackwidthProfileWClus2->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfileWClus2->setAxisTitle("Resolution (measurement frame) w=2",2);

  } 
  if(layerswitchResolxMFTrackwidthProfile_WClus3) {
    simplehitsMEs.meResolxMFTrackwidthProfileWClus3 = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile_WClus3","TProfResolxMFTrackwidthProfile_WClus3","Profile of Resolution in MF vs track width for w=3");
    simplehitsMEs.meResolxMFTrackwidthProfileWClus3->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfileWClus3->setAxisTitle("Resolution (measurement frame) w=3",2);
  }  
  if(layerswitchResolxMFTrackwidthProfile_WClus4) { 
    simplehitsMEs.meResolxMFTrackwidthProfileWClus4 = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile_WClus4","TProfResolxMFTrackwidthProfile_WClus4","Profile of Resolution in MF vs track width for w=4");
    simplehitsMEs.meResolxMFTrackwidthProfileWClus4->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfileWClus4->setAxisTitle("Resolution (measurement frame) w=3",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus1) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus1 = bookMEProfile(ibooker,"TProfResMFTrackwidthProfile_WClus1","TProfResMFTrackwidthProfile_WClus1","Profile of Residuals(x) in MF vs track width for w=1");
    simplehitsMEs.meResMFTrackwidthProfileWClus1->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus1->setAxisTitle("Residuals(x) (measurement frame) w=1",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus2) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus2 = bookMEProfile(ibooker,"TProfResMFTrackwidthProfile_WClus2","TProfResMFTrackwidthProfile_WClus2","Profile of Residuals(x) in MF vs track width for w=2");
    simplehitsMEs.meResMFTrackwidthProfileWClus2->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus2->setAxisTitle("Residuals(x) (measurement frame) w=2",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus21) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus21 = bookMEProfile(ibooker,"TProfResMFTrackwidthProfile_WClus21","TProfResMFTrackwidthProfile_WClus21","Profile of Residuals(x) in MF vs track width for w=2");
    simplehitsMEs.meResMFTrackwidthProfileWClus21->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus21->setAxisTitle("Residuals(x) (measurement frame) w=2",2);
  }
  if(layerswitchResMFTrackwidthProfile_WClus22) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus22 = bookMEProfile(ibooker,"TProfResMFTrackwidthProfile_WClus22","TProfResMFTrackwidthProfile_WClus22","Profile of Residuals(x) in MF vs track width for w=2");
    simplehitsMEs.meResMFTrackwidthProfileWClus22->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus22->setAxisTitle("Residuals(x) (measurement frame) w=2",2);

  } 
  if(layerswitchResMFTrackwidthProfile_WClus23) {
    simplehitsMEs.meResMFTrackwidthProfileWClus23 = bookMEProfile(ibooker,"TProfResMFTrackwidthProfile_WClus23","TProfResMFTrackwidthProfile_WClus23","Profile of Residuals(x) in MF vs track width for w=2");
    simplehitsMEs.meResMFTrackwidthProfileWClus23->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus23->setAxisTitle("Residuals(x) (measurement frame) w=2",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus3) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus3 = bookMEProfile(ibooker,"TProfResMFTrackwidthProfile_WClus3","TProfResMFTrackwidthProfile_WClus3","Profile of Residuals(x) in MF vs track width for w=3");
    simplehitsMEs.meResMFTrackwidthProfileWClus3->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus3->setAxisTitle("Residuals(x) (measurement frame) w=3",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus4) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus4 = bookMEProfile(ibooker,"TProfResMFTrackwidthProfile_WClus4","TProfResMFTrackwidthProfile_WClus4","Profile of Residuals(x) in MF vs track width for w=4");
    simplehitsMEs.meResMFTrackwidthProfileWClus4->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus4->setAxisTitle("Residuals(x) (measurement frame) w=4",2);
  } 
  if(layerswitchResolxMFTrackwidthProfile) {  
    simplehitsMEs.meResolxMFTrackwidthProfile = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile","TProfResolxMFTrackwidthProfile","Profile of Resolution in MF vs track width");
    simplehitsMEs.meResolxMFTrackwidthProfile->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfile->setAxisTitle("Resolution (measurement frame)",2);
  }
  if(layerswitchResolxMFTrackwidthProfile_Category1) {  
    simplehitsMEs.meResolxMFTrackwidthProfileCategory1 = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile_Category1","TProfResolxMFTrackwidthProfile_Category1","Profile of Resolution in MF vs track width (Category 1)");
    simplehitsMEs.meResolxMFTrackwidthProfileCategory1->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfileCategory1->setAxisTitle("Resolution (measurement frame) Category 1",2);
  }
  if(layerswitchResolxMFTrackwidthProfile_Category2) {  
    simplehitsMEs.meResolxMFTrackwidthProfileCategory2 = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile_Category2","TProfResolxMFTrackwidthProfile_Category2","Profile of Resolution in MF vs track width (Category 2)");
    simplehitsMEs.meResolxMFTrackwidthProfileCategory2->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfileCategory2->setAxisTitle("Resolution (measurement frame) Category 2",2);
  }
  if(layerswitchResolxMFTrackwidthProfile_Category3) { 
    simplehitsMEs.meResolxMFTrackwidthProfileCategory3 = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile_Category3","TProfResolxMFTrackwidthProfile_Category3","Profile of Resolution in MF vs track width (Category 3)");
    simplehitsMEs.meResolxMFTrackwidthProfileCategory3->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfileCategory3->setAxisTitle("Resolution (measurement frame) Category 3",2);
  }
  if(layerswitchResolxMFTrackwidthProfile_Category4) { 
    simplehitsMEs.meResolxMFTrackwidthProfileCategory4 = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfile_Category4","TProfResolxMFTrackwidthProfile_Category4","Profile of Resolution in MF vs track width (Category 4)");
    simplehitsMEs.meResolxMFTrackwidthProfileCategory4->setAxisTitle("Track width",1);
    simplehitsMEs.meResolxMFTrackwidthProfileCategory4->setAxisTitle("Resolution (measurement frame) Category 4",2);
  }
  if(layerswitchResolxMFClusterwidthProfile_Category1) {
    simplehitsMEs.meResolxMFClusterwidthProfileCategory1 = bookMEProfile(ibooker,"TProfResolxMFClusterwidthProfile_Category1","TProfResolxMFClusterwidthProfile_Category1","Profile of Resolution in MF vs cluster width (Category 1)");
    simplehitsMEs.meResolxMFClusterwidthProfileCategory1->setAxisTitle("Cluster width",1);
    simplehitsMEs.meResolxMFClusterwidthProfileCategory1->setAxisTitle("Resolution (measurement frame) Category 1",2);
  }
  if(layerswitchResolxMFAngleProfile) { 
    simplehitsMEs.meResolxMFAngleProfile = bookMEProfile(ibooker,"TProfResolxMFAngleProfile","TProfResolxMFAngleProfile","Profile of Resolution in MF vs Track angle alpha");
    simplehitsMEs.meResolxMFAngleProfile->setAxisTitle("Track angle alpha",1);
    simplehitsMEs.meResolxMFAngleProfile->setAxisTitle("Resolution (measurement frame)",2);
  } 

         
}
//------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::createLayerMEs(DQMStore::IBooker & ibooker,std::string label) 
{
  SiStripHistoId hidmanager;
  LayerMEs layerMEs; 

  layerMEs.meWclusRphi = 0;
  layerMEs.meAdcRphi = 0;
  layerMEs.meResolxLFRphi = 0;
  layerMEs.meResolxMFRphi = 0;
  layerMEs.meResolxMFRphiwclus1 = 0;
  layerMEs.meResolxMFRphiwclus2 = 0;
  layerMEs.meResolxMFRphiwclus3 = 0;
  layerMEs.meResolxMFRphiwclus4 = 0;
  layerMEs.meResLFRphi = 0;
  layerMEs.meResMFRphi = 0;
  layerMEs.meResMFRphiwclus1 = 0;
  layerMEs.meResMFRphiwclus2 = 0;
  layerMEs.meResMFRphiwclus3 = 0;
  layerMEs.meResMFRphiwclus4 = 0;
  layerMEs.mePullLFRphi = 0;
  layerMEs.mePullMFRphi = 0;
  layerMEs.mePullMFRphiwclus1 = 0;
  layerMEs.mePullMFRphiwclus2 = 0;
  layerMEs.mePullMFRphiwclus3 = 0;
  layerMEs.mePullMFRphiwclus4 = 0;
  layerMEs.meTrackangleRphi = 0;
  layerMEs.meTrackanglebetaRphi = 0;
  layerMEs.meTrackangle2Rphi = 0;
  layerMEs.mePullTrackangleProfileRphi = 0;
  layerMEs.mePullTrackangle2DRphi = 0;
  layerMEs.meTrackwidthRphi = 0;
  layerMEs.meExpectedwidthRphi = 0;
  layerMEs.meClusterwidthRphi = 0;
  layerMEs.meCategoryRphi = 0;
  layerMEs.mePullTrackwidthProfileRphi = 0;
  layerMEs.mePullTrackwidthProfileRphiwclus1 = 0;
  layerMEs.mePullTrackwidthProfileRphiwclus2 = 0;
  layerMEs.mePullTrackwidthProfileRphiwclus3 = 0;
  layerMEs.mePullTrackwidthProfileRphiwclus4 = 0;
  layerMEs.mePullTrackwidthProfileCategory1Rphi = 0;
  layerMEs.mePullTrackwidthProfileCategory2Rphi = 0;
  layerMEs.mePullTrackwidthProfileCategory3Rphi = 0;
  layerMEs.mePullTrackwidthProfileCategory4Rphi = 0;
  layerMEs.meResolxMFTrackwidthProfileRphi = 0;

  layerMEs.meResolxMFTrackwidthProfileWclus1Rphi = 0;
  layerMEs.meResolxMFTrackwidthProfileWclus2Rphi = 0;
  layerMEs.meResolxMFTrackwidthProfileWclus3Rphi = 0;
  layerMEs.meResolxMFTrackwidthProfileWclus4Rphi = 0;
  layerMEs.meResMFTrackwidthProfileWclus1Rphi = 0;
  layerMEs.meResMFTrackwidthProfileWclus2Rphi = 0;
  layerMEs.meResMFTrackwidthProfileWclus3Rphi = 0;
  layerMEs.meResMFTrackwidthProfileWclus4Rphi = 0;

  layerMEs.meResolxMFTrackwidthProfileCategory1Rphi = 0;
  layerMEs.meResolxMFTrackwidthProfileCategory2Rphi = 0;
  layerMEs.meResolxMFTrackwidthProfileCategory3Rphi = 0;
  layerMEs.meResolxMFTrackwidthProfileCategory4Rphi = 0;
  layerMEs.meResolxMFClusterwidthProfileCategory1Rphi = 0;
  layerMEs.meResolxMFAngleProfileRphi = 0;
  layerMEs.merapidityResProfilewclus1 = 0;
  layerMEs.merapidityResProfilewclus2 = 0;
  layerMEs.merapidityResProfilewclus3 = 0;
  layerMEs.merapidityResProfilewclus4 = 0;
      

  //WclusRphi
  if(layerswitchWclusRphi) {
    layerMEs.meWclusRphi = bookME1D(ibooker,"TH1WclusRphi", hidmanager.createHistoLayer("Wclus_rphi","layer",label,"").c_str() ,"Cluster Width - Number of strips that belong to the RecHit cluster"); 
    layerMEs.meWclusRphi->setAxisTitle(("Cluster Width [nr strips] in "+ label).c_str());
  }
  //AdcRphi
  if(layerswitchAdcRphi) {
    layerMEs.meAdcRphi = bookME1D(ibooker,"TH1AdcRphi", hidmanager.createHistoLayer("Adc_rphi","layer",label,"").c_str() ,"RecHit Cluster Charge");
    layerMEs.meAdcRphi->setAxisTitle(("cluster charge [ADC] in " + label).c_str());
  }
  //ResolxLFRphi
  if(layerswitchResolxLFRphi) {
    layerMEs.meResolxLFRphi = bookME1D(ibooker,"TH1ResolxLFRphi", hidmanager.createHistoLayer("Resolx_LF_rphi","layer",label,"").c_str() ,"RecHit resol(x) coord.");   //<resolor>~20micron  
    layerMEs.meResolxLFRphi->setAxisTitle(("resol(x) RecHit coord. (local frame) in " + label).c_str());
  }
  //ResolxMFRphi
  if(layerswitchResolxMFRphi) {
    layerMEs.meResolxMFRphi = bookME1D(ibooker,"TH1ResolxMFRphi", hidmanager.createHistoLayer("Resolx_MF_rphi","layer",label,"").c_str() ,"RecHit resol(x) coord.");   //<resolor>~20micron  
    layerMEs.meResolxMFRphi->setAxisTitle(("resol(x) RecHit coord. (measurement frame) in " + label).c_str());
  }
  //ResolxMFRphiwclus1
  if(layerswitchResolxMFRphiwclus1) {
    layerMEs.meResolxMFRphiwclus1 = bookME1D(ibooker,"TH1ResolxMFRphiwclus1", hidmanager.createHistoLayer("Resolx_MF_wclus1_rphi","layer",label,"").c_str() ,"RecHit resol(x) coord. w=1 ");   //<resolor>~20micron  
    layerMEs.meResolxMFRphiwclus1->setAxisTitle(("resol(x) RecHit coord. (measurement frame) for w=1 in " + label).c_str());
  }
  //ResolxMFRphiwclus2
  if(layerswitchResolxMFRphiwclus2) {
    layerMEs.meResolxMFRphiwclus2 = bookME1D(ibooker,"TH1ResolxMFRphiwclus2", hidmanager.createHistoLayer("Resolx_MF_wclus2_rphi","layer",label,"").c_str() ,"RecHit resol(x) coord. w=2 ");   //<resolor>~20micron  
    layerMEs.meResolxMFRphiwclus2->setAxisTitle(("resol(x) RecHit coord. (measurement frame) for w=2 in " + label).c_str());
  }
  //ResolxMFRphiwclus3
  if(layerswitchResolxMFRphiwclus3) {
    layerMEs.meResolxMFRphiwclus3 = bookME1D(ibooker,"TH1ResolxMFRphiwclus3", hidmanager.createHistoLayer("Resolx_MF_wclus3_rphi","layer",label,"").c_str() ,"RecHit resol(x) coord. w=3 ");   //<resolor>~20micron  
    layerMEs.meResolxMFRphiwclus3->setAxisTitle(("resol(x) RecHit coord. (measurement frame) for w=3 in " + label).c_str());
  }
  //ResolxMFRphiwclus4
  if(layerswitchResolxMFRphiwclus4) {
    layerMEs.meResolxMFRphiwclus4 = bookME1D(ibooker,"TH1ResolxMFRphiwclus4", hidmanager.createHistoLayer("Resolx_MF_wclus4_rphi","layer",label,"").c_str() ,"RecHit resol(x) coord. w=4 ");   //<resolor>~20micron  
    layerMEs.meResolxMFRphiwclus4->setAxisTitle(("resol(x) RecHit coord. (measurement frame) for w=4 in " + label).c_str());
  }
  //ResLFRphi
  if(layerswitchResLFRphi) {
    layerMEs.meResLFRphi = bookME1D(ibooker,"TH1ResLFRphi", hidmanager.createHistoLayer("Res_LF_rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate"); 
    layerMEs.meResLFRphi->setAxisTitle(("Hit Residuals(x) (local frame) in " + label).c_str());
  }
  //ResMFRphi
  if(layerswitchResMFRphi) {
    layerMEs.meResMFRphi = bookME1D(ibooker,"TH1ResMFRphi",hidmanager.createHistoLayer("Res_MF_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate");
    layerMEs.meResMFRphi->setAxisTitle(("Hit Residuals(x) (measurement frame) in "+ label).c_str());
  }
  //ResMFRphiwclus1
  if(layerswitchResMFRphiwclus1) {
    layerMEs.meResMFRphiwclus1 = bookME1D(ibooker,"TH1ResMFRphiwclus1",hidmanager.createHistoLayer("Res_MF_wclus1_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate w=1");
    layerMEs.meResMFRphiwclus1->setAxisTitle(("Hit Residuals(x) (measurement frame) for w=1 in "+ label).c_str());
  }
  //ResMFRphiwclus2
  if(layerswitchResMFRphiwclus2) {
    layerMEs.meResMFRphiwclus2 = bookME1D(ibooker,"TH1ResMFRphiwclus2",hidmanager.createHistoLayer("Res_MF_wclus2_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate w=2");
    layerMEs.meResMFRphiwclus2->setAxisTitle(("Hit Residuals(x) (measurement frame) for w=2 in "+ label).c_str());
  }
  //ResMFRphiwclus3
  if(layerswitchResMFRphiwclus3) {
    layerMEs.meResMFRphiwclus3 = bookME1D(ibooker,"TH1ResMFRphiwclus3",hidmanager.createHistoLayer("Res_MF_wclus3_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate w=3");
    layerMEs.meResMFRphiwclus3->setAxisTitle(("Hit Residuals(x) (measurement frame) for w=3 in "+ label).c_str());
  }
  //ResMFRphiwclus4
  if(layerswitchResMFRphiwclus4) {
    layerMEs.meResMFRphiwclus4 = bookME1D(ibooker,"TH1ResMFRphiwclus4",hidmanager.createHistoLayer("Res_MF_wclus4_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate w=4");
    layerMEs.meResMFRphiwclus4->setAxisTitle(("Hit Residuals(x) (measurement frame) for w=4 in "+ label).c_str());
  }
  //PullLFRphi
  if(layerswitchPullLFRphi) {
    layerMEs.mePullLFRphi = bookME1D(ibooker,"TH1PullLFRphi", hidmanager.createHistoLayer("Pull_LF_rphi","layer",label,"").c_str() ,"Pull distribution");  
    layerMEs.mePullLFRphi->setAxisTitle(("Pull distribution (local frame) in " + label).c_str());
  }
  //PullMFRphi
  if(layerswitchPullMFRphi) {
    layerMEs.mePullMFRphi = bookME1D(ibooker,"TH1PullMFRphi", hidmanager.createHistoLayer("Pull_MF_rphi","layer",label,"").c_str() ,"Pull distribution");  
    layerMEs.mePullMFRphi->setAxisTitle(("Pull distribution (measurement frame) in " + label).c_str());
  }
  //PullMFRphiwclus1
  if(layerswitchPullMFRphiwclus1) {
    layerMEs.mePullMFRphiwclus1 = bookME1D(ibooker,"TH1PullMFRphiwclus1", hidmanager.createHistoLayer("Pull_MF_wclus1_rphi","layer",label,"").c_str() ,"Pull distribution w=1");  
    layerMEs.mePullMFRphiwclus1->setAxisTitle(("Pull distribution (measurement frame) for w=1 in " + label).c_str());
  }
  //PullMFRphiwclus2
  if(layerswitchPullMFRphiwclus2) {
    layerMEs.mePullMFRphiwclus2 = bookME1D(ibooker,"TH1PullMFRphiwclus2", hidmanager.createHistoLayer("Pull_MF_wclus2_rphi","layer",label,"").c_str() ,"Pull distribution w=2");  
    layerMEs.mePullMFRphiwclus2->setAxisTitle(("Pull distribution (measurement frame) for w=2 in " + label).c_str());
  }
  //PullMFRphiwclus3
  if(layerswitchPullMFRphiwclus3) {
    layerMEs.mePullMFRphiwclus3 = bookME1D(ibooker,"TH1PullMFRphiwclus3", hidmanager.createHistoLayer("Pull_MF_wclus3_rphi","layer",label,"").c_str() ,"Pull distribution w=3");  
    layerMEs.mePullMFRphiwclus3->setAxisTitle(("Pull distribution (measurement frame) for w=3 in " + label).c_str());
  }
  //PullMFRphiwclus4
  if(layerswitchPullMFRphiwclus4) {
    layerMEs.mePullMFRphiwclus4 = bookME1D(ibooker,"TH1PullMFRphiwclus4", hidmanager.createHistoLayer("Pull_MF_wclus4_rphi","layer",label,"").c_str() ,"Pull distribution w=4");  
    layerMEs.mePullMFRphiwclus4->setAxisTitle(("Pull distribution (measurement frame) for w=4 in " + label).c_str());
  }

  if(layerswitchTrackangleRphi) {
    layerMEs.meTrackangleRphi = bookME1D(ibooker,"TH1TrackangleRphi",hidmanager.createHistoLayer("Track_angle_Rphi","layer",label,"").c_str() ,"Track angle alpha");
    layerMEs.meTrackangleRphi->setAxisTitle(("Track angle in "+ label).c_str());
  }
  if(layerswitchTrackanglebetaRphi) {
    layerMEs.meTrackanglebetaRphi = bookME1D(ibooker,"TH1TrackanglebetaRphi",hidmanager.createHistoLayer("Track_angle_beta_Rphi","layer",label,"").c_str() ,"Track angle beta");
    layerMEs.meTrackanglebetaRphi->setAxisTitle((""+ label).c_str());
  }
  if(layerswitchTrackangle2Rphi) {
    layerMEs.meTrackangle2Rphi = bookME1D(ibooker,"TH1Trackangle2Rphi",hidmanager.createHistoLayer("Track_angle2_Rphi","layer",label,"").c_str() ,"");
    layerMEs.meTrackangle2Rphi->setAxisTitle((""+ label).c_str());
  }
  if(layerswitchPullTrackangleProfileRphi) {
    layerMEs.mePullTrackangleProfileRphi = bookMEProfile(ibooker,"TProfPullTrackangleProfileRphi",hidmanager.createHistoLayer("Pull_Trackangle_Profile_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track angle alpha");
    layerMEs.mePullTrackangleProfileRphi->setAxisTitle(("Track angle alpha in "+ label).c_str(),1);
    layerMEs.mePullTrackangleProfileRphi->setAxisTitle(("Pull (MF) in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackangle2DRphi) {
    layerMEs.mePullTrackangle2DRphi = bookME1D(ibooker,"TH1PullTrackangle2DRphi",hidmanager.createHistoLayer("Pull_Trackangle_2D_Rphi","layer",label,"").c_str() ,"");
    layerMEs.mePullTrackangle2DRphi->setAxisTitle((""+ label).c_str());
  }
  if(layerswitchTrackwidthRphi) {
    layerMEs.meTrackwidthRphi = bookME1D(ibooker,"TH1TrackwidthRphi",hidmanager.createHistoLayer("Track_width_Rphi","layer",label,"").c_str() ,"Track width");
    layerMEs.meTrackwidthRphi->setAxisTitle(("Track width in "+ label).c_str());
  }
  if(layerswitchExpectedwidthRphi) {
    layerMEs.meExpectedwidthRphi = bookME1D(ibooker,"TH1ExpectedwidthRphi",hidmanager.createHistoLayer("Expected_width_Rphi","layer",label,"").c_str() ,"Expected width");
    layerMEs.meExpectedwidthRphi->setAxisTitle(("Expected width in "+ label).c_str());
  }
  if(layerswitchClusterwidthRphi) {
    layerMEs.meClusterwidthRphi = bookME1D(ibooker,"TH1ClusterwidthRphi",hidmanager.createHistoLayer("Cluster_width_Rphi","layer",label,"").c_str() ,"Cluster width");
    layerMEs.meClusterwidthRphi->setAxisTitle(("Cluster width in "+ label).c_str());
  }
  if(layerswitchCategoryRphi) {
    layerMEs.meCategoryRphi = bookME1D(ibooker,"TH1CategoryRphi",hidmanager.createHistoLayer("Category_Rphi","layer",label,"").c_str() ,"Category");
    layerMEs.meCategoryRphi->setAxisTitle(("Category in "+ label).c_str());
  }
  if(layerswitchPullTrackwidthProfileRphi) {
    layerMEs.mePullTrackwidthProfileRphi = bookMEProfile(ibooker,"TProfPullTrackwidthProfileRphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width");
    layerMEs.mePullTrackwidthProfileRphi->setAxisTitle(("track width in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphi->setAxisTitle(("Pull (MF) in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileRphiwclus1) {
    layerMEs.mePullTrackwidthProfileRphiwclus1 = bookMEProfile(ibooker,"TProfPullTrackwidthProfileRphiwclus1",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi_wclus1","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for w=1");
    layerMEs.mePullTrackwidthProfileRphiwclus1->setAxisTitle(("track width for w=1 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphiwclus1->setAxisTitle(("Pull (MF) for w=1 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileRphiwclus2) {
    layerMEs.mePullTrackwidthProfileRphiwclus2 = bookMEProfile(ibooker,"TProfPullTrackwidthProfileRphiwclus2",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi_wclus2","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for w=2");
    layerMEs.mePullTrackwidthProfileRphiwclus2->setAxisTitle(("track width for w=2 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphiwclus2->setAxisTitle(("Pull (MF) for w=2 in "+ label).c_str(),2);

  }
  if(layerswitchPullTrackwidthProfileRphiwclus3) {
    layerMEs.mePullTrackwidthProfileRphiwclus3 = bookMEProfile(ibooker,"TProfPullTrackwidthProfileRphiwclus3",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi_wclus3","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for w=3");
    layerMEs.mePullTrackwidthProfileRphiwclus3->setAxisTitle(("track width for w=3 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphiwclus3->setAxisTitle(("Pull (MF) for w=3 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileRphiwclus4) {
    layerMEs.mePullTrackwidthProfileRphiwclus4 = bookMEProfile(ibooker,"TProfPullTrackwidthProfileRphiwclus4",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi_wclus4","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for w=4");
    layerMEs.mePullTrackwidthProfileRphiwclus4->setAxisTitle(("track width for w=4 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphiwclus4->setAxisTitle(("Pull (MF) for w=4 in "+ label).c_str(),2);

  }
  if(layerswitchPullTrackwidthProfileCategory1Rphi) {
    layerMEs.mePullTrackwidthProfileCategory1Rphi = bookMEProfile(ibooker,"TProfPullTrackwidthProfileCategory1Rphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category1_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for Category 1");
    layerMEs.mePullTrackwidthProfileCategory1Rphi->setAxisTitle(("track width for Category 1 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileCategory1Rphi->setAxisTitle(("Pull (MF) for Category 1 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory2Rphi) {
    layerMEs.mePullTrackwidthProfileCategory2Rphi = bookMEProfile(ibooker,"TProfPullTrackwidthProfileCategory2Rphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category2_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for Category 2");
    layerMEs.mePullTrackwidthProfileCategory2Rphi->setAxisTitle(("track width for Category 2 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileCategory2Rphi->setAxisTitle(("Pull (MF) for Category 2 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory3Rphi) {
    layerMEs.mePullTrackwidthProfileCategory3Rphi = bookMEProfile(ibooker,"TProfPullTrackwidthProfileCategory3Rphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category3_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for Category 3");
    layerMEs.mePullTrackwidthProfileCategory3Rphi->setAxisTitle(("track width for Category 3 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileCategory3Rphi->setAxisTitle(("Pull (MF) for Category 3 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory4Rphi) {
    layerMEs.mePullTrackwidthProfileCategory4Rphi = bookMEProfile(ibooker,"TProfPullTrackwidthProfileCategory4Rphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category4_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for Category 4");
    layerMEs.mePullTrackwidthProfileCategory4Rphi->setAxisTitle(("track width for Category 4 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileCategory4Rphi->setAxisTitle(("Pull (MF) for Category 4 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileRphi) {
    layerMEs.meResolxMFTrackwidthProfileRphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileRphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width");
    layerMEs.meResolxMFTrackwidthProfileRphi->setAxisTitle(("track width in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileRphi->setAxisTitle(("Resolution in MF in "+ label).c_str(),2);
  }

  if(layerswitchResolxMFTrackwidthProfileWclus1Rphi) {
    layerMEs.meResolxMFTrackwidthProfileWclus1Rphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileWclus1Rphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Wclus1_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for w=1");
    layerMEs.meResolxMFTrackwidthProfileWclus1Rphi->setAxisTitle(("track width for w=1 in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileWclus1Rphi->setAxisTitle(("Resolution in MF for w=1 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileWclus2Rphi) {
    layerMEs.meResolxMFTrackwidthProfileWclus2Rphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileWclus2Rphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Wclus2_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for w=2");
    layerMEs.meResolxMFTrackwidthProfileWclus2Rphi->setAxisTitle(("track width for w=2 in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileWclus2Rphi->setAxisTitle(("Resolution in MF for w=2 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileWclus3Rphi) {
    layerMEs.meResolxMFTrackwidthProfileWclus3Rphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileWclus3Rphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Wclus3_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for w=3");
    layerMEs.meResolxMFTrackwidthProfileWclus3Rphi->setAxisTitle(("track width for w=3 in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileWclus3Rphi->setAxisTitle(("Resolution in MF for w=3 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileWclus4Rphi) {
    layerMEs.meResolxMFTrackwidthProfileWclus4Rphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileWclus4Rphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Wclus4_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for w=4");
    layerMEs.meResolxMFTrackwidthProfileWclus4Rphi->setAxisTitle(("track width for w=4 in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileWclus4Rphi->setAxisTitle(("Resolution in MF for w=4 in "+ label).c_str(),2);
  }
  if(layerswitchResMFTrackwidthProfileWclus1Rphi) {
    layerMEs.meResMFTrackwidthProfileWclus1Rphi = bookMEProfile(ibooker,"TProfResMFTrackwidthProfileWclus1Rphi",hidmanager.createHistoLayer("ResMF_Track_width_Profile_Wclus1_Rphi","layer",label,"").c_str() ,"Profile of Residuals(x) in MF vs track width for w=1");
    layerMEs.meResMFTrackwidthProfileWclus1Rphi->setAxisTitle(("track width for w=1 in "+ label).c_str(),1);
    layerMEs.meResMFTrackwidthProfileWclus1Rphi->setAxisTitle(("Residuals(x) in MF for w=1 in "+ label).c_str(),2);
  }
  if(layerswitchResMFTrackwidthProfileWclus2Rphi) {
    layerMEs.meResMFTrackwidthProfileWclus2Rphi = bookMEProfile(ibooker,"TProfResMFTrackwidthProfileWclus2Rphi",hidmanager.createHistoLayer("ResMF_Track_width_Profile_Wclus2_Rphi","layer",label,"").c_str() ,"Profile of Residuals(x) in MF vs track width for w=2");
    layerMEs.meResMFTrackwidthProfileWclus2Rphi->setAxisTitle(("track width for w=2 in "+ label).c_str(),1);
    layerMEs.meResMFTrackwidthProfileWclus2Rphi->setAxisTitle(("Residuals(x) in MF for w=2 in "+ label).c_str(),2);
  }
  if(layerswitchResMFTrackwidthProfileWclus3Rphi) {
    layerMEs.meResMFTrackwidthProfileWclus3Rphi = bookMEProfile(ibooker,"TProfResMFTrackwidthProfileWclus3Rphi",hidmanager.createHistoLayer("ResMF_Track_width_Profile_Wclus3_Rphi","layer",label,"").c_str() ,"Profile of Residuals(x) in MF vs track width for w=3");
    layerMEs.meResMFTrackwidthProfileWclus3Rphi->setAxisTitle(("track width for w=3 in "+ label).c_str(),1);
    layerMEs.meResMFTrackwidthProfileWclus3Rphi->setAxisTitle(("Residuals(x) in MF for w=3 in "+ label).c_str(),2);
  }
  if(layerswitchResMFTrackwidthProfileWclus4Rphi) {
    layerMEs.meResMFTrackwidthProfileWclus4Rphi = bookMEProfile(ibooker,"TProfResMFTrackwidthProfileWclus4Rphi",hidmanager.createHistoLayer("ResMF_Track_width_Profile_Wclus4_Rphi","layer",label,"").c_str() ,"Profile of Residuals(x) in MF vs track width for w=4");
    layerMEs.meResMFTrackwidthProfileWclus4Rphi->setAxisTitle(("track width for w=4 in "+ label).c_str(),1);
    layerMEs.meResMFTrackwidthProfileWclus4Rphi->setAxisTitle(("Residuals(x) in MF for w=4 in "+ label).c_str(),2);
  }

  if(layerswitchResolxMFTrackwidthProfileCategory1Rphi) {
    layerMEs.meResolxMFTrackwidthProfileCategory1Rphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileCategory1Rphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Category1_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for Category 1");
    layerMEs.meResolxMFTrackwidthProfileCategory1Rphi->setAxisTitle(("track width for Category 1 in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileCategory1Rphi->setAxisTitle(("Resolution in MF for Category 1 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileCategory2Rphi) {
    layerMEs.meResolxMFTrackwidthProfileCategory2Rphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileCategory2Rphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Category2_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for Category 2");
    layerMEs.meResolxMFTrackwidthProfileCategory2Rphi->setAxisTitle(("track width for Category 2 in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileCategory2Rphi->setAxisTitle(("Resolution in MF for Category 2 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileCategory3Rphi) {
    layerMEs.meResolxMFTrackwidthProfileCategory3Rphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileCategory3Rphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Category3_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for Category 3");
    layerMEs.meResolxMFTrackwidthProfileCategory3Rphi->setAxisTitle(("track width for Category 3 in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileCategory3Rphi->setAxisTitle(("Resolution in MF for Category 3 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileCategory4Rphi) {
    layerMEs.meResolxMFTrackwidthProfileCategory4Rphi = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileCategory4Rphi",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Category3_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for Category 4");
    layerMEs.meResolxMFTrackwidthProfileCategory4Rphi->setAxisTitle(("track width for Category 4 in "+ label).c_str(),1);
    layerMEs.meResolxMFTrackwidthProfileCategory4Rphi->setAxisTitle(("Resolution in MF for Category 4 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFClusterwidthProfileCategory1Rphi) {
    layerMEs.meResolxMFClusterwidthProfileCategory1Rphi = bookMEProfile(ibooker,"TProfResolxMFClusterwidthProfileCategory1Rphi",hidmanager.createHistoLayer("ResolxMF_Cluster_width_Profile_Category1_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs cluster width for Category 1");
    layerMEs.meResolxMFClusterwidthProfileCategory1Rphi->setAxisTitle(("cluster width for Category 1 in "+ label).c_str(),1);
    layerMEs.meResolxMFClusterwidthProfileCategory1Rphi->setAxisTitle(("Resolution in MF for Category 1 in "+ label).c_str(),2);
  }
  if(layerswitchResolxMFAngleProfileRphi) {
    layerMEs.meResolxMFAngleProfileRphi = bookMEProfile(ibooker,"TProfResolxMFAngleProfileRphi",hidmanager.createHistoLayer("ResolxMF_Angle_Profile_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track angle alpha");
    layerMEs.meResolxMFAngleProfileRphi->setAxisTitle(("track angle alpha in "+ label).c_str(),1);
    layerMEs.meResolxMFAngleProfileRphi->setAxisTitle(("Resolution in MF in "+ label).c_str(),2);
  }
  if(layerswitchrapidityResProfilewclus1) {
    layerMEs.merapidityResProfilewclus1 = bookMEProfile(ibooker,"TProfrapidityResProfilewclus1",hidmanager.createHistoLayer("rapidity_Res_Profile_wclus1","layer",label,"").c_str() ,"Profile of rapidity vs Res for w=1");
    layerMEs.merapidityResProfilewclus1->setAxisTitle(("Res for w=1 in "+ label).c_str(),1);
    layerMEs.merapidityResProfilewclus1->setAxisTitle(("rapidity for w=1 in "+ label).c_str(),2);
  }
  if(layerswitchrapidityResProfilewclus2) {
    layerMEs.merapidityResProfilewclus2 = bookMEProfile(ibooker,"TProfrapidityResProfilewclus2",hidmanager.createHistoLayer("rapidity_Res_Profile_wclus2","layer",label,"").c_str() ,"Profile of rapidity vs Res for w=2");
    layerMEs.merapidityResProfilewclus2->setAxisTitle(("Res for w=2 in "+ label).c_str(),1);
    layerMEs.merapidityResProfilewclus2->setAxisTitle(("rapidity for w=2 in "+ label).c_str(),2);
  }
  if(layerswitchrapidityResProfilewclus3) {
    layerMEs.merapidityResProfilewclus3 = bookMEProfile(ibooker,"TProfrapidityResProfilewclus3",hidmanager.createHistoLayer("rapidity_Res_Profile_wclus3","layer",label,"").c_str() ,"Profile of rapidity vs Res for w=3");
    layerMEs.merapidityResProfilewclus3->setAxisTitle(("Res for w=3 in "+ label).c_str(),1);
    layerMEs.merapidityResProfilewclus3->setAxisTitle(("rapidity for w=3 in "+ label).c_str(),2);
  }
  if(layerswitchrapidityResProfilewclus4) {
    layerMEs.merapidityResProfilewclus4 = bookMEProfile(ibooker,"TProfrapidityResProfilewclus4",hidmanager.createHistoLayer("rapidity_Res_Profile_wclus4","layer",label,"").c_str() ,"Profile of rapidity vs Res for w=4");
    layerMEs.merapidityResProfilewclus4->setAxisTitle(("Res for w=4 in "+ label).c_str(),1);
    layerMEs.merapidityResProfilewclus4->setAxisTitle(("rapidity for w=4 in "+ label).c_str(),2);
  }
      

  LayerMEsMap[label]=layerMEs;
 
}
//------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::createStereoAndMatchedMEs(DQMStore::IBooker & ibooker,std::string label) 
{
  SiStripHistoId hidmanager;
  StereoAndMatchedMEs stereoandmatchedMEs; 

  stereoandmatchedMEs.meWclusSas = 0;
  stereoandmatchedMEs.meAdcSas = 0;
  stereoandmatchedMEs.meResolxLFSas = 0;
  stereoandmatchedMEs.meResolxMFSas = 0;
  stereoandmatchedMEs.meResLFSas = 0;
  stereoandmatchedMEs.meResMFSas = 0;
  stereoandmatchedMEs.mePullLFSas = 0;
  stereoandmatchedMEs.mePullMFSas = 0;
  stereoandmatchedMEs.meTrackangleSas = 0;
  stereoandmatchedMEs.meTrackanglebetaSas = 0;
  stereoandmatchedMEs.mePullTrackangleProfileSas = 0;
  stereoandmatchedMEs.meTrackwidthSas = 0;
  stereoandmatchedMEs.meExpectedwidthSas = 0;
  stereoandmatchedMEs.meClusterwidthSas = 0;
  stereoandmatchedMEs.meCategorySas = 0;
  stereoandmatchedMEs.mePullTrackwidthProfileSas = 0;
  stereoandmatchedMEs.mePullTrackwidthProfileCategory1Sas = 0;
  stereoandmatchedMEs.mePullTrackwidthProfileCategory2Sas = 0;
  stereoandmatchedMEs.mePullTrackwidthProfileCategory3Sas = 0;
  stereoandmatchedMEs.mePullTrackwidthProfileCategory4Sas = 0;
  stereoandmatchedMEs.meResolxMFTrackwidthProfileSas = 0;
  stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory1Sas = 0;
  stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory2Sas = 0;
  stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory3Sas = 0;
  stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory4Sas = 0;
  stereoandmatchedMEs.meResolxMFClusterwidthProfileCategory1Sas = 0;
  stereoandmatchedMEs.meResolxMFAngleProfileSas = 0;

  stereoandmatchedMEs.mePosxMatched = 0;
  stereoandmatchedMEs.mePosyMatched = 0;
  stereoandmatchedMEs.meResolxMatched = 0;
  stereoandmatchedMEs.meResolyMatched = 0;
  stereoandmatchedMEs.meResxMatched = 0;
  stereoandmatchedMEs.meResyMatched = 0;
  stereoandmatchedMEs.mePullxMatched = 0;
  stereoandmatchedMEs.mePullyMatched = 0;

  //WclusSas
  if(layerswitchWclusSas) {
    stereoandmatchedMEs.meWclusSas = bookME1D(ibooker,"TH1WclusSas", hidmanager.createHistoLayer("Wclus_sas","layer",label,"").c_str() ,"Cluster Width - Number of strips that belong to the RecHit cluster");  
    stereoandmatchedMEs.meWclusSas->setAxisTitle(("Cluster Width [nr strips] (stereo) in "+ label).c_str());
  }
  //AdcSas
  if(layerswitchAdcSas) {
    stereoandmatchedMEs.meAdcSas = bookME1D(ibooker,"TH1AdcSas", hidmanager.createHistoLayer("Adc_sas","layer",label,"").c_str() ,"RecHit Cluster Charge"); 
    stereoandmatchedMEs.meAdcSas->setAxisTitle(("cluster charge [ADC] (stereo) in " + label).c_str());
  }
  //ResolxLFSas
  if(layerswitchResolxLFSas) {
    stereoandmatchedMEs.meResolxLFSas = bookME1D(ibooker,"TH1ResolxLFSas", hidmanager.createHistoLayer("Resolx_LF_sas","layer",label,"").c_str() ,"RecHit resol(x) coord.");  
    stereoandmatchedMEs.meResolxLFSas->setAxisTitle(("resol(x) RecHit coord. (local frame) (stereo) in " + label).c_str());
  }
  //ResolxMFSas
  if(layerswitchResolxMFSas) {
    stereoandmatchedMEs.meResolxMFSas = bookME1D(ibooker,"TH1ResolxMFSas", hidmanager.createHistoLayer("Resolx_MF_sas","layer",label,"").c_str() ,"RecHit resol(x) coord.");  
    stereoandmatchedMEs.meResolxMFSas->setAxisTitle(("resol(x) RecHit coord. (measurement frame) (stereo) in " + label).c_str());
  }
  //ResLFSas
  if(layerswitchResLFSas) {
    stereoandmatchedMEs.meResLFSas = bookME1D(ibooker,"TH1ResLFSas", hidmanager.createHistoLayer("Res_LF_sas","layer",label,"").c_str() ,"Residual of the hit x coordinate"); 
    stereoandmatchedMEs.meResLFSas->setAxisTitle(("Hit Residuals(x) (local frame) (stereo) in " + label).c_str());
  }
  //ResMFSas
  if(layerswitchResMFSas) {
    stereoandmatchedMEs.meResMFSas = bookME1D(ibooker,"TH1ResMFSas", hidmanager.createHistoLayer("Res_MF_sas","layer",label,"").c_str() ,"Residual of the hit x coordinate"); 
    stereoandmatchedMEs.meResMFSas->setAxisTitle(("Hit Residuals(x) (stereo) in " + label).c_str());
  }
  //PullLFSas
  if(layerswitchPullLFSas) {
    stereoandmatchedMEs.mePullLFSas = bookME1D(ibooker,"TH1PullLFSas", hidmanager.createHistoLayer("Pull_LF_sas","layer",label,"").c_str() ,"Pull distribution");  
    stereoandmatchedMEs.mePullLFSas->setAxisTitle(("Pull distribution (local frame) (stereo) in " + label).c_str());
  }
  //PullMFSas
  if(layerswitchPullMFSas) {
    stereoandmatchedMEs.mePullMFSas = bookME1D(ibooker,"TH1PullMFSas", hidmanager.createHistoLayer("Pull_MF_sas","layer",label,"").c_str() ,"Pull distribution");  
    stereoandmatchedMEs.mePullMFSas->setAxisTitle(("Pull distribution (measurement frame) (stereo) in " + label).c_str());
  }

  if(layerswitchTrackangleSas) {
    stereoandmatchedMEs.meTrackangleSas = bookME1D(ibooker,"TH1TrackangleSas",hidmanager.createHistoLayer("Track_angle_Sas","layer",label,"").c_str() ,"Track angle");
    stereoandmatchedMEs.meTrackangleSas->setAxisTitle(("Track angle (stereo) in " + label).c_str());
  }
  if(layerswitchTrackanglebetaSas) {
    stereoandmatchedMEs.meTrackanglebetaSas = bookME1D(ibooker,"TH1TrackanglebetaSas",hidmanager.createHistoLayer("Track_angle_beta_Sas","layer",label,"").c_str() ,"Track angle beta");
    stereoandmatchedMEs.meTrackanglebetaSas->setAxisTitle(("Track angle beta (stereo) in " + label).c_str());
  }
  if(layerswitchPullTrackangleProfileSas) {
    stereoandmatchedMEs.mePullTrackangleProfileSas = bookMEProfile(ibooker,"TProfPullTrackangleProfileSas",hidmanager.createHistoLayer("Pull_Track_angle_Profile_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track angle (stereo)");
    stereoandmatchedMEs.mePullTrackangleProfileSas->setAxisTitle(("track angle (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackangleProfileSas->setAxisTitle(("Pull in MF (stereo) in " + label).c_str(),2);
  }
  if(layerswitchTrackwidthSas) {
    stereoandmatchedMEs.meTrackwidthSas = bookME1D(ibooker,"TH1TrackwidthSas",hidmanager.createHistoLayer("Track_width_Sas","layer",label,"").c_str() ,"Track width");
    stereoandmatchedMEs.meTrackwidthSas->setAxisTitle(("Track width (stereo) in " + label).c_str());
  }
  if(layerswitchExpectedwidthSas) {
    stereoandmatchedMEs.meExpectedwidthSas = bookME1D(ibooker,"TH1ExpectedwidthSas",hidmanager.createHistoLayer("Expected_width_Sas","layer",label,"").c_str() ,"Expected width");
    stereoandmatchedMEs.meExpectedwidthSas->setAxisTitle(("Expected width (stereo) in " + label).c_str());
  }
  if(layerswitchClusterwidthSas) {
    stereoandmatchedMEs.meClusterwidthSas = bookME1D(ibooker,"TH1ClusterwidthSas",hidmanager.createHistoLayer("Cluster_width_Sas","layer",label,"").c_str() ,"Cluster width");
    stereoandmatchedMEs.meClusterwidthSas->setAxisTitle(("Cluster width (stereo) in " + label).c_str());
  }
  if(layerswitchCategorySas) {
    stereoandmatchedMEs.meCategorySas = bookME1D(ibooker,"TH1CategorySas",hidmanager.createHistoLayer("Category_Sas","layer",label,"").c_str() ,"Category");
    stereoandmatchedMEs.meCategorySas->setAxisTitle(("Category (stereo) in " + label).c_str());
  }
  if(layerswitchPullTrackwidthProfileSas) {
    stereoandmatchedMEs.mePullTrackwidthProfileSas = bookMEProfile(ibooker,"TProfPullTrackwidthProfileSas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileSas->setAxisTitle(("track width (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileSas->setAxisTitle(("Pull in MF (stereo) in " + label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory1Sas) {
    stereoandmatchedMEs.mePullTrackwidthProfileCategory1Sas = bookMEProfile(ibooker,"TProfPullTrackwidthProfileCategory1Sas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category1_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (Category 1) (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileCategory1Sas->setAxisTitle(("track width (Category 1) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileCategory1Sas->setAxisTitle(("Pull in MF (Category 1) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory2Sas) {
    stereoandmatchedMEs.mePullTrackwidthProfileCategory2Sas = bookMEProfile(ibooker,"TProfPullTrackwidthProfileCategory2Sas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category2_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (Category 2) (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileCategory2Sas->setAxisTitle(("track width (Category 2) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileCategory2Sas->setAxisTitle(("Pull in MF (Category 2) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory3Sas) {
    stereoandmatchedMEs.mePullTrackwidthProfileCategory3Sas = bookMEProfile(ibooker,"TProfPullTrackwidthProfileCategory3Sas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category3_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (Category 3) (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileCategory3Sas->setAxisTitle(("track width (Category 3) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileCategory3Sas->setAxisTitle(("Pull in MF (Category 3) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory4Sas) {
    stereoandmatchedMEs.mePullTrackwidthProfileCategory4Sas = bookMEProfile(ibooker,"TProfPullTrackwidthProfileCategory4Sas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category4_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (Category 4) (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileCategory4Sas->setAxisTitle(("track width (Category 4) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileCategory4Sas->setAxisTitle(("Pull in MF (Category 4) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileSas) {
    stereoandmatchedMEs.meResolxMFTrackwidthProfileSas = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileSas",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (stereo)");
    stereoandmatchedMEs.meResolxMFTrackwidthProfileSas->setAxisTitle(("track width (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meResolxMFTrackwidthProfileSas->setAxisTitle(("Resolution in MF (stereo) in " + label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileCategory1Sas) {
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory1Sas = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileCategory1Sas",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Category1_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (Category 1) (stereo)");
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory1Sas->setAxisTitle((" track width (Category 1) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory1Sas->setAxisTitle(("  Resolution in MF (Category 1) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileCategory2Sas) {
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory2Sas = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileCategory2Sas",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Category2_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (Category 2) (stereo)");
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory2Sas->setAxisTitle((" track width (Category 2) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory2Sas->setAxisTitle((" Resolution in MF (Category 2) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileCategory3Sas) {
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory3Sas = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileCategory3Sas",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Category3_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (Category 3) (stereo)");
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory3Sas->setAxisTitle((" track width (Category 3) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory3Sas->setAxisTitle((" Resolution in MF (Category 3) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchResolxMFTrackwidthProfileCategory4Sas) {
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory4Sas = bookMEProfile(ibooker,"TProfResolxMFTrackwidthProfileCategory4Sas",hidmanager.createHistoLayer("ResolxMF_Track_width_Profile_Category4_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (Category 4) (stereo)");
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory4Sas->setAxisTitle((" track width (Category 4) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meResolxMFTrackwidthProfileCategory4Sas->setAxisTitle((" Resolution in MF (Category 4) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchResolxMFClusterwidthProfileCategory1Sas) {
    stereoandmatchedMEs.meResolxMFClusterwidthProfileCategory1Sas = bookMEProfile(ibooker,"TProfResolxMFClusterwidthProfileCategory1Sas",hidmanager.createHistoLayer("ResolxMF_Cluster_width_Profile_Category1_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs cluster width (Category 1) (stereo)");
    stereoandmatchedMEs.meResolxMFClusterwidthProfileCategory1Sas->setAxisTitle(("cluster width (Category 1) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meResolxMFClusterwidthProfileCategory1Sas->setAxisTitle((" Resolution in MF (Category 1) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchResolxMFAngleProfileSas) {
    stereoandmatchedMEs.meResolxMFAngleProfileSas = bookMEProfile(ibooker,"TProfResolxMFAngleProfileSas",hidmanager.createHistoLayer("ResolxMF_Angle_Profile_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track angle (stereo)");
    stereoandmatchedMEs.meResolxMFAngleProfileSas->setAxisTitle(("track angle (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meResolxMFAngleProfileSas->setAxisTitle(("Resolution in MF (stereo) in " + label).c_str(),2);
  }
  //PosxMatched
  if(layerswitchPosxMatched) {
    stereoandmatchedMEs.mePosxMatched = bookME1D(ibooker,"TH1PosxMatched", hidmanager.createHistoLayer("Posx_matched","layer",label,"").c_str() ,"RecHit x coord.");  
    stereoandmatchedMEs.mePosxMatched->setAxisTitle(("x coord. matched RecHit (local frame) in " + label).c_str());
  }
  //PosyMatched
  if(layerswitchPosyMatched) {
    stereoandmatchedMEs.mePosyMatched = bookME1D(ibooker,"TH1PosyMatched", hidmanager.createHistoLayer("Posy_matched","layer",label,"").c_str() ,"RecHit y coord."); 
    stereoandmatchedMEs.mePosyMatched->setAxisTitle(("y coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ResolxMatched
  if(layerswitchResolxMatched) {
    stereoandmatchedMEs.meResolxMatched = bookME1D(ibooker,"TH1ResolxMatched", hidmanager.createHistoLayer("Resolx_matched","layer",label,"").c_str() ,"RecHit resol(x) coord.");  
    stereoandmatchedMEs.meResolxMatched->setAxisTitle(("resol(x) coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ResolyMatched
  if(layerswitchResolyMatched) {
    stereoandmatchedMEs.meResolyMatched = bookME1D(ibooker,"TH1ResolyMatched", hidmanager.createHistoLayer("Resoly_matched","layer",label,"").c_str() ,"RecHit resol(y) coord."); 
    stereoandmatchedMEs.meResolyMatched->setAxisTitle(("resol(y) coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ResxMatched
  if(layerswitchResxMatched) {
    stereoandmatchedMEs.meResxMatched = bookME1D(ibooker,"TH1ResxMatched", hidmanager.createHistoLayer("Resx_matched","layer",label,"").c_str() ,"Residual of the hit x coord."); 
    stereoandmatchedMEs.meResxMatched->setAxisTitle(("Residuals(x) in matched RecHit in " + label).c_str());
  }
  //ResyMatched
  if(layerswitchResyMatched) {
    stereoandmatchedMEs.meResyMatched = bookME1D(ibooker,"TH1ResyMatched", hidmanager.createHistoLayer("Resy_matched","layer",label,"").c_str() ,"Residual of the hit x coord."); 
    stereoandmatchedMEs.meResyMatched->setAxisTitle(("Res(y) in matched RecHit in " + label).c_str());
  }

  StereoAndMatchedMEsMap[label]=stereoandmatchedMEs;
 
}
//------------------------------------------------------------------------------------------
inline MonitorElement* SiStripTrackingRecHitsValid::bookME1D(DQMStore::IBooker & ibooker,const char* ParameterSetLabel, const char* HistoName, const char* HistoTitle)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book1D(HistoName,HistoTitle,
		      Parameters.getParameter<int32_t>("Nbinx"),
		      Parameters.getParameter<double>("xmin"),
		      Parameters.getParameter<double>("xmax")
		      );
}
//------------------------------------------------------------------------------------------
inline MonitorElement* SiStripTrackingRecHitsValid::bookMEProfile(DQMStore::IBooker & ibooker,const char* ParameterSetLabel, const char* HistoName, const char* HistoTitle)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  //The number of channels in Y is disregarded in a profile plot.
  return ibooker.bookProfile(HistoName,HistoTitle,
			   Parameters.getParameter<int32_t>("Nbinx"),
			   Parameters.getParameter<double>("xmin"),
			   Parameters.getParameter<double>("xmax"),
			   Parameters.getParameter<double>("ymin"),
			   Parameters.getParameter<double>("ymax"),
			   "" 
			   );
}

// DEFINE_FWK_MODULE(SiStripTrackingRecHitsValid);
