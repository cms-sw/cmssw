#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "Validation/RecoTrack/interface/SiStripTrackingRecHitsValid.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

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
  m_cacheID_(0)
  // trajectoryInput_( ps.getParameter<edm::InputTag>("trajectoryInput") )
{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  
  trajectoryInputToken_ = consumes<std::vector<Trajectory> >( conf_.getParameter<edm::InputTag>("trajectoryInput") ); 

  edm::ParameterSet ParametersErrx_LF =  conf_.getParameter<edm::ParameterSet>("TH1Errx_LF");
  layerswitchErrx_LF = ParametersErrx_LF.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrx_MF =  conf_.getParameter<edm::ParameterSet>("TH1Errx_MF");
  layerswitchErrx_MF = ParametersErrx_MF.getParameter<bool>("layerswitchon");

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

  edm::ParameterSet ParametersErrxMFTrackwidthProfile_WClus1 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile_WClus1");
  layerswitchErrxMFTrackwidthProfile_WClus1 = ParametersErrxMFTrackwidthProfile_WClus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfile_WClus2 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile_WClus2");
  layerswitchErrxMFTrackwidthProfile_WClus2 = ParametersErrxMFTrackwidthProfile_WClus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfile_WClus3 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile_WClus3");
  layerswitchErrxMFTrackwidthProfile_WClus3 = ParametersErrxMFTrackwidthProfile_WClus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfile_WClus4 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile_WClus4");
  layerswitchErrxMFTrackwidthProfile_WClus4 = ParametersErrxMFTrackwidthProfile_WClus4.getParameter<bool>("layerswitchon");

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

  edm::ParameterSet ParametersErrxMFTrackwidthProfile =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile");
  layerswitchErrxMFTrackwidthProfile = ParametersErrxMFTrackwidthProfile.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfile_Category1 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile_Category1");
  layerswitchErrxMFTrackwidthProfile_Category1 = ParametersErrxMFTrackwidthProfile_Category1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfile_Category2 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile_Category2");
  layerswitchErrxMFTrackwidthProfile_Category2 = ParametersErrxMFTrackwidthProfile_Category2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfile_Category3 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile_Category3");
  layerswitchErrxMFTrackwidthProfile_Category3 = ParametersErrxMFTrackwidthProfile_Category3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfile_Category4 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfile_Category4");
  layerswitchErrxMFTrackwidthProfile_Category4 = ParametersErrxMFTrackwidthProfile_Category4.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFClusterwidthProfile_Category1 =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFClusterwidthProfile_Category1");
  layerswitchErrxMFClusterwidthProfile_Category1 = ParametersErrxMFClusterwidthProfile_Category1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFAngleProfile =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFAngleProfile");
  layerswitchErrxMFAngleProfile = ParametersErrxMFAngleProfile.getParameter<bool>("layerswitchon");
  
  edm::ParameterSet ParametersNstpRphi =  conf_.getParameter<edm::ParameterSet>("TH1NstpRphi");
  layerswitchNstpRphi = ParametersNstpRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersAdcRphi =  conf_.getParameter<edm::ParameterSet>("TH1AdcRphi");
  layerswitchAdcRphi = ParametersAdcRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPosxRphi =  conf_.getParameter<edm::ParameterSet>("TH1PosxRphi");
  layerswitchPosxRphi = ParametersPosxRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxLFRphi =  conf_.getParameter<edm::ParameterSet>("TH1ErrxLFRphi");
  layerswitchErrxLFRphi = ParametersErrxLFRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFRphi =  conf_.getParameter<edm::ParameterSet>("TH1ErrxMFRphi");
  layerswitchErrxMFRphi = ParametersErrxMFRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFRphiwclus1 =  conf_.getParameter<edm::ParameterSet>("TH1ErrxMFRphiwclus1");
  layerswitchErrxMFRphiwclus1 = ParametersErrxMFRphiwclus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFRphiwclus2 =  conf_.getParameter<edm::ParameterSet>("TH1ErrxMFRphiwclus2");
  layerswitchErrxMFRphiwclus2 = ParametersErrxMFRphiwclus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFRphiwclus3 =  conf_.getParameter<edm::ParameterSet>("TH1ErrxMFRphiwclus3");
  layerswitchErrxMFRphiwclus3 = ParametersErrxMFRphiwclus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFRphiwclus4 =  conf_.getParameter<edm::ParameterSet>("TH1ErrxMFRphiwclus4");
  layerswitchErrxMFRphiwclus4 = ParametersErrxMFRphiwclus4.getParameter<bool>("layerswitchon");

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

  edm::ParameterSet ParametersErrxMFTrackwidthProfileRphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileRphi");
  layerswitchErrxMFTrackwidthProfileRphi = ParametersErrxMFTrackwidthProfileRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileWclus1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileWclus1Rphi");
  layerswitchErrxMFTrackwidthProfileWclus1Rphi = ParametersErrxMFTrackwidthProfileWclus1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileWclus2Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileWclus2Rphi");
  layerswitchErrxMFTrackwidthProfileWclus2Rphi = ParametersErrxMFTrackwidthProfileWclus2Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileWclus3Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileWclus3Rphi");
  layerswitchErrxMFTrackwidthProfileWclus3Rphi = ParametersErrxMFTrackwidthProfileWclus3Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileWclus4Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileWclus4Rphi");
  layerswitchErrxMFTrackwidthProfileWclus4Rphi = ParametersErrxMFTrackwidthProfileWclus4Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfileWclus1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfileWclus1Rphi");
  layerswitchResMFTrackwidthProfileWclus1Rphi = ParametersResMFTrackwidthProfileWclus1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfileWclus2Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfileWclus2Rphi");
  layerswitchResMFTrackwidthProfileWclus2Rphi = ParametersResMFTrackwidthProfileWclus2Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfileWclus3Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfileWclus3Rphi");
  layerswitchResMFTrackwidthProfileWclus3Rphi = ParametersResMFTrackwidthProfileWclus3Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersResMFTrackwidthProfileWclus4Rphi =  conf_.getParameter<edm::ParameterSet>("TProfResMFTrackwidthProfileWclus4Rphi");
  layerswitchResMFTrackwidthProfileWclus4Rphi = ParametersResMFTrackwidthProfileWclus4Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileCategory1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileCategory1Rphi");
  layerswitchErrxMFTrackwidthProfileCategory1Rphi = ParametersErrxMFTrackwidthProfileCategory1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileCategory2Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileCategory2Rphi");
  layerswitchErrxMFTrackwidthProfileCategory2Rphi = ParametersErrxMFTrackwidthProfileCategory2Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileCategory3Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileCategory3Rphi");
  layerswitchErrxMFTrackwidthProfileCategory3Rphi = ParametersErrxMFTrackwidthProfileCategory3Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileCategory4Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileCategory4Rphi");
  layerswitchErrxMFTrackwidthProfileCategory4Rphi = ParametersErrxMFTrackwidthProfileCategory4Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFAngleProfileRphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFAngleProfileRphi");
  layerswitchErrxMFAngleProfileRphi = ParametersErrxMFAngleProfileRphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFClusterwidthProfileCategory1Rphi =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFClusterwidthProfileCategory1Rphi");
  layerswitchErrxMFClusterwidthProfileCategory1Rphi = ParametersErrxMFClusterwidthProfileCategory1Rphi.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersrapidityResProfilewclus1 =  conf_.getParameter<edm::ParameterSet>("TProfrapidityResProfilewclus1");
  layerswitchrapidityResProfilewclus1 = ParametersrapidityResProfilewclus1.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersrapidityResProfilewclus2 =  conf_.getParameter<edm::ParameterSet>("TProfrapidityResProfilewclus2");
  layerswitchrapidityResProfilewclus2 = ParametersrapidityResProfilewclus2.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersrapidityResProfilewclus3 =  conf_.getParameter<edm::ParameterSet>("TProfrapidityResProfilewclus3");
  layerswitchrapidityResProfilewclus3 = ParametersrapidityResProfilewclus3.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersrapidityResProfilewclus4 =  conf_.getParameter<edm::ParameterSet>("TProfrapidityResProfilewclus4");
  layerswitchrapidityResProfilewclus4 = ParametersrapidityResProfilewclus4.getParameter<bool>("layerswitchon");
  
  edm::ParameterSet ParametersNstpSas =  conf_.getParameter<edm::ParameterSet>("TH1NstpSas");
  layerswitchNstpSas = ParametersNstpSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersAdcSas =  conf_.getParameter<edm::ParameterSet>("TH1AdcSas");
  layerswitchAdcSas = ParametersAdcSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPosxSas =  conf_.getParameter<edm::ParameterSet>("TH1PosxSas");
  layerswitchPosxSas = ParametersPosxSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxLFSas =  conf_.getParameter<edm::ParameterSet>("TH1ErrxLFSas");
  layerswitchErrxLFSas = ParametersErrxLFSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFSas =  conf_.getParameter<edm::ParameterSet>("TH1ErrxMFSas");
  layerswitchErrxMFSas = ParametersErrxMFSas.getParameter<bool>("layerswitchon");

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

  edm::ParameterSet ParametersErrxMFTrackwidthProfileSas =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileSas");
  layerswitchErrxMFTrackwidthProfileSas = ParametersErrxMFTrackwidthProfileSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileCategory1Sas =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileCategory1Sas");
  layerswitchErrxMFTrackwidthProfileCategory1Sas = ParametersErrxMFTrackwidthProfileCategory1Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileCategory2Sas =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileCategory2Sas");
  layerswitchErrxMFTrackwidthProfileCategory2Sas = ParametersErrxMFTrackwidthProfileCategory2Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileCategory3Sas =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileCategory3Sas");
  layerswitchErrxMFTrackwidthProfileCategory3Sas = ParametersErrxMFTrackwidthProfileCategory3Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFTrackwidthProfileCategory4Sas =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFTrackwidthProfileCategory4Sas");
  layerswitchErrxMFTrackwidthProfileCategory4Sas = ParametersErrxMFTrackwidthProfileCategory4Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFAngleProfileSas =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFAngleProfileSas");
  layerswitchErrxMFAngleProfileSas = ParametersErrxMFAngleProfileSas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMFClusterwidthProfileCategory1Sas =  conf_.getParameter<edm::ParameterSet>("TProfErrxMFClusterwidthProfileCategory1Sas");
  layerswitchErrxMFClusterwidthProfileCategory1Sas = ParametersErrxMFClusterwidthProfileCategory1Sas.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPosxMatched =  conf_.getParameter<edm::ParameterSet>("TH1PosxMatched");
  layerswitchPosxMatched = ParametersPosxMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersPosyMatched =  conf_.getParameter<edm::ParameterSet>("TH1PosyMatched");
  layerswitchPosyMatched = ParametersPosyMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErrxMatched =  conf_.getParameter<edm::ParameterSet>("TH1ErrxMatched");
  layerswitchErrxMatched = ParametersErrxMatched.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersErryMatched =  conf_.getParameter<edm::ParameterSet>("TH1ErryMatched");
  layerswitchErryMatched = ParametersErryMatched.getParameter<bool>("layerswitchon");

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
SiStripTrackingRecHitsValid::~SiStripTrackingRecHitsValid()
{
  // if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}
//--------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::beginRun(const edm::Run& run, const edm::EventSetup& es){

  unsigned long long cacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
    edm::LogInfo("SiStripRecHitsValid") <<"SiStripRecHitsValid::beginRun: " 
					  << " Creating MEs for new Cabling ";     
    
    createMEs(es);
  }
}


void SiStripTrackingRecHitsValid::beginJob(const edm::EventSetup& es){
  
}

void SiStripTrackingRecHitsValid::endJob() {

  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("outputFile");
 
  // save histos in a file
  if(outputMEsInRootFile) dbe_->save(outputFileName);

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

  TrackerHitAssociator associate(e, conf_);
  PSimHit closest;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
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
  // e.getByLabel(trajectoryInput_, trajCollectionHandle);
  e.getByToken(trajectoryInputToken_, trajCollectionHandle);

  edm::LogVerbatim("TrajectoryAnalyzer") << "trajColl->size(): " << trajCollectionHandle->size();
  //cout<<"trajColl->size() = "<<trajCollectionHandle->size()<<endl;

  for (vector < Trajectory >::const_iterator it = trajCollectionHandle->begin(); it != trajCollectionHandle->end(); it++) {

    edm::LogVerbatim("TrajectoryAnalyzer") << "this traj has " << it->foundHits() << " valid hits" << " , " << "isValid: " << it->isValid();

    vector < TrajectoryMeasurement > tmColl = it->measurements();
    for (vector < TrajectoryMeasurement >::const_iterator itTraj = tmColl.begin(); itTraj != tmColl.end(); itTraj++) {
      if (!itTraj->updatedState().isValid()) continue;
      
      rechitrphi.clear();
      rechitstereo.clear();
      rechitmatched.clear();
      
      //edm::LogVerbatim("TrajectoryAnalyzer") << "tm number: " <<
      //   (itTraj - tmColl.begin()) + 1<< " , " << "tm.backwardState.pt: " <<
      //   itTraj->backwardPredictedState().globalMomentum().perp() << " , " <<
      //   "tm.forwardState.pt:  " << itTraj->forwardPredictedState().globalMomentum().perp() <<
      //   " , " << "tm.updatedState.pt:  " << itTraj->updatedState().globalMomentum().perp() <<
      //   " , " << "tm.globalPos.perp: "   << itTraj->updatedState().globalPosition().perp();

      if (itTraj->updatedState().globalMomentum().perp() < 0.5)	continue;

      TrajectoryStateOnSurface tsos = itTraj->updatedState();

      DetId detid2 = itTraj->recHit()->geographicalId();

      const TransientTrackingRecHit::ConstRecHitPointer thit2 = itTraj->recHit();
      const SiStripMatchedRecHit2D *matchedhit = dynamic_cast < const SiStripMatchedRecHit2D * >((*thit2).hit());
      const SiStripRecHit2D *hit2d = dynamic_cast < const SiStripRecHit2D * >((*thit2).hit());
      const SiStripRecHit1D *hit1d = dynamic_cast < const SiStripRecHit1D * >((*thit2).hit());
      //if(matchedhit) cout<<"manganomatchedhit"<<endl;
      //if(hit) cout<<"manganosimplehit"<<endl;
      //if (hit && matchedhit) cout<<"manganosimpleandmatchedhit"<<endl;
      const TrackingRecHit *thit = (*thit2).hit();

      detid = (thit)->geographicalId();
      myid = ((thit)->geographicalId()).rawId();
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
	matchedmonorstereo = "matched";
	rechitanalysis_matched(tsos, thit2, gluedDet, associate, stripcpe, matchedmonorstereo );
	// rechitmatched.push_back(rechitpro);

      }

      std::map<std::string, StereoAndMatchedMEs>::iterator iStereoAndMatchedME  = StereoAndMatchedMEsMap.find(label);

      //Filling Histograms for Matched hits

      if (isrechitmatched) {

	if(iStereoAndMatchedME != StereoAndMatchedMEsMap.end()){
	  fillME(iStereoAndMatchedME->second.mePosxMatched,rechitpro.x);
	  fillME(iStereoAndMatchedME->second.mePosyMatched,rechitpro.y);
	  fillME(iStereoAndMatchedME->second.meErrxMatched,sqrt(rechitpro.errxx));
	  fillME(iStereoAndMatchedME->second.meErryMatched,sqrt(rechitpro.erryy));
	  fillME(iStereoAndMatchedME->second.meResxMatched,rechitpro.resx);
	  fillME(iStereoAndMatchedME->second.meResyMatched,rechitpro.resy);
	  fillME(iStereoAndMatchedME->second.mePullxMatched,rechitpro.pullx);
	  fillME(iStereoAndMatchedME->second.mePullyMatched,rechitpro.pully);
	}
	
      }
    

      ///////////////////////////////////////////////////////
      // simple hits from matched hits
      ///////////////////////////////////////////////////////
 
      if (tsos.globalDirection().transverse() != 0) {
	track_rapidity = tsos.globalDirection().eta();
      } else {
	track_rapidity = -999.0;
      }

      GluedGeomDet *gdet;
      const SiStripRecHit2D *monohit;

      if (matchedhit) {
	auto hm = matchedhit->monoHit();
	monohit = &hm;
	//      const GeomDetUnit * monodet=gdet->monoDet();
	gdet = (GluedGeomDet *) tracker2->idToDet(matchedhit->geographicalId());
	  
	if (monohit) {

	  isrechitrphi = 1;
	  
	  //Analysis
	  matchedmonorstereo = "monoHit";
	  rechitanalysis_matched(tsos, thit2, gdet, associate, stripcpe, matchedmonorstereo );

	}

	auto s = matchedhit->stereoHit();
	const SiStripRecHit2D *stereohit = &s;
	
	if (stereohit) {
	
	  isrechitsas = 1;
	  
	  //Analysis
	  matchedmonorstereo = "stereoHit";
	  rechitanalysis_matched(tsos, thit2, gdet, associate, stripcpe, matchedmonorstereo );
	}
      }
      
      if (hit1d) {
	// simple hits are mono or stereo
	//      cout<<"simple hit"<<endl;
	if (StripSubdet.stereo() == 0) {
	  isrechitrphi = 1;
	  //      cout<<"simple hit mono"<<endl;

	  const GeomDetUnit *det = tracker.idToDetUnit(detid2);
	  const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (det);
	  
	  //Analysis for hit1d mono
	  rechitanalysis(tsos, thit2, stripdet, stripcpe, associate, true);

	}

	if (StripSubdet.stereo() == 1) {

	  //cout<<"simple hit stereo"<<endl;
	  isrechitsas = 1;

	  const GeomDetUnit *det = tracker.idToDetUnit(detid2);
	  const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (det);

	  //Analysis for hit1d stereo
	  rechitanalysis(tsos, thit2, stripdet, stripcpe, associate, true);

	}
      }


      if (hit2d) {
	// simple hits are mono or stereo
	//      cout<<"simple hit"<<endl;
	if (StripSubdet.stereo() == 0) {
	  isrechitrphi = 1;
	  //      cout<<"simple hit mono"<<endl;

	  const GeomDetUnit *det = tracker.idToDetUnit(detid2);
	  const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (det);

	  //Analysis for hit2d mono
	  rechitanalysis(tsos, thit2, stripdet, stripcpe, associate, false);

	}

	if (StripSubdet.stereo() == 1) {

	  //cout<<"simple hit stereo"<<endl;
	  isrechitsas = 1;

	  const GeomDetUnit *det = tracker.idToDetUnit(detid2);
	  const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (det);

	  //Analysis for hit2d stereo
	  rechitanalysis(tsos, thit2, stripdet, stripcpe, associate, false);

	}
      }

      //Filling Histograms for simple hits
      //cout<<"isrechitrphi,isrechitsas = "<<isrechitrphi<<","<<isrechitsas<<endl;

      std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(label);

      if (isrechitrphi > 0 || isrechitsas > 0) {



	if (isrechitrphi > 0) {

	  fillME(simplehitsMEs.meCategory,rechitpro.category);
	  fillME(simplehitsMEs.meTrackwidth,rechitpro.trackwidth);
	  fillME(simplehitsMEs.meExpectedwidth,rechitpro.expectedwidth);
	  fillME(simplehitsMEs.meClusterwidth,rechitpro.clusiz);
	  fillME(simplehitsMEs.meTrackanglealpha,rechitpro.trackangle);
	  fillME(simplehitsMEs.meTrackanglebeta,rechitpro.trackanglebeta);

	  fillME(simplehitsMEs.meErrxMFAngleProfile,rechitpro.trackangle, sqrt(rechitpro.errxxMF));
	  fillME(simplehitsMEs.meErrxMFTrackwidthProfile,rechitpro.trackwidth, sqrt(rechitpro.errxxMF));

	  if (rechitpro.clusiz == 1) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileWClus1,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus1,rechitpro.trackwidth, fabs(rechitpro.resxMF));
	  }
	  if (rechitpro.clusiz == 2) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileWClus2,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus2,rechitpro.trackwidth, fabs(rechitpro.resxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus21,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus22,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus23,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	  }
	  if (rechitpro.clusiz == 3) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileWClus3,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus3,rechitpro.trackwidth, fabs(rechitpro.resxMF));
	  }
	  if (rechitpro.clusiz == 4) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileWClus4,rechitpro.trackwidth, sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus4,rechitpro.trackwidth, fabs(rechitpro.resxMF));
	  }

	  if (rechitpro.category == 1) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileCategory1,rechitpro.trackwidth, sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meErrxMFClusterwidthProfileCategory1,rechitpro.clusiz, sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 2) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileCategory2,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 3) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileCategory3,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 4) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileCategory4,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }

	  fillME(simplehitsMEs.meErrxMF,sqrt(rechitpro.errxxMF));
	  fillME(simplehitsMEs.meErrxLF,sqrt(rechitpro.errxx));
	  fillME(simplehitsMEs.meResMF,rechitpro.resxMF);
	  fillME(simplehitsMEs.meResLF,rechitpro.resx);
	  fillME(simplehitsMEs.mePullMF,rechitpro.pullxMF);
	  fillME(simplehitsMEs.mePullLF,rechitpro.pullx);

	}

	if (isrechitsas > 0) {

	  fillME(simplehitsMEs.meCategory,rechitpro.category);
	  fillME(simplehitsMEs.meTrackwidth,rechitpro.trackwidth);
	  fillME(simplehitsMEs.meExpectedwidth,rechitpro.expectedwidth);
	  fillME(simplehitsMEs.meClusterwidth,rechitpro.clusiz);
	  fillME(simplehitsMEs.meTrackanglealpha,rechitpro.trackangle);
	  fillME(simplehitsMEs.meTrackanglebeta,rechitpro.trackanglebeta);

	  fillME(simplehitsMEs.meErrxMFAngleProfile,rechitpro.trackangle, sqrt(rechitpro.errxxMF));
	  fillME(simplehitsMEs.meErrxMFTrackwidthProfile,rechitpro.trackwidth, sqrt(rechitpro.errxxMF));

	  if (rechitpro.clusiz == 1) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileWClus1,rechitpro.trackwidth, sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus1,rechitpro.trackwidth, rechitpro.resxMF);
	  }

	  if (rechitpro.clusiz == 2) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileWClus2,rechitpro.trackwidth, sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus2,rechitpro.trackwidth, rechitpro.resxMF);
	  }
	  if (rechitpro.clusiz == 3) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileWClus3,rechitpro.trackwidth, sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus3,rechitpro.trackwidth, rechitpro.resxMF);
	  }
	  if (rechitpro.clusiz == 4) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileWClus4,rechitpro.trackwidth, sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meResMFTrackwidthProfileWClus4,rechitpro.trackwidth, rechitpro.resxMF);
	  }
	  if (rechitpro.category == 1) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileCategory1,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(simplehitsMEs.meErrxMFClusterwidthProfileCategory1,rechitpro.clusiz, sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 2) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileCategory2,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 3) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileCategory3,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 4) {
	    fillME(simplehitsMEs.meErrxMFTrackwidthProfileCategory4,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }

	  fillME(simplehitsMEs.meErrxMF,sqrt(rechitpro.errxxMF));
	  fillME(simplehitsMEs.meErrxLF,sqrt(rechitpro.errxx));
	  fillME(simplehitsMEs.meResMF,rechitpro.resxMF);
	  fillME(simplehitsMEs.meResLF,rechitpro.resx);
	  fillME(simplehitsMEs.mePullMF,rechitpro.pullxMF);
	  fillME(simplehitsMEs.mePullLF,rechitpro.pullx);

	}


	
	if(iLayerME != LayerMEsMap.end()){

	  fillME(iLayerME->second.meNstpRphi,rechitpro.clusiz);
	  fillME(iLayerME->second.meAdcRphi,rechitpro.cluchg);
	  fillME(iLayerME->second.mePosxRphi,rechitpro.x);
	  fillME(iLayerME->second.meErrxLFRphi,sqrt(rechitpro.errxx));
	  fillME(iLayerME->second.meErrxMFRphi,sqrt(rechitpro.errxxMF));

	  if( (min(rechitpro.clusiz, 4) - 1) == 1 ){fillME(iLayerME->second.meErrxMFRphiwclus1,sqrt(rechitpro.errxxMF));}
	  if( (min(rechitpro.clusiz, 4) - 1) == 2 ){fillME(iLayerME->second.meErrxMFRphiwclus2,sqrt(rechitpro.errxxMF));}
	  if( (min(rechitpro.clusiz, 4) - 1) == 3 ){fillME(iLayerME->second.meErrxMFRphiwclus3,sqrt(rechitpro.errxxMF));}
	  if( (min(rechitpro.clusiz, 4) - 1) == 4 ){fillME(iLayerME->second.meErrxMFRphiwclus4,sqrt(rechitpro.errxxMF));}

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
	    fillME(iLayerME->second.meErrxMFTrackwidthProfileWclus1Rphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(iLayerME->second.meResMFTrackwidthProfileWclus1Rphi,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	  }
	  if (rechitpro.clusiz == 2) {
	    fillME(iLayerME->second.meErrxMFTrackwidthProfileWclus2Rphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(iLayerME->second.meResMFTrackwidthProfileWclus2Rphi,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	  }
	  if (rechitpro.clusiz == 3) {
	    fillME(iLayerME->second.meErrxMFTrackwidthProfileWclus3Rphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(iLayerME->second.meResMFTrackwidthProfileWclus3Rphi,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	  }
	  if (rechitpro.clusiz == 4) {
	    fillME(iLayerME->second.meErrxMFTrackwidthProfileWclus4Rphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(iLayerME->second.meResMFTrackwidthProfileWclus4Rphi,rechitpro.trackwidth,fabs(rechitpro.resxMF));
	  }


	  if (rechitpro.category == 1) {
	    fillME(iLayerME->second.mePullTrackwidthProfileCategory1Rphi,rechitpro.trackwidth,fabs(rechitpro.pullxMF));
	    fillME(iLayerME->second.meErrxMFTrackwidthProfileCategory1Rphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(iLayerME->second.meErrxMFClusterwidthProfileCategory1Rphi,rechitpro.clusiz,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 2) {
	    fillME(iLayerME->second.mePullTrackwidthProfileCategory2Rphi,rechitpro.trackwidth,fabs(rechitpro.pullxMF));
	    fillME(iLayerME->second.meErrxMFTrackwidthProfileCategory2Rphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 3) {
	    fillME(iLayerME->second.mePullTrackwidthProfileCategory3Rphi,rechitpro.trackwidth,fabs(rechitpro.pullxMF));
	    fillME(iLayerME->second.meErrxMFTrackwidthProfileCategory3Rphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 4) {
	    fillME(iLayerME->second.mePullTrackwidthProfileCategory4Rphi,rechitpro.trackwidth,fabs(rechitpro.pullxMF));
	    fillME(iLayerME->second.meErrxMFTrackwidthProfileCategory4Rphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	    
	  fillME(iLayerME->second.meTrackwidthRphi,rechitpro.trackwidth);
	  fillME(iLayerME->second.meExpectedwidthRphi,rechitpro.expectedwidth);
	  fillME(iLayerME->second.meClusterwidthRphi,rechitpro.clusiz);
	  fillME(iLayerME->second.meCategoryRphi,rechitpro.category);
	  fillME(iLayerME->second.meErrxMFTrackwidthProfileRphi,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  fillME(iLayerME->second.meErrxMFAngleProfileRphi,rechitpro.trackangle,sqrt(rechitpro.errxxMF));
	}

	if(iStereoAndMatchedME != StereoAndMatchedMEsMap.end()){
	  
	  fillME(iStereoAndMatchedME->second.meNstpSas,rechitpro.clusiz);
	  fillME(iStereoAndMatchedME->second.meAdcSas,rechitpro.cluchg);
	  fillME(iStereoAndMatchedME->second.mePosxSas,rechitpro.x);
	  fillME(iStereoAndMatchedME->second.meErrxLFSas,sqrt(rechitpro.errxx));
	  fillME(iStereoAndMatchedME->second.meResLFSas,rechitpro.resx);
	  fillME(iStereoAndMatchedME->second.mePullLFSas,rechitpro.pullx);
	  fillME(iStereoAndMatchedME->second.meErrxMFSas,sqrt(rechitpro.errxxMF));
	  fillME(iStereoAndMatchedME->second.meResMFSas,rechitpro.resxMF);
	  fillME(iStereoAndMatchedME->second.mePullMFSas,rechitpro.pullxMF);
	  fillME(iStereoAndMatchedME->second.meTrackangleSas,rechitpro.trackangle);
	  fillME(iStereoAndMatchedME->second.mePullTrackangleProfileSas,rechitpro.trackangle, rechitpro.pullxMF);
	  fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileSas,rechitpro.trackwidth, rechitpro.pullxMF);
	  if (rechitpro.category == 1) {
	    fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileCategory1Sas,rechitpro.trackwidth,rechitpro.pullxMF);
	    fillME(iStereoAndMatchedME->second.meErrxMFTrackwidthProfileCategory1Sas,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	    fillME(iStereoAndMatchedME->second.meErrxMFClusterwidthProfileCategory1Sas,rechitpro.clusiz,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 2) {
	    fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileCategory2Sas,rechitpro.trackwidth,rechitpro.pullxMF);
	    fillME(iStereoAndMatchedME->second.meErrxMFTrackwidthProfileCategory2Sas,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 3) {
	    fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileCategory3Sas,rechitpro.trackwidth,rechitpro.pullxMF);
	    fillME(iStereoAndMatchedME->second.meErrxMFTrackwidthProfileCategory3Sas,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  if (rechitpro.category == 4) {
	    fillME(iStereoAndMatchedME->second.mePullTrackwidthProfileCategory4Sas,rechitpro.trackwidth,rechitpro.pullxMF);
	    fillME(iStereoAndMatchedME->second.meErrxMFTrackwidthProfileCategory4Sas,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  }
	  fillME(iStereoAndMatchedME->second.meTrackwidthSas,rechitpro.trackwidth);
	  fillME(iStereoAndMatchedME->second.meExpectedwidthSas,rechitpro.expectedwidth);
	  fillME(iStereoAndMatchedME->second.meClusterwidthSas,rechitpro.clusiz);
	  fillME(iStereoAndMatchedME->second.meCategorySas,rechitpro.category);
	  fillME(iStereoAndMatchedME->second.meErrxMFTrackwidthProfileSas,rechitpro.trackwidth,sqrt(rechitpro.errxxMF));
	  fillME(iStereoAndMatchedME->second.meErrxMFAngleProfileSas,rechitpro.trackangle, rechitpro.errxxMF);
	    
	}
	

      }                     //simplehits
      //cout<<"DebugLine301"<<endl;

    }
    //cout<<"DebugLine302"<<endl;

  }
  //cout<<"DebugLine303"<<endl;

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
void SiStripTrackingRecHitsValid::rechitanalysis_matched(TrajectoryStateOnSurface tsos, const TransientTrackingRecHit::ConstRecHitPointer thit, const GluedGeomDet* gluedDet, TrackerHitAssociator associate, edm::ESHandle<StripClusterParameterEstimator> stripcpe, std::string matchedmonorstereo){
  
  rechitpro.x = -999999.; rechitpro.y = -999999.; rechitpro.z = -999999.; rechitpro.errxx = -999999.; rechitpro.errxy = -999999.;   rechitpro.erryy = -999999.; 
  rechitpro.errxxMF = -999999.; rechitpro.phi = -999999.;rechitpro.resx = -999999.; rechitpro.resy = -999999.; rechitpro.resxMF = -999999.; 
  rechitpro.pullx = -999999.; rechitpro.pully = -999999.; rechitpro.pullxMF = -999999.; rechitpro.trackangle = -999999.; rechitpro.trackanglebeta = -999999.; 
  rechitpro.trackangle2 = -999999.; rechitpro.trackwidth = -999999.; rechitpro.expectedwidth = -999999.; rechitpro.category = -999999.; rechitpro.thickness = -999999.; 
  rechitpro.clusiz = -999999.; rechitpro.cluchg = -999999.; 

  const GeomDetUnit *monodet = gluedDet->monoDet(); 
  const GeomDetUnit *stereodet = gluedDet->stereoDet();
  //We initialized it to monoHit case because it complains that it may be uninitialized
  //and it will change value in the stereoHit case. The matched case do not use this
  const StripGeomDetUnit *stripdet = (const StripGeomDetUnit *) (monodet) ; 

  const SiStripMatchedRecHit2D *matchedhit = dynamic_cast < const SiStripMatchedRecHit2D * >((*thit).hit());
  const SiStripRecHit2D *monohit;
  const SiStripRecHit2D *stereohit;

  if (matchedmonorstereo == "monoHit"){
    auto hm = matchedhit->monoHit();
    monohit = &hm;
    stripdet = (const StripGeomDetUnit *) (monodet);
  } 
  if (matchedmonorstereo == "stereoHit"){
    auto s = matchedhit->stereoHit();
    stereohit = &s;
    stripdet = (const StripGeomDetUnit *) (stereodet);
  }
  //if(matchedhit) cout<<"manganomatchedhit"<<endl;
  //if(hit) cout<<"manganosimplehit"<<endl;
  //if (hit && matchedhit) cout<<"manganosimpleandmatchedhit"<<endl;
  const StripTopology & topol = (const StripTopology &) stripdet->topology();
  const TrackingRecHit *rechit = (*thit).hit();

  LocalPoint position;
  LocalError error;
  MeasurementPoint Mposition;
  MeasurementError Merror;

  if (matchedmonorstereo == "matched"){
    position=rechit->localPosition();
    error=rechit->localPositionError();
  }
  if(matchedmonorstereo == "monoHit"){
    position = monohit->localPosition();
    error = monohit->localPositionError();
    Mposition = topol.measurementPosition(position);
    Merror = topol.measurementError(position, error);
  } 
  if (matchedmonorstereo == "stereoHit"){
    position = stereohit->localPosition();
    error = stereohit->localPositionError();
    Mposition = topol.measurementPosition(position);
    Merror = topol.measurementError(position, error);
  }

  LocalVector trackdirection = tsos.localDirection();

  GlobalVector gtrkdir = gluedDet->toGlobal(trackdirection);
  LocalVector monotkdir = monodet->toLocal(gtrkdir);
  LocalVector stereotkdir = stereodet->toLocal(gtrkdir);
  
  if(matchedmonorstereo == "monoHit"){
    if (monotkdir.z() != 0) {
      rechitpro.trackangle = atan(monotkdir.x() / monotkdir.z()) * TMath::RadToDeg();
      rechitpro.trackanglebeta = atan(monotkdir.y() / monotkdir.z()) * TMath::RadToDeg();
    }
  }
  if (matchedmonorstereo == "stereoHit"){
    if (stereotkdir.z() != 0) {
      rechitpro.trackangle = atan(stereotkdir.x() / stereotkdir.z()) * TMath::RadToDeg();
      rechitpro.trackanglebeta = atan(stereotkdir.y() / stereotkdir.z()) * TMath::RadToDeg();
    }
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

  SiStripRecHit2D::ClusterRef clust;
  if(matchedmonorstereo == "monoHit"){clust = monohit->cluster();}
  if(matchedmonorstereo == "stereoHit"){clust = stereohit->cluster();}

  int clusiz=0;
  int totcharge=0;
  clusiz = clust->amplitudes().size();
  const std::vector<uint8_t> amplitudes=clust->amplitudes();
  for(size_t ia=0; ia<amplitudes.size();ia++){
    totcharge+=amplitudes[ia];
  }

  rechitpro.x = position.x();
  rechitpro.y = position.y();
  rechitpro.z = position.z();
  rechitpro.errxx = error.xx();
  rechitpro.errxy = error.xy();
  rechitpro.erryy = error.yy();
  rechitpro.errxxMF = Merror.uu();
  rechitpro.clusiz = clusiz;
  rechitpro.cluchg = totcharge;
 
  unsigned int iopt;
  if (rechitpro.clusiz > rechitpro.expectedwidth + 2) {
    iopt = 1;
  } else if (rechitpro.expectedwidth == 1) {
    iopt = 2;
  } else if (rechitpro.clusiz <= rechitpro.expectedwidth) {
    iopt = 3;
  } else {
    iopt = 4;
  }
  rechitpro.category = iopt;

  if(matchedmonorstereo == "matched"){matched.clear();matched = associate.associateHit(*matchedhit);}
  if(matchedmonorstereo == "monoHit"){matched.clear();matched = associate.associateHit(*monohit);}
  if(matchedmonorstereo == "stereoHit"){matched.clear();matched = associate.associateHit(*stereohit);}

  double mindist = 999999;
  double dist = 999999;
  double distx = 999999;
  double disty = 999999;
  std::pair<LocalPoint,LocalVector> closestPair;
  PSimHit closest;

  if(!matched.empty()){

    const StripGeomDetUnit* partnerstripdet =(StripGeomDetUnit*) gluedDet->stereoDet();
    std::pair<LocalPoint,LocalVector> hitPair;
    
    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
      //project simhit;
      if(matchedmonorstereo == "matched"){
	hitPair= projectHit((*m),partnerstripdet,gluedDet->surface());
	distx = fabs(rechitpro.x - hitPair.first.x());
	disty = fabs(rechitpro.y - hitPair.first.y());
	dist = sqrt(distx*distx+disty*disty);
      }
      if(matchedmonorstereo == "monoHit"){dist = abs((monohit)->localPosition().x() - (*m).localPosition().x());}
      if(matchedmonorstereo == "stereoHit"){dist = abs((stereohit)->localPosition().x() - (*m).localPosition().x());}

      // std::cout << " Simhit position x = " << hitPair.first.x() 
      //      << " y = " << hitPair.first.y() << " dist = " << dist << std::endl;
      if(dist<mindist){
	mindist = dist;
	closestPair = hitPair;
	closest = (*m);
      }
    }  
    
    if(matchedmonorstereo == "matched"){
      rechitpro.resx = rechitpro.x - closestPair.first.x();
      rechitpro.resy = rechitpro.y - closestPair.first.y();
      rechitpro.pullx = ((rechit)->localPosition().x() - (closestPair.first.x())) / sqrt(error.xx());
      rechitpro.pully = ((rechit)->localPosition().y() - (closestPair.first.y())) / sqrt(error.yy());
    }
    
    if( (matchedmonorstereo == "monoHit") || (matchedmonorstereo == "stereoHit") ){
      rechitpro.resx = rechitpro.x - closest.localPosition().x();
      rechitpro.resxMF = Mposition.x() - (topol.measurementPosition(closest.localPosition())).x();
      rechitpro.pullx = (rechit->localPosition().x() - (closest).localPosition().x()) / sqrt(error.xx());
      rechitpro.pullxMF = (rechitpro.resxMF)/sqrt(Merror.uu());
    }

  }
}
//--------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::rechitanalysis(TrajectoryStateOnSurface tsos, const TransientTrackingRecHit::ConstRecHitPointer thit, const StripGeomDetUnit *stripdet,edm::ESHandle<StripClusterParameterEstimator> stripcpe, TrackerHitAssociator associate,  bool simplehit1or2D){

  rechitpro.x = -999999.; rechitpro.y = -999999.; rechitpro.z = -999999.; rechitpro.errxx = -999999.; rechitpro.errxy = -999999.;   rechitpro.erryy = -999999.; 
  rechitpro.errxxMF = -999999.; rechitpro.phi = -999999.;rechitpro.resx = -999999.; rechitpro.resy = -999999.; rechitpro.resxMF = -999999.; 
  rechitpro.pullx = -999999.; rechitpro.pully = -999999.; rechitpro.pullxMF = -999999.; rechitpro.trackangle = -999999.; rechitpro.trackanglebeta = -999999.; 
  rechitpro.trackangle2 = -999999.; rechitpro.trackwidth = -999999.; rechitpro.expectedwidth = -999999.; rechitpro.category = -999999.; rechitpro.thickness = -999999.; 
  rechitpro.clusiz = -999999.; rechitpro.cluchg = -999999.; 
  
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

  SiStripRecHit1D::ClusterRef clust1d;
  SiStripRecHit2D::ClusterRef clust2d;
  int clusiz=0;
  int totcharge=0;
 
  if(!simplehit1or2D){
    clust2d = hit2d->cluster();
    clusiz = clust2d->amplitudes().size();
    const std::vector<uint8_t> amplitudes2d = clust2d->amplitudes();
    for(size_t ia=0; ia<amplitudes2d.size();ia++){
      totcharge+=amplitudes2d[ia];
    }
  } else {
    clust1d = hit1d->cluster();
    clusiz = clust1d->amplitudes().size();
    const std::vector<uint8_t> amplitudes1d = clust1d->amplitudes();
    for(size_t ia=0; ia<amplitudes1d.size();ia++){
      totcharge+=amplitudes1d[ia];
    }
  }

  rechitpro.x = position.x();
  rechitpro.y = position.y();
  rechitpro.z = position.z();
  rechitpro.errxx = error.xx();
  rechitpro.errxy = error.xy();
  rechitpro.erryy = error.yy();
  rechitpro.errxxMF = Merror.uu();
  rechitpro.clusiz = clusiz;
  rechitpro.cluchg = totcharge;

  unsigned int iopt;
  if (rechitpro.clusiz > rechitpro.expectedwidth + 2) {
    iopt = 1;
  } else if (rechitpro.expectedwidth == 1) {
    iopt = 2;
  } else if (rechitpro.clusiz <= rechitpro.expectedwidth) {
    iopt = 3;
  } else {
    iopt = 4;
  }
  rechitpro.category = iopt;


  matched.clear();
  if(!simplehit1or2D){
    matched = associate.associateHit(*hit2d);
  } else {
    matched = associate.associateHit(*hit1d);
  }

  double mindist = 999999;
  double dist = 999999;
  PSimHit closest;
  
  if(!matched.empty()){

    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
      if(!simplehit1or2D){
	dist = abs((hit2d)->localPosition().x() - (*m).localPosition().x());
      } else {
	dist = abs((hit1d)->localPosition().x() - (*m).localPosition().x());
      }
	  
      if(dist<mindist){
	mindist = dist;
	closest = (*m);
      }
    }  
    rechitpro.resx = rechitpro.x - closest.localPosition().x();
    rechitpro.resxMF = Mposition.x() - (topol.measurementPosition(closest.localPosition())).x();
    rechitpro.pullx = (rechit->localPosition().x() - (closest).localPosition().x()) / sqrt(error.xx());
    rechitpro.pullxMF = (rechitpro.resxMF)/sqrt(Merror.uu());
    
  }

}
//--------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::createMEs(const edm::EventSetup& es){

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
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

  createSimpleHitsMEs();

  // loop over detectors and book MEs
  edm::LogInfo("SiStripTrackingRecHitsValid|SiStripTrackingRecHitsValid")<<"nr. of activeDets:  "<<activeDets.size();
  for(std::vector<uint32_t>::iterator detid_iterator = activeDets.begin(); detid_iterator!=activeDets.end(); detid_iterator++){
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
      std::vector<uint32_t> layerDetIds;        
      if (det_layer_pair.first == "TIB") {
	substructure.getTIBDetectors(activeDets,layerDetIds,lnumber,0,0,0);
      } else if (det_layer_pair.first == "TOB") {
	substructure.getTOBDetectors(activeDets,layerDetIds,lnumber,0,0);
      } else if (det_layer_pair.first == "TID" && lnumber > 0) {
	substructure.getTIDDetectors(activeDets,layerDetIds,2,0,abs(lnumber),0);
      } else if (det_layer_pair.first == "TID" && lnumber < 0) {
	substructure.getTIDDetectors(activeDets,layerDetIds,1,0,abs(lnumber),0);
      } else if (det_layer_pair.first == "TEC" && lnumber > 0) {
	substructure.getTECDetectors(activeDets,layerDetIds,2,0,0,0,abs(lnumber),0);
      } else if (det_layer_pair.first == "TEC" && lnumber < 0) {
	substructure.getTECDetectors(activeDets,layerDetIds,1,0,0,0,abs(lnumber),0);
      }
      LayerDetMap[label] = layerDetIds;

      // book Layer MEs 
      folder_organizer.setLayerFolder(detid,tTopo,det_layer_pair.second,true);
      // std::stringstream ss;
      // folder_organizer.getLayerFolderName(ss, detid, tTopo, true); 
      // std::cout << "Folder Name " << ss.str().c_str() << std::endl;
      // folder_organizer.setLayerFolder(detid,det_layer_pair.second,true);
      createLayerMEs(label);
    }
    //Create StereoAndMatchedMEs
    std::map<std::string, StereoAndMatchedMEs>::iterator iStereoAndMatchedME  = StereoAndMatchedMEsMap.find(label);
    if(iStereoAndMatchedME==StereoAndMatchedMEsMap.end()) {
	
      // get detids for the stereo and matched layer. We are going to need a bool for these layers
      bool isStereo = false;
      // Keep in mind that when we are on the TID or TEC we deal with rings not wheel 
      int32_t stereolnumber = det_layer_pair.second;
      std::vector<uint32_t> stereoandmatchedDetIds;        
      if ( (det_layer_pair.first == "TIB") &&  (TIBDetId(detid).stereo()== 1) ) {
	substructure.getTIBDetectors(activeDets,stereoandmatchedDetIds,stereolnumber,0,0,0);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TOB") &&  (TOBDetId(detid).stereo()== 1) ) {
	substructure.getTOBDetectors(activeDets,stereoandmatchedDetIds,stereolnumber,0,0);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TID") && (stereolnumber > 0) && (TIDDetId(detid).stereo()== 1) ) {
	substructure.getTIDDetectors(activeDets,stereoandmatchedDetIds,2,0,abs(stereolnumber),1);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TID") && (stereolnumber < 0) && (TIDDetId(detid).stereo()== 1) ) {
	substructure.getTIDDetectors(activeDets,stereoandmatchedDetIds,1,0,abs(stereolnumber),1);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TEC") && (stereolnumber > 0) && (TECDetId(detid).stereo()== 1) ) {
	substructure.getTECDetectors(activeDets,stereoandmatchedDetIds,2,0,0,0,abs(stereolnumber),1);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TEC") && (stereolnumber < 0) && (TECDetId(detid).stereo()== 1) ) {
	substructure.getTECDetectors(activeDets,stereoandmatchedDetIds,1,0,0,0,abs(stereolnumber),1);
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
	createStereoAndMatchedMEs(label);
      }
    }
 

  }//end of loop over detectors
}
//------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::createSimpleHitsMEs() 
{
  simplehitsMEs.meCategory = 0;
  simplehitsMEs.meTrackwidth = 0;
  simplehitsMEs.meExpectedwidth = 0;
  simplehitsMEs.meClusterwidth = 0;
  simplehitsMEs.meTrackanglealpha = 0;
  simplehitsMEs.meTrackanglebeta = 0;
  simplehitsMEs.meErrxMFTrackwidthProfile = 0;
  simplehitsMEs.meErrxMFTrackwidthProfileWClus1 = 0;
  simplehitsMEs.meErrxMFTrackwidthProfileWClus2 = 0;
  simplehitsMEs.meErrxMFTrackwidthProfileWClus3 = 0;
  simplehitsMEs.meErrxMFTrackwidthProfileWClus4 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus1 = 0;

  simplehitsMEs.meResMFTrackwidthProfileWClus2 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus21 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus22 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus23 = 0;

  simplehitsMEs.meResMFTrackwidthProfileWClus3 = 0;
  simplehitsMEs.meResMFTrackwidthProfileWClus4 = 0;
  simplehitsMEs.meErrxMFTrackwidthProfileCategory1 = 0;
  simplehitsMEs.meErrxMFTrackwidthProfileCategory2 = 0;
  simplehitsMEs.meErrxMFTrackwidthProfileCategory3 = 0;
  simplehitsMEs.meErrxMFTrackwidthProfileCategory4 = 0;
  simplehitsMEs.meErrxMFClusterwidthProfileCategory1 = 0;
  simplehitsMEs.meErrxMFAngleProfile = 0;
  simplehitsMEs.meErrxLF = 0;
  simplehitsMEs.meResLF = 0;
  simplehitsMEs.mePullLF = 0;
  simplehitsMEs.meErrxMF = 0;
  simplehitsMEs.meResMF = 0;
  simplehitsMEs.mePullMF = 0;
    

  if(layerswitchErrx_LF) { 
    simplehitsMEs.meErrxLF = bookME1D("TH1Errx_LF", "TH1Errx_LF" ,"RecHit err(x) coord. (local frame)");
    simplehitsMEs.meErrxLF->setAxisTitle("err(x) RecHit coord. (local frame)");
  }
  if(layerswitchErrx_MF) { 
    simplehitsMEs.meErrxMF = bookME1D("TH1Errx_MF", "TH1Errx_MF" ,"RecHit err(x) coord. (measurement frame)");
    simplehitsMEs.meErrxMF->setAxisTitle("err(x) RecHit coord. (measurement frame)");
  }
  if(layerswitchRes_LF) { 
    simplehitsMEs.meResLF = bookME1D("TH1Res_LF", "TH1Res_LF" ,"Residual of the hit x coordinate (local frame)");
    simplehitsMEs.meResLF->setAxisTitle("Hit Res(x) (local frame)");
  }
  if(layerswitchRes_MF) {
    simplehitsMEs.meResMF = bookME1D("TH1Res_MF", "TH1Res_MF" ,"Residual of the hit x coordinate (measurement frame)");
    simplehitsMEs.meResMF->setAxisTitle("Hit Res(x) (measurement frame)");
  }
  if(layerswitchPull_LF) {
    simplehitsMEs.mePullLF = bookME1D("TH1Pull_LF", "TH1Pull_LF" ,"Pull distribution (local frame)");
    simplehitsMEs.mePullLF->setAxisTitle("Pull distribution (local frame)");
  } 
  if(layerswitchPull_MF) { 
    simplehitsMEs.mePullMF = bookME1D("TH1Pull_MF", "TH1Pull_MF" ,"Pull distribution (measurement frame)");
    simplehitsMEs.mePullMF->setAxisTitle("Pull distribution (measurement frame)");
  }
  if(layerswitchCategory) {
    simplehitsMEs.meCategory = bookME1D("TH1Category", "TH1Category" ,"Category");
    simplehitsMEs.meCategory->setAxisTitle("Category");
  } 
  if(layerswitchTrackwidth) { 
    simplehitsMEs.meTrackwidth = bookME1D("TH1Trackwidth", "TH1Trackwidth" ,"Track width");
    simplehitsMEs.meTrackwidth->setAxisTitle("Track width");
  }
  if(layerswitchExpectedwidth) { 
    simplehitsMEs.meExpectedwidth = bookME1D("TH1Expectedwidth", "TH1Expectedwidth" ,"Expected width");
    simplehitsMEs.meExpectedwidth->setAxisTitle("Expected width");
  }
  if(layerswitchClusterwidth) { 
    simplehitsMEs.meClusterwidth = bookME1D("TH1Clusterwidth", "TH1Clusterwidth" ,"Cluster width");
    simplehitsMEs.meClusterwidth->setAxisTitle("Cluster width");
  } 
  if(layerswitchTrackanglealpha) { 
    simplehitsMEs.meTrackanglealpha = bookME1D("TH1Trackanglealpha", "TH1Trackanglealpha" ,"Track angle alpha");
    simplehitsMEs.meTrackanglealpha->setAxisTitle("Track angle alpha");
  } 
  if(layerswitchTrackanglebeta) { 
    simplehitsMEs.meTrackanglebeta = bookME1D("TH1Trackanglebeta", "TH1Trackanglebeta" ,"Track angle beta");
    simplehitsMEs.meTrackanglebeta->setAxisTitle("Track angle beta");
  } 
  if(layerswitchErrxMFTrackwidthProfile_WClus1) { 
    simplehitsMEs.meErrxMFTrackwidthProfileWClus1 = bookMEProfile("TProfErrxMFTrackwidthProfile_WClus1","TProfErrxMFTrackwidthProfile_WClus1","Profile of Resolution in MF vs track width for w=1");
    simplehitsMEs.meErrxMFTrackwidthProfileWClus1->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfileWClus1->setAxisTitle("Resolution (measurement frame) w=1",2);
  }
  if(layerswitchErrxMFTrackwidthProfile_WClus2) { 
    simplehitsMEs.meErrxMFTrackwidthProfileWClus2 = bookMEProfile("TProfErrxMFTrackwidthProfile_WClus2","TProfErrxMFTrackwidthProfile_WClus2","Profile of Resolution in MF vs track width for w=2");
    simplehitsMEs.meErrxMFTrackwidthProfileWClus2->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfileWClus2->setAxisTitle("Resolution (measurement frame) w=2",2);

 } 
  if(layerswitchErrxMFTrackwidthProfile_WClus3) {
    simplehitsMEs.meErrxMFTrackwidthProfileWClus3 = bookMEProfile("TProfErrxMFTrackwidthProfile_WClus3","TProfErrxMFTrackwidthProfile_WClus3","Profile of Resolution in MF vs track width for w=3");
    simplehitsMEs.meErrxMFTrackwidthProfileWClus3->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfileWClus3->setAxisTitle("Resolution (measurement frame) w=3",2);
  }  
  if(layerswitchErrxMFTrackwidthProfile_WClus4) { 
    simplehitsMEs.meErrxMFTrackwidthProfileWClus4 = bookMEProfile("TProfErrxMFTrackwidthProfile_WClus4","TProfErrxMFTrackwidthProfile_WClus4","Profile of Resolution in MF vs track width for w=4");
    simplehitsMEs.meErrxMFTrackwidthProfileWClus4->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfileWClus4->setAxisTitle("Resolution (measurement frame) w=3",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus1) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus1 = bookMEProfile("TProfResMFTrackwidthProfile_WClus1","TProfResMFTrackwidthProfile_WClus1","Profile of Residuals(x) in MF vs track width for w=1");
    simplehitsMEs.meResMFTrackwidthProfileWClus1->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus1->setAxisTitle("Residuals(x) (measurement frame) w=1",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus2) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus2 = bookMEProfile("TProfResMFTrackwidthProfile_WClus2","TProfResMFTrackwidthProfile_WClus2","Profile of Residuals(x) in MF vs track width for w=2");
    simplehitsMEs.meResMFTrackwidthProfileWClus2->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus2->setAxisTitle("Residuals(x) (measurement frame) w=2",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus21) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus21 = bookMEProfile("TProfResMFTrackwidthProfile_WClus21","TProfResMFTrackwidthProfile_WClus21","Profile of Residuals(x) in MF vs track width for w=2");
    simplehitsMEs.meResMFTrackwidthProfileWClus21->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus21->setAxisTitle("Residuals(x) (measurement frame) w=2",2);
  }
  if(layerswitchResMFTrackwidthProfile_WClus22) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus22 = bookMEProfile("TProfResMFTrackwidthProfile_WClus22","TProfResMFTrackwidthProfile_WClus22","Profile of Residuals(x) in MF vs track width for w=2");
    simplehitsMEs.meResMFTrackwidthProfileWClus22->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus22->setAxisTitle("Residuals(x) (measurement frame) w=2",2);

  } 
  if(layerswitchResMFTrackwidthProfile_WClus23) {
    simplehitsMEs.meResMFTrackwidthProfileWClus23 = bookMEProfile("TProfResMFTrackwidthProfile_WClus23","TProfResMFTrackwidthProfile_WClus23","Profile of Residuals(x) in MF vs track width for w=2");
    simplehitsMEs.meResMFTrackwidthProfileWClus23->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus23->setAxisTitle("Residuals(x) (measurement frame) w=2",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus3) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus3 = bookMEProfile("TProfResMFTrackwidthProfile_WClus3","TProfResMFTrackwidthProfile_WClus3","Profile of Residuals(x) in MF vs track width for w=3");
    simplehitsMEs.meResMFTrackwidthProfileWClus3->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus3->setAxisTitle("Residuals(x) (measurement frame) w=3",2);
  } 
  if(layerswitchResMFTrackwidthProfile_WClus4) { 
    simplehitsMEs.meResMFTrackwidthProfileWClus4 = bookMEProfile("TProfResMFTrackwidthProfile_WClus4","TProfResMFTrackwidthProfile_WClus4","Profile of Residuals(x) in MF vs track width for w=4");
    simplehitsMEs.meResMFTrackwidthProfileWClus4->setAxisTitle("Track width",1);
    simplehitsMEs.meResMFTrackwidthProfileWClus4->setAxisTitle("Residuals(x) (measurement frame) w=4",2);
  } 
  if(layerswitchErrxMFTrackwidthProfile) {  
    simplehitsMEs.meErrxMFTrackwidthProfile = bookMEProfile("TProfErrxMFTrackwidthProfile","TProfErrxMFTrackwidthProfile","Profile of Resolution in MF vs track width");
    simplehitsMEs.meErrxMFTrackwidthProfile->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfile->setAxisTitle("Resolution (measurement frame)",2);
  }
  if(layerswitchErrxMFTrackwidthProfile_Category1) {  
    simplehitsMEs.meErrxMFTrackwidthProfileCategory1 = bookMEProfile("TProfErrxMFTrackwidthProfile_Category1","TProfErrxMFTrackwidthProfile_Category1","Profile of Resolution in MF vs track width (Category 1)");
    simplehitsMEs.meErrxMFTrackwidthProfileCategory1->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfileCategory1->setAxisTitle("Resolution (measurement frame) Category 1",2);
  }
  if(layerswitchErrxMFTrackwidthProfile_Category2) {  
    simplehitsMEs.meErrxMFTrackwidthProfileCategory2 = bookMEProfile("TProfErrxMFTrackwidthProfile_Category2","TProfErrxMFTrackwidthProfile_Category2","Profile of Resolution in MF vs track width (Category 2)");
    simplehitsMEs.meErrxMFTrackwidthProfileCategory2->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfileCategory2->setAxisTitle("Resolution (measurement frame) Category 2",2);
  }
  if(layerswitchErrxMFTrackwidthProfile_Category3) { 
    simplehitsMEs.meErrxMFTrackwidthProfileCategory3 = bookMEProfile("TProfErrxMFTrackwidthProfile_Category3","TProfErrxMFTrackwidthProfile_Category3","Profile of Resolution in MF vs track width (Category 3)");
    simplehitsMEs.meErrxMFTrackwidthProfileCategory3->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfileCategory3->setAxisTitle("Resolution (measurement frame) Category 3",2);
  }
  if(layerswitchErrxMFTrackwidthProfile_Category4) { 
    simplehitsMEs.meErrxMFTrackwidthProfileCategory4 = bookMEProfile("TProfErrxMFTrackwidthProfile_Category4","TProfErrxMFTrackwidthProfile_Category4","Profile of Resolution in MF vs track width (Category 4)");
    simplehitsMEs.meErrxMFTrackwidthProfileCategory4->setAxisTitle("Track width",1);
    simplehitsMEs.meErrxMFTrackwidthProfileCategory4->setAxisTitle("Resolution (measurement frame) Category 4",2);
  }
  if(layerswitchErrxMFClusterwidthProfile_Category1) {
    simplehitsMEs.meErrxMFClusterwidthProfileCategory1 = bookMEProfile("TProfErrxMFClusterwidthProfile_Category1","TProfErrxMFClusterwidthProfile_Category1","Profile of Resolution in MF vs cluster width (Category 1)");
    simplehitsMEs.meErrxMFClusterwidthProfileCategory1->setAxisTitle("Cluster width",1);
    simplehitsMEs.meErrxMFClusterwidthProfileCategory1->setAxisTitle("Resolution (measurement frame) Category 1",2);
  }
  if(layerswitchErrxMFAngleProfile) { 
    simplehitsMEs.meErrxMFAngleProfile = bookMEProfile("TProfErrxMFAngleProfile","TProfErrxMFAngleProfile","Profile of Resolution in MF vs Track angle alpha");
    simplehitsMEs.meErrxMFAngleProfile->setAxisTitle("Track angle alpha",1);
    simplehitsMEs.meErrxMFAngleProfile->setAxisTitle("Resolution (measurement frame)",2);
  } 

         
}
//------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::createLayerMEs(std::string label) 
{
  SiStripHistoId hidmanager;
  LayerMEs layerMEs; 

  layerMEs.meNstpRphi = 0;
  layerMEs.meAdcRphi = 0;
  layerMEs.mePosxRphi = 0;
  layerMEs.meErrxLFRphi = 0;
  layerMEs.meErrxMFRphi = 0;
  layerMEs.meErrxMFRphiwclus1 = 0;
  layerMEs.meErrxMFRphiwclus2 = 0;
  layerMEs.meErrxMFRphiwclus3 = 0;
  layerMEs.meErrxMFRphiwclus4 = 0;
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
  layerMEs.meErrxMFTrackwidthProfileRphi = 0;

  layerMEs.meErrxMFTrackwidthProfileWclus1Rphi = 0;
  layerMEs.meErrxMFTrackwidthProfileWclus2Rphi = 0;
  layerMEs.meErrxMFTrackwidthProfileWclus3Rphi = 0;
  layerMEs.meErrxMFTrackwidthProfileWclus4Rphi = 0;
  layerMEs.meResMFTrackwidthProfileWclus1Rphi = 0;
  layerMEs.meResMFTrackwidthProfileWclus2Rphi = 0;
  layerMEs.meResMFTrackwidthProfileWclus3Rphi = 0;
  layerMEs.meResMFTrackwidthProfileWclus4Rphi = 0;

  layerMEs.meErrxMFTrackwidthProfileCategory1Rphi = 0;
  layerMEs.meErrxMFTrackwidthProfileCategory2Rphi = 0;
  layerMEs.meErrxMFTrackwidthProfileCategory3Rphi = 0;
  layerMEs.meErrxMFTrackwidthProfileCategory4Rphi = 0;
  layerMEs.meErrxMFClusterwidthProfileCategory1Rphi = 0;
  layerMEs.meErrxMFAngleProfileRphi = 0;
  layerMEs.merapidityResProfilewclus1 = 0;
  layerMEs.merapidityResProfilewclus2 = 0;
  layerMEs.merapidityResProfilewclus3 = 0;
  layerMEs.merapidityResProfilewclus4 = 0;
      

  //NstpRphi
  if(layerswitchNstpRphi) {
    layerMEs.meNstpRphi = bookME1D("TH1NstpRphi", hidmanager.createHistoLayer("Nstp_rphi","layer",label,"").c_str() ,"Cluster Width - Number of strips that belong to the RecHit cluster"); 
    layerMEs.meNstpRphi->setAxisTitle(("Cluster Width [nr strips] in "+ label).c_str());
  }
  //AdcRphi
  if(layerswitchAdcRphi) {
    layerMEs.meAdcRphi = bookME1D("TH1AdcRphi", hidmanager.createHistoLayer("Adc_rphi","layer",label,"").c_str() ,"RecHit Cluster Charge");
    layerMEs.meAdcRphi->setAxisTitle(("cluster charge [ADC] in " + label).c_str());
  }
  //PosxRphi
  if(layerswitchPosxRphi) {
    layerMEs.mePosxRphi = bookME1D("TH1PosxRphi", hidmanager.createHistoLayer("Posx_rphi","layer",label,"").c_str() ,"RecHit x coord."); 
    layerMEs.mePosxRphi->setAxisTitle(("x RecHit coord. (local frame) in " + label).c_str());
  }
  //ErrxLFRphi
  if(layerswitchErrxLFRphi) {
    layerMEs.meErrxLFRphi = bookME1D("TH1ErrxLFRphi", hidmanager.createHistoLayer("Errx_LF_rphi","layer",label,"").c_str() ,"RecHit err(x) coord.");   //<error>~20micron  
    layerMEs.meErrxLFRphi->setAxisTitle(("err(x) RecHit coord. (local frame) in " + label).c_str());
  }
  //ErrxMFRphi
  if(layerswitchErrxMFRphi) {
    layerMEs.meErrxMFRphi = bookME1D("TH1ErrxMFRphi", hidmanager.createHistoLayer("Errx_MF_rphi","layer",label,"").c_str() ,"RecHit err(x) coord.");   //<error>~20micron  
    layerMEs.meErrxMFRphi->setAxisTitle(("err(x) RecHit coord. (measurement frame) in " + label).c_str());
  }
  //ErrxMFRphiwclus1
  if(layerswitchErrxMFRphiwclus1) {
    layerMEs.meErrxMFRphiwclus1 = bookME1D("TH1ErrxMFRphiwclus1", hidmanager.createHistoLayer("Errx_MF_wclus1_rphi","layer",label,"").c_str() ,"RecHit err(x) coord. w=1 ");   //<error>~20micron  
    layerMEs.meErrxMFRphiwclus1->setAxisTitle(("err(x) RecHit coord. (measurement frame) for w=1 in " + label).c_str());
  }
  //ErrxMFRphiwclus2
  if(layerswitchErrxMFRphiwclus2) {
    layerMEs.meErrxMFRphiwclus2 = bookME1D("TH1ErrxMFRphiwclus2", hidmanager.createHistoLayer("Errx_MF_wclus2_rphi","layer",label,"").c_str() ,"RecHit err(x) coord. w=2 ");   //<error>~20micron  
    layerMEs.meErrxMFRphiwclus2->setAxisTitle(("err(x) RecHit coord. (measurement frame) for w=2 in " + label).c_str());
  }
  //ErrxMFRphiwclus3
  if(layerswitchErrxMFRphiwclus3) {
    layerMEs.meErrxMFRphiwclus3 = bookME1D("TH1ErrxMFRphiwclus3", hidmanager.createHistoLayer("Errx_MF_wclus3_rphi","layer",label,"").c_str() ,"RecHit err(x) coord. w=3 ");   //<error>~20micron  
    layerMEs.meErrxMFRphiwclus3->setAxisTitle(("err(x) RecHit coord. (measurement frame) for w=3 in " + label).c_str());
  }
  //ErrxMFRphiwclus4
  if(layerswitchErrxMFRphiwclus4) {
    layerMEs.meErrxMFRphiwclus4 = bookME1D("TH1ErrxMFRphiwclus4", hidmanager.createHistoLayer("Errx_MF_wclus4_rphi","layer",label,"").c_str() ,"RecHit err(x) coord. w=4 ");   //<error>~20micron  
    layerMEs.meErrxMFRphiwclus4->setAxisTitle(("err(x) RecHit coord. (measurement frame) for w=4 in " + label).c_str());
  }
  //ResLFRphi
  if(layerswitchResLFRphi) {
    layerMEs.meResLFRphi = bookME1D("TH1ResLFRphi", hidmanager.createHistoLayer("Res_LF_rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate"); 
    layerMEs.meResLFRphi->setAxisTitle(("Hit Residuals(x) (local frame) in " + label).c_str());
  }
  //ResMFRphi
  if(layerswitchResMFRphi) {
    layerMEs.meResMFRphi = bookME1D("TH1ResMFRphi",hidmanager.createHistoLayer("Res_MF_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate");
    layerMEs.meResMFRphi->setAxisTitle(("Hit Residuals(x) (measurement frame) in "+ label).c_str());
  }
  //ResMFRphiwclus1
  if(layerswitchResMFRphiwclus1) {
    layerMEs.meResMFRphiwclus1 = bookME1D("TH1ResMFRphiwclus1",hidmanager.createHistoLayer("Res_MF_wclus1_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate w=1");
    layerMEs.meResMFRphiwclus1->setAxisTitle(("Hit Residuals(x) (measurement frame) for w=1 in "+ label).c_str());
  }
  //ResMFRphiwclus2
  if(layerswitchResMFRphiwclus2) {
    layerMEs.meResMFRphiwclus2 = bookME1D("TH1ResMFRphiwclus2",hidmanager.createHistoLayer("Res_MF_wclus2_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate w=2");
    layerMEs.meResMFRphiwclus2->setAxisTitle(("Hit Residuals(x) (measurement frame) for w=2 in "+ label).c_str());
  }
  //ResMFRphiwclus3
  if(layerswitchResMFRphiwclus3) {
    layerMEs.meResMFRphiwclus3 = bookME1D("TH1ResMFRphiwclus3",hidmanager.createHistoLayer("Res_MF_wclus3_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate w=3");
    layerMEs.meResMFRphiwclus3->setAxisTitle(("Hit Residuals(x) (measurement frame) for w=3 in "+ label).c_str());
  }
  //ResMFRphiwclus4
  if(layerswitchResMFRphiwclus4) {
    layerMEs.meResMFRphiwclus4 = bookME1D("TH1ResMFRphiwclus4",hidmanager.createHistoLayer("Res_MF_wclus4_Rphi","layer",label,"").c_str() ,"Residual of the hit x coordinate w=4");
    layerMEs.meResMFRphiwclus4->setAxisTitle(("Hit Residuals(x) (measurement frame) for w=4 in "+ label).c_str());
  }
  //PullLFRphi
  if(layerswitchPullLFRphi) {
    layerMEs.mePullLFRphi = bookME1D("TH1PullLFRphi", hidmanager.createHistoLayer("Pull_LF_rphi","layer",label,"").c_str() ,"Pull distribution");  
    layerMEs.mePullLFRphi->setAxisTitle(("Pull distribution (local frame) in " + label).c_str());
  }
  //PullMFRphi
  if(layerswitchPullMFRphi) {
    layerMEs.mePullMFRphi = bookME1D("TH1PullMFRphi", hidmanager.createHistoLayer("Pull_MF_rphi","layer",label,"").c_str() ,"Pull distribution");  
    layerMEs.mePullMFRphi->setAxisTitle(("Pull distribution (measurement frame) in " + label).c_str());
  }
  //PullMFRphiwclus1
  if(layerswitchPullMFRphiwclus1) {
    layerMEs.mePullMFRphiwclus1 = bookME1D("TH1PullMFRphiwclus1", hidmanager.createHistoLayer("Pull_MF_wclus1_rphi","layer",label,"").c_str() ,"Pull distribution w=1");  
    layerMEs.mePullMFRphiwclus1->setAxisTitle(("Pull distribution (measurement frame) for w=1 in " + label).c_str());
  }
  //PullMFRphiwclus2
  if(layerswitchPullMFRphiwclus2) {
    layerMEs.mePullMFRphiwclus2 = bookME1D("TH1PullMFRphiwclus2", hidmanager.createHistoLayer("Pull_MF_wclus2_rphi","layer",label,"").c_str() ,"Pull distribution w=2");  
    layerMEs.mePullMFRphiwclus2->setAxisTitle(("Pull distribution (measurement frame) for w=2 in " + label).c_str());
  }
  //PullMFRphiwclus3
  if(layerswitchPullMFRphiwclus3) {
    layerMEs.mePullMFRphiwclus3 = bookME1D("TH1PullMFRphiwclus3", hidmanager.createHistoLayer("Pull_MF_wclus3_rphi","layer",label,"").c_str() ,"Pull distribution w=3");  
    layerMEs.mePullMFRphiwclus3->setAxisTitle(("Pull distribution (measurement frame) for w=3 in " + label).c_str());
  }
  //PullMFRphiwclus4
  if(layerswitchPullMFRphiwclus4) {
    layerMEs.mePullMFRphiwclus4 = bookME1D("TH1PullMFRphiwclus4", hidmanager.createHistoLayer("Pull_MF_wclus4_rphi","layer",label,"").c_str() ,"Pull distribution w=4");  
    layerMEs.mePullMFRphiwclus4->setAxisTitle(("Pull distribution (measurement frame) for w=4 in " + label).c_str());
  }

  if(layerswitchTrackangleRphi) {
    layerMEs.meTrackangleRphi = bookME1D("TH1TrackangleRphi",hidmanager.createHistoLayer("Track_angle_Rphi","layer",label,"").c_str() ,"Track angle alpha");
    layerMEs.meTrackangleRphi->setAxisTitle(("Track angle in "+ label).c_str());
  }
  if(layerswitchTrackanglebetaRphi) {
    layerMEs.meTrackanglebetaRphi = bookME1D("TH1TrackanglebetaRphi",hidmanager.createHistoLayer("Track_angle_beta_Rphi","layer",label,"").c_str() ,"Track angle beta");
    layerMEs.meTrackanglebetaRphi->setAxisTitle((""+ label).c_str());
  }
  if(layerswitchTrackangle2Rphi) {
    layerMEs.meTrackangle2Rphi = bookME1D("TH1Trackangle2Rphi",hidmanager.createHistoLayer("Track_angle2_Rphi","layer",label,"").c_str() ,"");
    layerMEs.meTrackangle2Rphi->setAxisTitle((""+ label).c_str());
  }
  if(layerswitchPullTrackangleProfileRphi) {
    layerMEs.mePullTrackangleProfileRphi = bookMEProfile("TProfPullTrackangleProfileRphi",hidmanager.createHistoLayer("Pull_Trackangle_Profile_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track angle alpha");
    layerMEs.mePullTrackangleProfileRphi->setAxisTitle(("Track angle alpha in "+ label).c_str(),1);
    layerMEs.mePullTrackangleProfileRphi->setAxisTitle(("Pull (MF) in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackangle2DRphi) {
    layerMEs.mePullTrackangle2DRphi = bookME1D("TH1PullTrackangle2DRphi",hidmanager.createHistoLayer("Pull_Trackangle_2D_Rphi","layer",label,"").c_str() ,"");
    layerMEs.mePullTrackangle2DRphi->setAxisTitle((""+ label).c_str());
  }
  if(layerswitchTrackwidthRphi) {
    layerMEs.meTrackwidthRphi = bookME1D("TH1TrackwidthRphi",hidmanager.createHistoLayer("Track_width_Rphi","layer",label,"").c_str() ,"Track width");
    layerMEs.meTrackwidthRphi->setAxisTitle(("Track width in "+ label).c_str());
  }
  if(layerswitchExpectedwidthRphi) {
    layerMEs.meExpectedwidthRphi = bookME1D("TH1ExpectedwidthRphi",hidmanager.createHistoLayer("Expected_width_Rphi","layer",label,"").c_str() ,"Expected width");
    layerMEs.meExpectedwidthRphi->setAxisTitle(("Expected width in "+ label).c_str());
  }
  if(layerswitchClusterwidthRphi) {
    layerMEs.meClusterwidthRphi = bookME1D("TH1ClusterwidthRphi",hidmanager.createHistoLayer("Cluster_width_Rphi","layer",label,"").c_str() ,"Cluster width");
    layerMEs.meClusterwidthRphi->setAxisTitle(("Cluster width in "+ label).c_str());
  }
  if(layerswitchCategoryRphi) {
    layerMEs.meCategoryRphi = bookME1D("TH1CategoryRphi",hidmanager.createHistoLayer("Category_Rphi","layer",label,"").c_str() ,"Category");
    layerMEs.meCategoryRphi->setAxisTitle(("Category in "+ label).c_str());
  }
  if(layerswitchPullTrackwidthProfileRphi) {
    layerMEs.mePullTrackwidthProfileRphi = bookMEProfile("TProfPullTrackwidthProfileRphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width");
    layerMEs.mePullTrackwidthProfileRphi->setAxisTitle(("track width in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphi->setAxisTitle(("Pull (MF) in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileRphiwclus1) {
    layerMEs.mePullTrackwidthProfileRphiwclus1 = bookMEProfile("TProfPullTrackwidthProfileRphiwclus1",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi_wclus1","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for w=1");
    layerMEs.mePullTrackwidthProfileRphiwclus1->setAxisTitle(("track width for w=1 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphiwclus1->setAxisTitle(("Pull (MF) for w=1 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileRphiwclus2) {
    layerMEs.mePullTrackwidthProfileRphiwclus2 = bookMEProfile("TProfPullTrackwidthProfileRphiwclus2",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi_wclus2","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for w=2");
    layerMEs.mePullTrackwidthProfileRphiwclus2->setAxisTitle(("track width for w=2 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphiwclus2->setAxisTitle(("Pull (MF) for w=2 in "+ label).c_str(),2);

  }
  if(layerswitchPullTrackwidthProfileRphiwclus3) {
    layerMEs.mePullTrackwidthProfileRphiwclus3 = bookMEProfile("TProfPullTrackwidthProfileRphiwclus3",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi_wclus3","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for w=3");
    layerMEs.mePullTrackwidthProfileRphiwclus3->setAxisTitle(("track width for w=3 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphiwclus3->setAxisTitle(("Pull (MF) for w=3 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileRphiwclus4) {
    layerMEs.mePullTrackwidthProfileRphiwclus4 = bookMEProfile("TProfPullTrackwidthProfileRphiwclus4",hidmanager.createHistoLayer("Pull_Track_width_Profile_Rphi_wclus4","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for w=4");
    layerMEs.mePullTrackwidthProfileRphiwclus4->setAxisTitle(("track width for w=4 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileRphiwclus4->setAxisTitle(("Pull (MF) for w=4 in "+ label).c_str(),2);

  }
  if(layerswitchPullTrackwidthProfileCategory1Rphi) {
    layerMEs.mePullTrackwidthProfileCategory1Rphi = bookMEProfile("TProfPullTrackwidthProfileCategory1Rphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category1_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for Category 1");
    layerMEs.mePullTrackwidthProfileCategory1Rphi->setAxisTitle(("track width for Category 1 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileCategory1Rphi->setAxisTitle(("Pull (MF) for Category 1 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory2Rphi) {
    layerMEs.mePullTrackwidthProfileCategory2Rphi = bookMEProfile("TProfPullTrackwidthProfileCategory2Rphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category2_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for Category 2");
    layerMEs.mePullTrackwidthProfileCategory2Rphi->setAxisTitle(("track width for Category 2 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileCategory2Rphi->setAxisTitle(("Pull (MF) for Category 2 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory3Rphi) {
    layerMEs.mePullTrackwidthProfileCategory3Rphi = bookMEProfile("TProfPullTrackwidthProfileCategory3Rphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category3_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for Category 3");
    layerMEs.mePullTrackwidthProfileCategory3Rphi->setAxisTitle(("track width for Category 3 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileCategory3Rphi->setAxisTitle(("Pull (MF) for Category 3 in "+ label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory4Rphi) {
    layerMEs.mePullTrackwidthProfileCategory4Rphi = bookMEProfile("TProfPullTrackwidthProfileCategory4Rphi",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category4_Rphi","layer",label,"").c_str() ,"Profile of Pull in MF vs track width for Category 4");
    layerMEs.mePullTrackwidthProfileCategory4Rphi->setAxisTitle(("track width for Category 4 in "+ label).c_str(),1);
    layerMEs.mePullTrackwidthProfileCategory4Rphi->setAxisTitle(("Pull (MF) for Category 4 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileRphi) {
    layerMEs.meErrxMFTrackwidthProfileRphi = bookMEProfile("TProfErrxMFTrackwidthProfileRphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width");
    layerMEs.meErrxMFTrackwidthProfileRphi->setAxisTitle(("track width in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileRphi->setAxisTitle(("Resolution in MF in "+ label).c_str(),2);
  }

  if(layerswitchErrxMFTrackwidthProfileWclus1Rphi) {
    layerMEs.meErrxMFTrackwidthProfileWclus1Rphi = bookMEProfile("TProfErrxMFTrackwidthProfileWclus1Rphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Wclus1_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for w=1");
    layerMEs.meErrxMFTrackwidthProfileWclus1Rphi->setAxisTitle(("track width for w=1 in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileWclus1Rphi->setAxisTitle(("Resolution in MF for w=1 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileWclus2Rphi) {
    layerMEs.meErrxMFTrackwidthProfileWclus2Rphi = bookMEProfile("TProfErrxMFTrackwidthProfileWclus2Rphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Wclus2_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for w=2");
    layerMEs.meErrxMFTrackwidthProfileWclus2Rphi->setAxisTitle(("track width for w=2 in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileWclus2Rphi->setAxisTitle(("Resolution in MF for w=2 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileWclus3Rphi) {
    layerMEs.meErrxMFTrackwidthProfileWclus3Rphi = bookMEProfile("TProfErrxMFTrackwidthProfileWclus3Rphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Wclus3_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for w=3");
    layerMEs.meErrxMFTrackwidthProfileWclus3Rphi->setAxisTitle(("track width for w=3 in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileWclus3Rphi->setAxisTitle(("Resolution in MF for w=3 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileWclus4Rphi) {
    layerMEs.meErrxMFTrackwidthProfileWclus4Rphi = bookMEProfile("TProfErrxMFTrackwidthProfileWclus4Rphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Wclus4_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for w=4");
    layerMEs.meErrxMFTrackwidthProfileWclus4Rphi->setAxisTitle(("track width for w=4 in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileWclus4Rphi->setAxisTitle(("Resolution in MF for w=4 in "+ label).c_str(),2);
  }
  if(layerswitchResMFTrackwidthProfileWclus1Rphi) {
    layerMEs.meResMFTrackwidthProfileWclus1Rphi = bookMEProfile("TProfResMFTrackwidthProfileWclus1Rphi",hidmanager.createHistoLayer("ResMF_Track_width_Profile_Wclus1_Rphi","layer",label,"").c_str() ,"Profile of Residuals(x) in MF vs track width for w=1");
    layerMEs.meResMFTrackwidthProfileWclus1Rphi->setAxisTitle(("track width for w=1 in "+ label).c_str(),1);
    layerMEs.meResMFTrackwidthProfileWclus1Rphi->setAxisTitle(("Residuals(x) in MF for w=1 in "+ label).c_str(),2);
  }
  if(layerswitchResMFTrackwidthProfileWclus2Rphi) {
    layerMEs.meResMFTrackwidthProfileWclus2Rphi = bookMEProfile("TProfResMFTrackwidthProfileWclus2Rphi",hidmanager.createHistoLayer("ResMF_Track_width_Profile_Wclus2_Rphi","layer",label,"").c_str() ,"Profile of Residuals(x) in MF vs track width for w=2");
    layerMEs.meResMFTrackwidthProfileWclus2Rphi->setAxisTitle(("track width for w=2 in "+ label).c_str(),1);
    layerMEs.meResMFTrackwidthProfileWclus2Rphi->setAxisTitle(("Residuals(x) in MF for w=2 in "+ label).c_str(),2);
  }
  if(layerswitchResMFTrackwidthProfileWclus3Rphi) {
    layerMEs.meResMFTrackwidthProfileWclus3Rphi = bookMEProfile("TProfResMFTrackwidthProfileWclus3Rphi",hidmanager.createHistoLayer("ResMF_Track_width_Profile_Wclus3_Rphi","layer",label,"").c_str() ,"Profile of Residuals(x) in MF vs track width for w=3");
    layerMEs.meResMFTrackwidthProfileWclus3Rphi->setAxisTitle(("track width for w=3 in "+ label).c_str(),1);
    layerMEs.meResMFTrackwidthProfileWclus3Rphi->setAxisTitle(("Residuals(x) in MF for w=3 in "+ label).c_str(),2);
  }
  if(layerswitchResMFTrackwidthProfileWclus4Rphi) {
    layerMEs.meResMFTrackwidthProfileWclus4Rphi = bookMEProfile("TProfResMFTrackwidthProfileWclus4Rphi",hidmanager.createHistoLayer("ResMF_Track_width_Profile_Wclus4_Rphi","layer",label,"").c_str() ,"Profile of Residuals(x) in MF vs track width for w=4");
    layerMEs.meResMFTrackwidthProfileWclus4Rphi->setAxisTitle(("track width for w=4 in "+ label).c_str(),1);
    layerMEs.meResMFTrackwidthProfileWclus4Rphi->setAxisTitle(("Residuals(x) in MF for w=4 in "+ label).c_str(),2);
  }

  if(layerswitchErrxMFTrackwidthProfileCategory1Rphi) {
    layerMEs.meErrxMFTrackwidthProfileCategory1Rphi = bookMEProfile("TProfErrxMFTrackwidthProfileCategory1Rphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Category1_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for Category 1");
    layerMEs.meErrxMFTrackwidthProfileCategory1Rphi->setAxisTitle(("track width for Category 1 in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileCategory1Rphi->setAxisTitle(("Resolution in MF for Category 1 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileCategory2Rphi) {
    layerMEs.meErrxMFTrackwidthProfileCategory2Rphi = bookMEProfile("TProfErrxMFTrackwidthProfileCategory2Rphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Category2_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for Category 2");
    layerMEs.meErrxMFTrackwidthProfileCategory2Rphi->setAxisTitle(("track width for Category 2 in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileCategory2Rphi->setAxisTitle(("Resolution in MF for Category 2 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileCategory3Rphi) {
    layerMEs.meErrxMFTrackwidthProfileCategory3Rphi = bookMEProfile("TProfErrxMFTrackwidthProfileCategory3Rphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Category3_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for Category 3");
    layerMEs.meErrxMFTrackwidthProfileCategory3Rphi->setAxisTitle(("track width for Category 3 in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileCategory3Rphi->setAxisTitle(("Resolution in MF for Category 3 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileCategory4Rphi) {
    layerMEs.meErrxMFTrackwidthProfileCategory4Rphi = bookMEProfile("TProfErrxMFTrackwidthProfileCategory4Rphi",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Category3_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width for Category 4");
    layerMEs.meErrxMFTrackwidthProfileCategory3Rphi->setAxisTitle(("track width for Category 4 in "+ label).c_str(),1);
    layerMEs.meErrxMFTrackwidthProfileCategory3Rphi->setAxisTitle(("Resolution in MF for Category 4 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFClusterwidthProfileCategory1Rphi) {
    layerMEs.meErrxMFClusterwidthProfileCategory1Rphi = bookMEProfile("TProfErrxMFClusterwidthProfileCategory1Rphi",hidmanager.createHistoLayer("ErrxMF_Cluster_width_Profile_Category1_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs cluster width for Category 1");
    layerMEs.meErrxMFClusterwidthProfileCategory1Rphi->setAxisTitle(("cluster width for Category 1 in "+ label).c_str(),1);
    layerMEs.meErrxMFClusterwidthProfileCategory1Rphi->setAxisTitle(("Resolution in MF for Category 1 in "+ label).c_str(),2);
  }
  if(layerswitchErrxMFAngleProfileRphi) {
    layerMEs.meErrxMFAngleProfileRphi = bookMEProfile("TProfErrxMFAngleProfileRphi",hidmanager.createHistoLayer("ErrxMF_Angle_Profile_Rphi","layer",label,"").c_str() ,"Profile of Resolution in MF vs track angle alpha");
    layerMEs.meErrxMFAngleProfileRphi->setAxisTitle(("track angle alpha in "+ label).c_str(),1);
    layerMEs.meErrxMFAngleProfileRphi->setAxisTitle(("Resolution in MF in "+ label).c_str(),2);
  }
  if(layerswitchrapidityResProfilewclus1) {
    layerMEs.merapidityResProfilewclus1 = bookMEProfile("TProfrapidityResProfilewclus1",hidmanager.createHistoLayer("rapidity_Res_Profile_wclus1","layer",label,"").c_str() ,"Profile of rapidity vs Res for w=1");
    layerMEs.merapidityResProfilewclus1->setAxisTitle(("Res for w=1 in "+ label).c_str(),1);
    layerMEs.merapidityResProfilewclus1->setAxisTitle(("rapidity for w=1 in "+ label).c_str(),2);
  }
  if(layerswitchrapidityResProfilewclus2) {
    layerMEs.merapidityResProfilewclus2 = bookMEProfile("TProfrapidityResProfilewclus2",hidmanager.createHistoLayer("rapidity_Res_Profile_wclus2","layer",label,"").c_str() ,"Profile of rapidity vs Res for w=2");
    layerMEs.merapidityResProfilewclus2->setAxisTitle(("Res for w=2 in "+ label).c_str(),1);
    layerMEs.merapidityResProfilewclus2->setAxisTitle(("rapidity for w=2 in "+ label).c_str(),2);
  }
  if(layerswitchrapidityResProfilewclus3) {
    layerMEs.merapidityResProfilewclus3 = bookMEProfile("TProfrapidityResProfilewclus3",hidmanager.createHistoLayer("rapidity_Res_Profile_wclus3","layer",label,"").c_str() ,"Profile of rapidity vs Res for w=3");
    layerMEs.merapidityResProfilewclus3->setAxisTitle(("Res for w=3 in "+ label).c_str(),1);
    layerMEs.merapidityResProfilewclus3->setAxisTitle(("rapidity for w=3 in "+ label).c_str(),2);
  }
  if(layerswitchrapidityResProfilewclus4) {
    layerMEs.merapidityResProfilewclus4 = bookMEProfile("TProfrapidityResProfilewclus4",hidmanager.createHistoLayer("rapidity_Res_Profile_wclus4","layer",label,"").c_str() ,"Profile of rapidity vs Res for w=4");
    layerMEs.merapidityResProfilewclus4->setAxisTitle(("Res for w=4 in "+ label).c_str(),1);
    layerMEs.merapidityResProfilewclus4->setAxisTitle(("rapidity for w=4 in "+ label).c_str(),2);
  }
      

  LayerMEsMap[label]=layerMEs;
 
}
//------------------------------------------------------------------------------------------
void SiStripTrackingRecHitsValid::createStereoAndMatchedMEs(std::string label) 
{
  SiStripHistoId hidmanager;
  StereoAndMatchedMEs stereoandmatchedMEs; 

  stereoandmatchedMEs.meNstpSas = 0;
  stereoandmatchedMEs.meAdcSas = 0;
  stereoandmatchedMEs.mePosxSas = 0;
  stereoandmatchedMEs.meErrxLFSas = 0;
  stereoandmatchedMEs.meErrxMFSas = 0;
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
  stereoandmatchedMEs.meErrxMFTrackwidthProfileSas = 0;
  stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory1Sas = 0;
  stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory2Sas = 0;
  stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory3Sas = 0;
  stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory4Sas = 0;
  stereoandmatchedMEs.meErrxMFClusterwidthProfileCategory1Sas = 0;
  stereoandmatchedMEs.meErrxMFAngleProfileSas = 0;

  stereoandmatchedMEs.mePosxMatched = 0;
  stereoandmatchedMEs.mePosyMatched = 0;
  stereoandmatchedMEs.meErrxMatched = 0;
  stereoandmatchedMEs.meErryMatched = 0;
  stereoandmatchedMEs.meResxMatched = 0;
  stereoandmatchedMEs.meResyMatched = 0;
  stereoandmatchedMEs.mePullxMatched = 0;
  stereoandmatchedMEs.mePullyMatched = 0;

  //NstpSas
  if(layerswitchNstpSas) {
    stereoandmatchedMEs.meNstpSas = bookME1D("TH1NstpSas", hidmanager.createHistoLayer("Nstp_sas","layer",label,"").c_str() ,"Cluster Width - Number of strips that belong to the RecHit cluster");  
    stereoandmatchedMEs.meNstpSas->setAxisTitle(("Cluster Width [nr strips] (stereo) in "+ label).c_str());
  }
  //AdcSas
  if(layerswitchAdcSas) {
    stereoandmatchedMEs.meAdcSas = bookME1D("TH1AdcSas", hidmanager.createHistoLayer("Adc_sas","layer",label,"").c_str() ,"RecHit Cluster Charge"); 
    stereoandmatchedMEs.meAdcSas->setAxisTitle(("cluster charge [ADC] (stereo) in " + label).c_str());
  }
  //PosxSas
  if(layerswitchPosxSas) {
    stereoandmatchedMEs.mePosxSas = bookME1D("TH1PosxSas", hidmanager.createHistoLayer("Posx_sas","layer",label,"").c_str() ,"RecHit x coord."); 
    stereoandmatchedMEs.mePosxSas->setAxisTitle(("x RecHit coord. (local frame) (stereo) in " + label).c_str());
  }
  //ErrxLFSas
  if(layerswitchErrxLFSas) {
    stereoandmatchedMEs.meErrxLFSas = bookME1D("TH1ErrxLFSas", hidmanager.createHistoLayer("Errx_LF_sas","layer",label,"").c_str() ,"RecHit err(x) coord.");  
    stereoandmatchedMEs.meErrxLFSas->setAxisTitle(("err(x) RecHit coord. (local frame) (stereo) in " + label).c_str());
  }
  //ErrxMFSas
  if(layerswitchErrxMFSas) {
    stereoandmatchedMEs.meErrxMFSas = bookME1D("TH1ErrxMFSas", hidmanager.createHistoLayer("Errx_MF_sas","layer",label,"").c_str() ,"RecHit err(x) coord.");  
    stereoandmatchedMEs.meErrxMFSas->setAxisTitle(("err(x) RecHit coord. (measurement frame) (stereo) in " + label).c_str());
  }
  //ResLFSas
  if(layerswitchResLFSas) {
    stereoandmatchedMEs.meResLFSas = bookME1D("TH1ResLFSas", hidmanager.createHistoLayer("Res_LF_sas","layer",label,"").c_str() ,"Residual of the hit x coordinate"); 
    stereoandmatchedMEs.meResLFSas->setAxisTitle(("Hit Residuals(x) (local frame) (stereo) in " + label).c_str());
  }
  //ResMFSas
  if(layerswitchResMFSas) {
    stereoandmatchedMEs.meResMFSas = bookME1D("TH1ResMFSas", hidmanager.createHistoLayer("Res_MF_sas","layer",label,"").c_str() ,"Residual of the hit x coordinate"); 
    stereoandmatchedMEs.meResMFSas->setAxisTitle(("Hit Residuals(x) (stereo) in " + label).c_str());
  }
  //PullLFSas
  if(layerswitchPullLFSas) {
    stereoandmatchedMEs.mePullLFSas = bookME1D("TH1PullLFSas", hidmanager.createHistoLayer("Pull_LF_sas","layer",label,"").c_str() ,"Pull distribution");  
    stereoandmatchedMEs.mePullLFSas->setAxisTitle(("Pull distribution (local frame) (stereo) in " + label).c_str());
  }
  //PullMFSas
  if(layerswitchPullMFSas) {
    stereoandmatchedMEs.mePullMFSas = bookME1D("TH1PullMFSas", hidmanager.createHistoLayer("Pull_MF_sas","layer",label,"").c_str() ,"Pull distribution");  
    stereoandmatchedMEs.mePullMFSas->setAxisTitle(("Pull distribution (measurement frame) (stereo) in " + label).c_str());
  }

  if(layerswitchTrackangleSas) {
    stereoandmatchedMEs.meTrackangleSas = bookME1D("TH1TrackangleSas",hidmanager.createHistoLayer("Track_angle_Sas","layer",label,"").c_str() ,"Track angle");
    stereoandmatchedMEs.meTrackangleSas->setAxisTitle(("Track angle (stereo) in " + label).c_str());
  }
  if(layerswitchTrackanglebetaSas) {
    stereoandmatchedMEs.meTrackanglebetaSas = bookME1D("TH1TrackanglebetaSas",hidmanager.createHistoLayer("Track_angle_beta_Sas","layer",label,"").c_str() ,"Track angle beta");
    stereoandmatchedMEs.meTrackanglebetaSas->setAxisTitle(("Track angle beta (stereo) in " + label).c_str());
  }
  if(layerswitchPullTrackangleProfileSas) {
    stereoandmatchedMEs.mePullTrackangleProfileSas = bookMEProfile("TProfPullTrackangleProfileSas",hidmanager.createHistoLayer("Pull_Track_angle_Profile_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track angle (stereo)");
    stereoandmatchedMEs.mePullTrackangleProfileSas->setAxisTitle(("track angle (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackangleProfileSas->setAxisTitle(("Pull in MF (stereo) in " + label).c_str(),2);
  }
  if(layerswitchTrackwidthSas) {
    stereoandmatchedMEs.meTrackwidthSas = bookME1D("TH1TrackwidthSas",hidmanager.createHistoLayer("Track_width_Sas","layer",label,"").c_str() ,"Track width");
    stereoandmatchedMEs.meTrackwidthSas->setAxisTitle(("Track width (stereo) in " + label).c_str());
  }
  if(layerswitchExpectedwidthSas) {
    stereoandmatchedMEs.meExpectedwidthSas = bookME1D("TH1ExpectedwidthSas",hidmanager.createHistoLayer("Expected_width_Sas","layer",label,"").c_str() ,"Expected width");
    stereoandmatchedMEs.meExpectedwidthSas->setAxisTitle(("Expected width (stereo) in " + label).c_str());
  }
  if(layerswitchClusterwidthSas) {
    stereoandmatchedMEs.meClusterwidthSas = bookME1D("TH1ClusterwidthSas",hidmanager.createHistoLayer("Cluster_width_Sas","layer",label,"").c_str() ,"Cluster width");
    stereoandmatchedMEs.meClusterwidthSas->setAxisTitle(("Cluster width (stereo) in " + label).c_str());
  }
  if(layerswitchCategorySas) {
    stereoandmatchedMEs.meCategorySas = bookME1D("TH1CategorySas",hidmanager.createHistoLayer("Category_Sas","layer",label,"").c_str() ,"Category");
    stereoandmatchedMEs.meCategorySas->setAxisTitle(("Category (stereo) in " + label).c_str());
  }
  if(layerswitchPullTrackwidthProfileSas) {
    stereoandmatchedMEs.mePullTrackwidthProfileSas = bookMEProfile("TProfPullTrackwidthProfileSas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileSas->setAxisTitle(("track width (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileSas->setAxisTitle(("Pull in MF (stereo) in " + label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory1Sas) {
    stereoandmatchedMEs.mePullTrackwidthProfileCategory1Sas = bookMEProfile("TProfPullTrackwidthProfileCategory1Sas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category1_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (Category 1) (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileCategory1Sas->setAxisTitle(("track width (Category 1) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileCategory1Sas->setAxisTitle(("Pull in MF (Category 1) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory2Sas) {
    stereoandmatchedMEs.mePullTrackwidthProfileCategory2Sas = bookMEProfile("TProfPullTrackwidthProfileCategory2Sas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category2_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (Category 2) (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileCategory2Sas->setAxisTitle(("track width (Category 2) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileCategory2Sas->setAxisTitle(("Pull in MF (Category 2) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory3Sas) {
    stereoandmatchedMEs.mePullTrackwidthProfileCategory3Sas = bookMEProfile("TProfPullTrackwidthProfileCategory3Sas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category3_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (Category 3) (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileCategory3Sas->setAxisTitle(("track width (Category 3) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileCategory3Sas->setAxisTitle(("Pull in MF (Category 3) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchPullTrackwidthProfileCategory4Sas) {
    stereoandmatchedMEs.mePullTrackwidthProfileCategory4Sas = bookMEProfile("TProfPullTrackwidthProfileCategory4Sas",hidmanager.createHistoLayer("Pull_Track_width_Profile_Category4_Sas","layer",label,"").c_str() ,"Profile of Pull in MF vs track width (Category 4) (stereo)");
    stereoandmatchedMEs.mePullTrackwidthProfileCategory4Sas->setAxisTitle(("track width (Category 4) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.mePullTrackwidthProfileCategory4Sas->setAxisTitle(("Pull in MF (Category 4) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileSas) {
    stereoandmatchedMEs.meErrxMFTrackwidthProfileSas = bookMEProfile("TProfErrxMFTrackwidthProfileSas",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (stereo)");
    stereoandmatchedMEs.meErrxMFTrackwidthProfileSas->setAxisTitle(("track width (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meErrxMFTrackwidthProfileSas->setAxisTitle(("Resolution in MF (stereo) in " + label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileCategory1Sas) {
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory1Sas = bookMEProfile("TProfErrxMFTrackwidthProfileCategory1Sas",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Category1_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (Category 1) (stereo)");
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory1Sas->setAxisTitle((" track width (Category 1) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory1Sas->setAxisTitle(("  Resolution in MF (Category 1) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileCategory2Sas) {
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory2Sas = bookMEProfile("TProfErrxMFTrackwidthProfileCategory2Sas",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Category2_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (Category 2) (stereo)");
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory2Sas->setAxisTitle((" track width (Category 2) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory2Sas->setAxisTitle((" Resolution in MF (Category 2) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileCategory3Sas) {
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory3Sas = bookMEProfile("TProfErrxMFTrackwidthProfileCategory3Sas",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Category3_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (Category 3) (stereo)");
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory3Sas->setAxisTitle((" track width (Category 3) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory3Sas->setAxisTitle((" Resolution in MF (Category 3) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchErrxMFTrackwidthProfileCategory4Sas) {
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory4Sas = bookMEProfile("TProfErrxMFTrackwidthProfileCategory4Sas",hidmanager.createHistoLayer("ErrxMF_Track_width_Profile_Category4_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track width (Category 4) (stereo)");
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory4Sas->setAxisTitle((" track width (Category 4) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meErrxMFTrackwidthProfileCategory4Sas->setAxisTitle((" Resolution in MF (Category 4) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchErrxMFClusterwidthProfileCategory1Sas) {
    stereoandmatchedMEs.meErrxMFClusterwidthProfileCategory1Sas = bookMEProfile("TProfErrxMFClusterwidthProfileCategory1Sas",hidmanager.createHistoLayer("ErrxMF_Cluster_width_Profile_Category1_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs cluster width (Category 1) (stereo)");
    stereoandmatchedMEs.meErrxMFClusterwidthProfileCategory1Sas->setAxisTitle(("cluster width (Category 1) (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meErrxMFClusterwidthProfileCategory1Sas->setAxisTitle((" Resolution in MF (Category 1) (stereo) in " + label).c_str(),2);
  }
  if(layerswitchErrxMFAngleProfileSas) {
    stereoandmatchedMEs.meErrxMFAngleProfileSas = bookMEProfile("TProfErrxMFAngleProfileSas",hidmanager.createHistoLayer("ErrxMF_Angle_Profile_Sas","layer",label,"").c_str() ,"Profile of Resolution in MF vs track angle (stereo)");
    stereoandmatchedMEs.meErrxMFAngleProfileSas->setAxisTitle(("track angle (stereo) in " + label).c_str(),1);
    stereoandmatchedMEs.meErrxMFAngleProfileSas->setAxisTitle(("Resolution in MF (stereo) in " + label).c_str(),2);
  }
  //PosxMatched
  if(layerswitchPosxMatched) {
    stereoandmatchedMEs.mePosxMatched = bookME1D("TH1PosxMatched", hidmanager.createHistoLayer("Posx_matched","layer",label,"").c_str() ,"RecHit x coord.");  
    stereoandmatchedMEs.mePosxMatched->setAxisTitle(("x coord. matched RecHit (local frame) in " + label).c_str());
  }
  //PosyMatched
  if(layerswitchPosyMatched) {
    stereoandmatchedMEs.mePosyMatched = bookME1D("TH1PosyMatched", hidmanager.createHistoLayer("Posy_matched","layer",label,"").c_str() ,"RecHit y coord."); 
    stereoandmatchedMEs.mePosyMatched->setAxisTitle(("y coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ErrxMatched
  if(layerswitchErrxMatched) {
    stereoandmatchedMEs.meErrxMatched = bookME1D("TH1ErrxMatched", hidmanager.createHistoLayer("Errx_matched","layer",label,"").c_str() ,"RecHit err(x) coord.");  
    stereoandmatchedMEs.meErrxMatched->setAxisTitle(("err(x) coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ErryMatched
  if(layerswitchErryMatched) {
    stereoandmatchedMEs.meErryMatched = bookME1D("TH1ErryMatched", hidmanager.createHistoLayer("Erry_matched","layer",label,"").c_str() ,"RecHit err(y) coord."); 
    stereoandmatchedMEs.meErryMatched->setAxisTitle(("err(y) coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ResxMatched
  if(layerswitchResxMatched) {
    stereoandmatchedMEs.meResxMatched = bookME1D("TH1ResxMatched", hidmanager.createHistoLayer("Resx_matched","layer",label,"").c_str() ,"Residual of the hit x coord."); 
    stereoandmatchedMEs.meResxMatched->setAxisTitle(("Residuals(x) in matched RecHit in " + label).c_str());
  }
  //ResyMatched
  if(layerswitchResyMatched) {
    stereoandmatchedMEs.meResyMatched = bookME1D("TH1ResyMatched", hidmanager.createHistoLayer("Resy_matched","layer",label,"").c_str() ,"Residual of the hit x coord."); 
    stereoandmatchedMEs.meResyMatched->setAxisTitle(("Res(y) in matched RecHit in " + label).c_str());
  }

  StereoAndMatchedMEsMap[label]=stereoandmatchedMEs;
 
}
//------------------------------------------------------------------------------------------
MonitorElement* SiStripTrackingRecHitsValid::bookME1D(const char* ParameterSetLabel, const char* HistoName, const char* HistoTitle)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return dbe_->book1D(HistoName,HistoTitle,
		      Parameters.getParameter<int32_t>("Nbinx"),
		      Parameters.getParameter<double>("xmin"),
		      Parameters.getParameter<double>("xmax")
		      );
}
//------------------------------------------------------------------------------------------
MonitorElement* SiStripTrackingRecHitsValid::bookMEProfile(const char* ParameterSetLabel, const char* HistoName, const char* HistoTitle)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  //The number of channels in Y is disregarded in a profile plot.
  return dbe_->bookProfile(HistoName,HistoTitle,
				Parameters.getParameter<int32_t>("Nbinx"),
				Parameters.getParameter<double>("xmin"),
				Parameters.getParameter<double>("xmax"),
				Parameters.getParameter<double>("ymin"),
				Parameters.getParameter<double>("ymax"),
				"" 
				);
}

// DEFINE_FWK_MODULE(SiStripTrackingRecHitsValid);
