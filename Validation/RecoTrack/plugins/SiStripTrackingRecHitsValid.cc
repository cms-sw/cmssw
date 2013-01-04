// File: SiStripTrackingRecHitsValid.cc
// // Author:  Arnaud Gay.
// Creation Date:  July 2006.
//
//--------------------------------------------

#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "Validation/RecoTrack/interface/SiStripTrackingRecHitsValid.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

using namespace std;

// ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1.h"
#include "TH2.h"
class TFile;

SiStripTrackingRecHitsValid::SiStripTrackingRecHitsValid(const edm::ParameterSet& ps):dbe_(0)
{

  conf_ = ps;

  // slices

  //Read config file
  //MTCCtrack_ = ps.getParameter<bool>("MTCCtrack");
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "striptrackingrechitshisto.root");
  //src_ = ps.getUntrackedParameter<std::string>( "src" );
  //builderName_ = ps.getParameter<std::string>("TTRHBuilder");   

  // Book histograms
  dbe_ = edm::Service<DQMStore>().operator->();
  //  dbe_->showDirStructure();

  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Strip/ALL");

  Char_t histo[200];

  sprintf(histo,"Errx_LF");
  meErrxLF = dbe_->book1D(histo,"RecHit err(x) Local Frame coord.",100,0,0.005);  
  //  const float Entries1 = meErrxLF->getEntries();

  sprintf(histo,"Errx_MF");
  meErrxMF = dbe_->book1D(histo,"RecHit err(x) Meas. Frame coord.",100,0,0.5);  

  sprintf(histo,"Res_LF");
  meResLF = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  

  sprintf(histo,"Res_MF");
  meResMF = dbe_->book1D(histo,"RecHit Residual",100,-2,+2);  

  sprintf(histo,"Pull_LF");
  mePullLF = dbe_->book1D(histo,"Pull",100,-5.,5.);  

  sprintf(histo,"Pull_MF");
  mePullMF = dbe_->book1D(histo,"Pull",100,-5.,5.);  

  sprintf(histo,"Category");
  meCategory = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

  sprintf(histo,"Trackwidth");
  meTrackwidth = dbe_->book1D(histo,"Track width",100,0.,4.);  

  sprintf(histo,"Expectedwidth");
  meExpectedwidth = dbe_->book1D(histo,"Expected width",10,0.,10.);  

  sprintf(histo,"Clusterwidth");
  meClusterwidth = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

  sprintf(histo,"Trackanglealpha");
  meTrackanglealpha = dbe_->book1D(histo,"Track angle alpha",100,-100.,100.);  

  sprintf(histo,"Trackanglebeta");
  meTrackanglebeta = dbe_->book1D(histo,"Track angle beta",100,-100.,100.);  

  sprintf(histo,"ErrxMFTrackwidthProfile_WClus1");
  meErrxMFTrackwidthProfileWClus1 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"s");

  sprintf(histo,"ErrxMFTrackwidthProfile_WClus2");
  meErrxMFTrackwidthProfileWClus2 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"");

  sprintf(histo,"ErrxMFTrackwidthProfile_WClus3");
  meErrxMFTrackwidthProfileWClus3 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"s");

  sprintf(histo,"ErrxMFTrackwidthProfile_WClus4");
  meErrxMFTrackwidthProfileWClus4 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"s");

  sprintf(histo,"ResMFTrackwidthProfile_WClus1");
  meResMFTrackwidthProfileWClus1 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"s");

  sprintf(histo,"ResMFTrackwidthProfile_WClus2");
  meResMFTrackwidthProfileWClus2 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"");
  sprintf(histo,"ResMFTrackwidthProfile_WClus21");
  meResMFTrackwidthProfileWClus21 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, -2.,2.,"");
  sprintf(histo,"ResMFTrackwidthProfile_WClus22");
  meResMFTrackwidthProfileWClus22 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, -5.,5.,"");
  sprintf(histo,"ResMFTrackwidthProfile_WClus23");
  meResMFTrackwidthProfileWClus23 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, -0.5,0.5,"");
  //sprintf(histo,"ResMFTrackwidthProfile_WClus2");
  //meResMFTrackwidthProfileWClus22 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4., 0.,2.,"s");

  sprintf(histo,"ResMFTrackwidthProfile_WClus3");
  meResMFTrackwidthProfileWClus3 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"s");

  sprintf(histo,"ResMFTrackwidthProfile_WClus4");
  meResMFTrackwidthProfileWClus4 = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"s");

  sprintf(histo,"ErrxMFTrackwidthProfile");
  meErrxMFTrackwidthProfile = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, 0.,2.,"s");

  sprintf(histo,"ErrxMFTrackwidthProfile_Category1");
  meErrxMFTrackwidthProfileCategory1 = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12, 0., 4.,100, -2.,2.,"s");
  sprintf(histo,"ErrxMFTrackwidthProfile_Category2");
  meErrxMFTrackwidthProfileCategory2 = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12, 0., 4.,100, -2.,2.,"s");

  sprintf(histo,"ErrxMFTrackwidthProfile_Category3");
  meErrxMFTrackwidthProfileCategory3 = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12, 0., 4.,100, -2.,2.,"s");

  sprintf(histo,"ErrxMFTrackwidthProfile_Category4");
  meErrxMFTrackwidthProfileCategory4 = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12, 0., 4.,100, -2.,2.,"s");

  sprintf(histo,"ErrxMFClusterwidthProfile_Category1");
   meErrxMFClusterwidthProfileCategory1= dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");

  sprintf(histo,"ErrxMFAngleProfile");
  meErrxMFAngleProfile = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 60.,100, 0.,2.,"s");

  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Strip/TIB");
  //one histo per Layer rphi hits
  for(int i = 0 ;i<4 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtib",i+1);
    meNstpRphiTIB[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%dtib",i+1);
    meAdcRphiTIB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtib",i+1);
    mePosxRphiTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  

    sprintf(histo,"Errx_LF_rphi_layer%dtib",i+1);
    meErrxLFRphiTIB[i] = dbe_->book1D(histo,"RecHit err(x) Local Frame coord.",100,0,0.005);  
    sprintf(histo,"Errx_MF_rphi_layer%dtib",i+1);
    meErrxMFRphiTIB[i] = dbe_->book1D(histo,"RecHit err(x) Meas. Frame coord.",100,0,0.5);  

    sprintf(histo,"Res_LF_rphi_layer%dtib",i+1);
    meResLFRphiTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Res_MF_rphi_layer%dtib",i+1);
    meResMFRphiTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-2,+2);  

    sprintf(histo,"Pull_LF_rphi_layer%dtib",i+1);
    mePullLFRphiTIB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_rphi_layer%dtib",i+1);
    mePullMFRphiTIB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  

    sprintf(histo,"Trackangle_rphi_layer%dtib",i+1);
    meTrackangleRphiTIB[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  

    sprintf(histo,"Trackanglebeta_rphi_layer%dtib",i+1);
    meTrackanglebetaRphiTIB[i] = dbe_->book1D(histo,"Track angle beta",100,-20.,20.);  

    sprintf(histo,"Trackangle2_rphi_layer%dtib",i+1);
    meTrackangle2RphiTIB[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  

    sprintf(histo,"PullTrackangleProfile_rphi_layer%dtib",i+1);
    mePullTrackangleProfileRphiTIB[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -20., 20.,100, -2.,2.,"s");

    sprintf(histo,"Trackwidth_rphi_layer%dtib",i+1);
    meTrackwidthRphiTIB[i] = dbe_->book1D(histo,"Track width",100,0.,1.);  

    sprintf(histo,"Expectedwidth_rphi_layer%dtib",i+1);
    meExpectedwidthRphiTIB[i] = dbe_->book1D(histo,"Expected width",10,0.,10.);  

    sprintf(histo,"Clusterwidth_rphi_layer%dtib",i+1);
    meClusterwidthRphiTIB[i] = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

    sprintf(histo,"Category_rphi_layer%dtib",i+1);
    meCategoryRphiTIB[i] = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

    sprintf(histo,"PullTrackwidthProfile_rphi_layer%dtib",i+1);
    mePullTrackwidthProfileRphiTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category1_rphi_layer%dtib",i+1);
    mePullTrackwidthProfileCategory1RphiTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category1", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category2_rphi_layer%dtib",i+1);
    mePullTrackwidthProfileCategory2RphiTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category2", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category3_rphi_layer%dtib",i+1);
    mePullTrackwidthProfileCategory3RphiTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category3", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category4_rphi_layer%dtib",i+1);
    mePullTrackwidthProfileCategory4RphiTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category4", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"ErrxMFTrackwidthProfile_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileRphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile", 12, 0., 4.,100, -2.,2.,"s");


    sprintf(histo,"ErrxMFTrackwidthProfile_WClus1_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileWclus1RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_WClus2_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileWclus2RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_WClus3_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileWclus3RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_WClus4_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileWclus4RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ResMFTrackwidthProfile_WClus1_rphi_layer%dtib",i+1);
    meResMFTrackwidthProfileWclus1RphiTIB[i] = dbe_->bookProfile(histo,"Residue Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");
    sprintf(histo,"ResMFTrackwidthProfile_WClus2_rphi_layer%dtib",i+1);
    meResMFTrackwidthProfileWclus2RphiTIB[i] = dbe_->bookProfile(histo,"Residue Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ResMFTrackwidthProfile_WClus3_rphi_layer%dtib",i+1);
    meResMFTrackwidthProfileWclus3RphiTIB[i] = dbe_->bookProfile(histo,"Residue Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ResMFTrackwidthProfile_WClus4_rphi_layer%dtib",i+1);
    meResMFTrackwidthProfileWclus4RphiTIB[i] = dbe_->bookProfile(histo,"Residue Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");



    sprintf(histo,"ErrxMFTrackwidthProfile_Category1_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileCategory1RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category2_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileCategory2RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category3_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileCategory3RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category4_rphi_layer%dtib",i+1);
    meErrxMFTrackwidthProfileCategory4RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFAngleProfile_rphi_layer%dtib",i+1);
    meErrxMFAngleProfileRphiTIB[i] = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFClusterwidthProfile_Category1_rphi_layer%dtib",i+1);
    meErrxMFClusterwidthProfileCategory1RphiTIB[i] = dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");
    
  }

  //one histo per Layer stereo and matched hits
  for(int i = 0 ;i<2 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_sas_layer%dtib",i+1);
    meNstpSasTIB[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_sas_layer%dtib",i+1);
    meAdcSasTIB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_sas_layer%dtib",i+1);
    mePosxSasTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  

    sprintf(histo,"Errx_LF_sas_layer%dtib",i+1);
    meErrxLFSasTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.005);  
    sprintf(histo,"Errx_MF_sas_layer%dtib",i+1);
    meErrxMFSasTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.5);  

    sprintf(histo,"Res_LF_sas_layer%dtib",i+1);
    meResLFSasTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Res_MF_sas_layer%dtib",i+1);
    meResMFSasTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-2,+2);  

    sprintf(histo,"Pull_LF_sas_layer%dtib",i+1);
    mePullLFSasTIB[i] = dbe_->book1D(histo,"Pull",100,-4.,4.);  
    sprintf(histo,"Pull_MF_sas_layer%dtib",i+1);
    mePullMFSasTIB[i] = dbe_->book1D(histo,"Pull",100,-4.,4.);  

    sprintf(histo,"Trackangle_sas_layer%dtib",i+1);
    meTrackangleSasTIB[i] = dbe_->book1D(histo,"Track angle",100,-40.,40.);  

    sprintf(histo,"Trackanglebeta_sas_layer%dtib",i+1);
    meTrackanglebetaSasTIB[i] = dbe_->book1D(histo,"Track angle beta",100,-40.,40.);  

    sprintf(histo,"PullTrackangleProfile_sas_layer%dtib",i+1);
    mePullTrackangleProfileSasTIB[i] = dbe_->bookProfile(histo,"Pull Track angle Profile",  100, -40., 40.,100,-4.,4.,"s");

    sprintf(histo,"Trackwidth_sas_layer%dtib",i+1);
    meTrackwidthSasTIB[i] = dbe_->book1D(histo,"Track width",100,0.,1.);  

    sprintf(histo,"Expectedwidth_sas_layer%dtib",i+1);
    meExpectedwidthSasTIB[i] = dbe_->book1D(histo,"Expected width",10,0.,10.);  

    sprintf(histo,"Clusterwidth_sas_layer%dtib",i+1);
    meClusterwidthSasTIB[i] = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

    sprintf(histo,"Category_sas_layer%dtib",i+1);
    meCategorySasTIB[i] = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

    sprintf(histo,"PullTrackwidthProfile_sas_layer%dtib",i+1);
    mePullTrackwidthProfileSasTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category1_sas_layer%dtib",i+1);
    mePullTrackwidthProfileCategory1SasTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category1", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category2_sas_layer%dtib",i+1);
    mePullTrackwidthProfileCategory2SasTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category2", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category3_sas_layer%dtib",i+1);
    mePullTrackwidthProfileCategory3SasTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category3", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category4_sas_layer%dtib",i+1);
    mePullTrackwidthProfileCategory4SasTIB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category4", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_sas_layer%dtib",i+1);
    meErrxMFTrackwidthProfileSasTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category1_sas_layer%dtib",i+1);
    meErrxMFTrackwidthProfileCategory1SasTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category2_sas_layer%dtib",i+1);
    meErrxMFTrackwidthProfileCategory2SasTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category3_sas_layer%dtib",i+1);
    meErrxMFTrackwidthProfileCategory3SasTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category4_sas_layer%dtib",i+1);
    meErrxMFTrackwidthProfileCategory4SasTIB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFAngleProfile_sas_layer%dtib",i+1);
    meErrxMFAngleProfileSasTIB[i] = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFClusterwidthProfile_Category1_sas_layer%dtib",i+1);
    meErrxMFClusterwidthProfileCategory1SasTIB[i] = dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");
    


    sprintf(histo,"Posx_matched_layer%dtib",i+1);
    mePosxMatchedTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%dtib",i+1);
    mePosyMatchedTIB[i] = dbe_->book1D(histo,"RecHit y coord.",100,-6.0, +6.0);  
    sprintf(histo,"Errx_matched_layer%dtib",i+1);
    meErrxMatchedTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.05);  
    sprintf(histo,"Erry_matched_layer%dtib",i+1);
    meErryMatchedTIB[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.05);  
    sprintf(histo,"Resx_matched_layer%dtib",i+1);
    meResxMatchedTIB[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.02, +0.02);  
    sprintf(histo,"Resy_matched_layer%dtib",i+1);
    meResyMatchedTIB[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
    sprintf(histo,"Pullx_matched_layer%dtib",i+1);
    mePullxMatchedTIB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pully_matched_layer%dtib",i+1);
    mePullyMatchedTIB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
  }

  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Strip/TOB");
  //one histo per Layer rphi hits
  for(int i = 0 ;i<6 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtob",i+1);
    meNstpRphiTOB[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%dtob",i+1);
    meAdcRphiTOB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtob",i+1);
    mePosxRphiTOB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  

    sprintf(histo,"Errx_LF_rphi_layer%dtob",i+1);
    meErrxLFRphiTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.01);  
    sprintf(histo,"Errx_MF_rphi_layer%dtob",i+1);
    meErrxMFRphiTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.5);  

    sprintf(histo,"Res_LF_rphi_layer%dtob",i+1);
    meResLFRphiTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Res_MF_rphi_layer%dtob",i+1);
    meResMFRphiTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-2,2);  

    sprintf(histo,"Pull_LF_rphi_layer%dtob",i+1);
    mePullLFRphiTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_rphi_layer%dtob",i+1);
    mePullMFRphiTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  

    sprintf(histo,"Trackangle_rphi_layer%dtob",i+1);
    meTrackangleRphiTOB[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  

    sprintf(histo,"Trackanglebeta_rphi_layer%dtob",i+1);
    meTrackanglebetaRphiTOB[i] = dbe_->book1D(histo,"Track angle beta",100,-20.,20.);  

    sprintf(histo,"PullTrackangleProfile_rphi_layer%dtob",i+1);
    mePullTrackangleProfileRphiTOB[i] = dbe_->bookProfile(histo,"Pull Track angle Profile",  100, -20., 20.,100,-5.,5.,"s");

    sprintf(histo,"Trackwidth_rphi_layer%dtob",i+1);
    meTrackwidthRphiTOB[i] = dbe_->book1D(histo,"Track width",100,0.,4.);  

    sprintf(histo,"Expectedwidth_rphi_layer%dtob",i+1);
    meExpectedwidthRphiTOB[i] = dbe_->book1D(histo,"Expected width",10,0.,10.);  

    sprintf(histo,"Clusterwidth_rphi_layer%dtob",i+1);
    meClusterwidthRphiTOB[i] = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

    sprintf(histo,"Category_rphi_layer%dtob",i+1);
    meCategoryRphiTOB[i] = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

    sprintf(histo,"PullTrackwidthProfile_rphi_layer%dtob",i+1);
    mePullTrackwidthProfileRphiTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category1_rphi_layer%dtob",i+1);
    mePullTrackwidthProfileCategory1RphiTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category1", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category2_rphi_layer%dtob",i+1);
    mePullTrackwidthProfileCategory2RphiTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category2", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category3_rphi_layer%dtob",i+1);
    mePullTrackwidthProfileCategory3RphiTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category3", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category4_rphi_layer%dtob",i+1);
    mePullTrackwidthProfileCategory4RphiTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category4", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileRphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_WClus1_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileWclus1RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_WClus2_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileWclus2RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_WClus3_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileWclus3RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_WClus4_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileWclus4RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ResMFTrackwidthProfile_WClus1_rphi_layer%dtob",i+1);
    meResMFTrackwidthProfileWclus1RphiTOB[i] = dbe_->bookProfile(histo,"Residue Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");
    sprintf(histo,"ResMFTrackwidthProfile_WClus2_rphi_layer%dtob",i+1);
    meResMFTrackwidthProfileWclus2RphiTOB[i] = dbe_->bookProfile(histo,"Residue Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ResMFTrackwidthProfile_WClus3_rphi_layer%dtob",i+1);
    meResMFTrackwidthProfileWclus3RphiTOB[i] = dbe_->bookProfile(histo,"Residue Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");

    sprintf(histo,"ResMFTrackwidthProfile_WClus4_rphi_layer%dtob",i+1);
    meResMFTrackwidthProfileWclus4RphiTOB[i] = dbe_->bookProfile(histo,"Residue Track width Profile Wclus1", 12, 0., 4.,100, -2.,2.,"s");



    sprintf(histo,"ErrxMFTrackwidthProfile_Category1_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileCategory1RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category2_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileCategory2RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category3_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileCategory3RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category4_rphi_layer%dtob",i+1);
    meErrxMFTrackwidthProfileCategory4RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFAngleProfile_rphi_layer%dtob",i+1);
    meErrxMFAngleProfileRphiTOB[i] = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFClusterwidthProfile_Category1_rphi_layer%dtob",i+1);
    meErrxMFClusterwidthProfileCategory1RphiTOB[i] = dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");
    
  }

  //one histo per Layer stereo and matched hits
  for(int i = 0 ;i<2 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_sas_layer%dtob",i+1);
    meNstpSasTOB[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_sas_layer%dtob",i+1);
    meAdcSasTOB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_sas_layer%dtob",i+1);
    mePosxSasTOB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  

    sprintf(histo,"Errx_LF_sas_layer%dtob",i+1);
    meErrxLFSasTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.01);  
    sprintf(histo,"Errx_MF_sas_layer%dtob",i+1);
    meErrxMFSasTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.5);  

    sprintf(histo,"Res_LF_sas_layer%dtob",i+1);
    meResLFSasTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Res_MF_sas_layer%dtob",i+1);
    meResMFSasTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-2,2);  

    sprintf(histo,"Pull_LF_sas_layer%dtob",i+1);
    mePullLFSasTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_sas_layer%dtob",i+1);
    mePullMFSasTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  

    sprintf(histo,"Trackangle_sas_layer%dtob",i+1);
    meTrackangleSasTOB[i] = dbe_->book1D(histo,"Track angle",100,-25.,25.);  

    sprintf(histo,"Trackanglebeta_sas_layer%dtob",i+1);
    meTrackanglebetaSasTOB[i] = dbe_->book1D(histo,"Track angle beta",100,-25.,25.);  

    sprintf(histo,"PullTrackangleProfile_sas_layer%dtob",i+1);
    mePullTrackangleProfileSasTOB[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -25., 25. ,100 , -5., 5.,"s");

    sprintf(histo,"Trackwidth_sas_layer%dtob",i+1);
    meTrackwidthSasTOB[i] = dbe_->book1D(histo,"Track width",100,0.,1.);  

    sprintf(histo,"Expectedwidth_sas_layer%dtob",i+1);
    meExpectedwidthSasTOB[i] = dbe_->book1D(histo,"Expected width",10,0.,10.);  

    sprintf(histo,"Clusterwidth_sas_layer%dtob",i+1);
    meClusterwidthSasTOB[i] = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

    sprintf(histo,"Category_sas_layer%dtob",i+1);
    meCategorySasTOB[i] = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

    sprintf(histo,"PullTrackwidthProfile_sas_layer%dtob",i+1);
    mePullTrackwidthProfileSasTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category1_sas_layer%dtob",i+1);
    mePullTrackwidthProfileCategory1SasTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category1", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category2_sas_layer%dtob",i+1);
    mePullTrackwidthProfileCategory2SasTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category2", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category3_sas_layer%dtob",i+1);
    mePullTrackwidthProfileCategory3SasTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category3", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category4_sas_layer%dtob",i+1);
    mePullTrackwidthProfileCategory4SasTOB[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category4", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_sas_layer%dtob",i+1);
    meErrxMFTrackwidthProfileSasTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category1_sas_layer%dtob",i+1);
    meErrxMFTrackwidthProfileCategory1SasTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category2_sas_layer%dtob",i+1);
    meErrxMFTrackwidthProfileCategory2SasTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category3_sas_layer%dtob",i+1);
    meErrxMFTrackwidthProfileCategory3SasTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category4_sas_layer%dtob",i+1);
    meErrxMFTrackwidthProfileCategory4SasTOB[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFAngleProfile_sas_layer%dtob",i+1);
    meErrxMFAngleProfileSasTOB[i] = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 1.,100, -2.,2.,"s");

     sprintf(histo,"ErrxMFClusterwidthProfile_Category1_sas_layer%dtob",i+1);
    meErrxMFClusterwidthProfileCategory1SasTOB[i] = dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");
    
   sprintf(histo,"Posx_matched_layer%dtob",i+1);
    mePosxMatchedTOB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%dtob",i+1);
    mePosyMatchedTOB[i] = dbe_->book1D(histo,"RecHit y coord.",100,-6.0, +6.0);  
    sprintf(histo,"Errx_matched_layer%dtob",i+1);
    meErrxMatchedTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.05);  
    sprintf(histo,"Erry_matched_layer%dtob",i+1);
    meErryMatchedTOB[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.05);  
    sprintf(histo,"Resx_matched_layer%dtob",i+1);
    meResxMatchedTOB[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.02, +0.02);  
    sprintf(histo,"Resy_matched_layer%dtob",i+1);
    meResyMatchedTOB[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
    sprintf(histo,"Pullx_matched_layer%dtob",i+1);
    mePullxMatchedTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pully_matched_layer%dtob",i+1);
    mePullyMatchedTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
  }

  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Strip/TID");
  //one histo per Ring rphi hits: 3 rings, 6 disks, 2 inner rings are glued 
  for(int i = 0 ;i<3 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtid",i+1);
    meNstpRphiTID[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%dtid",i+1);
    meAdcRphiTID[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtid",i+1);
    mePosxRphiTID[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_LF_rphi_layer%dtid",i+1);
    meErrxLFRphiTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.5);  
    sprintf(histo,"Errx_MF_rphi_layer%dtid",i+1);
    meErrxMFRphiTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.5);  
    sprintf(histo,"Res_LF_rphi_layer%dtid",i+1);
    meResLFRphiTID[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Res_MF_rphi_layer%dtid",i+1);
    meResMFRphiTID[i] = dbe_->book1D(histo,"RecHit Residual",100,-2,2);  
    sprintf(histo,"Pull_LF_rphi_layer%dtid",i+1);
    mePullLFRphiTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_rphi_layer%dtid",i+1);
    mePullMFRphiTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Trackangle_rphi_layer%dtid",i+1);
    meTrackangleRphiTID[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  

    sprintf(histo,"Trackanglebeta_rphi_layer%dtid",i+1);
    meTrackanglebetaRphiTID[i] = dbe_->book1D(histo,"Track angle beta",100,-20.,20.);  

    sprintf(histo,"PullTrackangleProfile_rphi_layer%dtid",i+1);
    mePullTrackangleProfileRphiTID[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -20., 20.,100, -5., 5.,"s");

    sprintf(histo,"Trackwidth_rphi_layer%dtid",i+1);
    meTrackwidthRphiTID[i] = dbe_->book1D(histo,"Track width",100,0.,1.);  

    sprintf(histo,"Expectedwidth_rphi_layer%dtid",i+1);
    meExpectedwidthRphiTID[i] = dbe_->book1D(histo,"Expected width",10,0.,10.);  

    sprintf(histo,"Clusterwidth_rphi_layer%dtid",i+1);
    meClusterwidthRphiTID[i] = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

    sprintf(histo,"Category_rphi_layer%dtid",i+1);
    meCategoryRphiTID[i] = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

    sprintf(histo,"PullTrackwidthProfile_rphi_layer%dtid",i+1);
    mePullTrackwidthProfileRphiTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category1_rphi_layer%dtid",i+1);
    mePullTrackwidthProfileCategory1RphiTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category1", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category2_rphi_layer%dtid",i+1);
    mePullTrackwidthProfileCategory2RphiTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category2", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category3_rphi_layer%dtid",i+1);
    mePullTrackwidthProfileCategory3RphiTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category3", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category4_rphi_layer%dtid",i+1);
    mePullTrackwidthProfileCategory4RphiTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category4", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_rphi_layer%dtid",i+1);
    meErrxMFTrackwidthProfileRphiTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category1_rphi_layer%dtid",i+1);
    meErrxMFTrackwidthProfileCategory1RphiTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category2_rphi_layer%dtid",i+1);
    meErrxMFTrackwidthProfileCategory2RphiTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category3_rphi_layer%dtid",i+1);
    meErrxMFTrackwidthProfileCategory3RphiTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category4_rphi_layer%dtid",i+1);
    meErrxMFTrackwidthProfileCategory4RphiTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFAngleProfile_rphi_layer%dtid",i+1);
    meErrxMFAngleProfileRphiTID[i] = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFClusterwidthProfile_Category1_rphi_layer%dtid",i+1);
    meErrxMFClusterwidthProfileCategory1RphiTID[i] = dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");
    
  }

  //one histo per Ring stereo and matched hits
  for(int i = 0 ;i<2 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_sas_layer%dtid",i+1);
    meNstpSasTID[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_sas_layer%dtid",i+1);
    meAdcSasTID[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_sas_layer%dtid",i+1);
    mePosxSasTID[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_LF_sas_layer%dtid",i+1);
    meErrxLFSasTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.5);  
    sprintf(histo,"Errx_MF_sas_layer%dtid",i+1);
    meErrxMFSasTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.5);  
    sprintf(histo,"Res_LF_sas_layer%dtid",i+1);
    meResLFSasTID[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Res_MF_sas_layer%dtid",i+1);
    meResMFSasTID[i] = dbe_->book1D(histo,"RecHit Residual",100,-2,2);  
    sprintf(histo,"Pull_LF_sas_layer%dtid",i+1);
    mePullLFSasTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_sas_layer%dtid",i+1);
    mePullMFSasTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Trackangle_sas_layer%dtid",i+1);
    meTrackangleSasTID[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  
   sprintf(histo,"Trackanglebeta_sas_layer%dtid",i+1);
    meTrackanglebetaSasTID[i] = dbe_->book1D(histo,"Track angle beta",100,-20.,20.);  

    sprintf(histo,"PullTrackangleProfile_sas_layer%dtid",i+1);
    mePullTrackangleProfileSasTID[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -20., 20.,100, -5., 5.,"s");

    sprintf(histo,"Trackwidth_sas_layer%dtid",i+1);
    meTrackwidthSasTID[i] = dbe_->book1D(histo,"Track width",100,0.,1.);  

    sprintf(histo,"Expectedwidth_sas_layer%dtid",i+1);
    meExpectedwidthSasTID[i] = dbe_->book1D(histo,"Expected width",10,0.,10.);  

    sprintf(histo,"Clusterwidth_sas_layer%dtid",i+1);
    meClusterwidthSasTID[i] = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

    sprintf(histo,"Category_sas_layer%dtid",i+1);
    meCategorySasTID[i] = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

    sprintf(histo,"PullTrackwidthProfile_sas_layer%dtid",i+1);
    mePullTrackwidthProfileSasTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category1_sas_layer%dtid",i+1);
    mePullTrackwidthProfileCategory1SasTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category1", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category2_sas_layer%dtid",i+1);
    mePullTrackwidthProfileCategory2SasTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category2", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category3_sas_layer%dtid",i+1);
    mePullTrackwidthProfileCategory3SasTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category3", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category4_sas_layer%dtid",i+1);
    mePullTrackwidthProfileCategory4SasTID[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category4", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_sas_layer%dtid",i+1);
    meErrxMFTrackwidthProfileSasTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category1_sas_layer%dtid",i+1);
    meErrxMFTrackwidthProfileCategory1SasTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category2_sas_layer%dtid",i+1);
    meErrxMFTrackwidthProfileCategory2SasTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category3_sas_layer%dtid",i+1);
    meErrxMFTrackwidthProfileCategory3SasTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category4_sas_layer%dtid",i+1);
    meErrxMFTrackwidthProfileCategory4SasTID[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFAngleProfile_sas_layer%dtid",i+1);
    meErrxMFAngleProfileSasTID[i] = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 1.,100, -2.,2.,"s");

     sprintf(histo,"ErrxMFClusterwidthProfile_Category1_sas_layer%dtid",i+1);
    meErrxMFClusterwidthProfileCategory1SasTID[i] = dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");
    
   sprintf(histo,"Posx_matched_layer%dtid",i+1);
    mePosxMatchedTID[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%dtid",i+1);
    mePosyMatchedTID[i] = dbe_->book1D(histo,"RecHit y coord.",100,-6.0, +6.0);  
    sprintf(histo,"Errx_matched_layer%dtid",i+1);
    meErrxMatchedTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.02);  
    sprintf(histo,"Erry_matched_layer%dtid",i+1);
    meErryMatchedTID[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.1);  
    sprintf(histo,"Resx_matched_layer%dtid",i+1);
    meResxMatchedTID[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.2, +0.2);  
    sprintf(histo,"Resy_matched_layer%dtid",i+1);
    meResyMatchedTID[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
    sprintf(histo,"Pullx_matched_layer%dtid",i+1);
    mePullxMatchedTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pully_matched_layer%dtid",i+1);
    mePullyMatchedTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
  }

  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Strip/TEC");
  //one histo per Ring rphi hits: 7 rings, 18 disks. Innermost 3 rings are same as TID above.  
  for(int i = 0 ;i<7 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtec",i+1);
    meNstpRphiTEC[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%dtec",i+1);
    meAdcRphiTEC[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtec",i+1);
    mePosxRphiTEC[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  

    sprintf(histo,"Errx_LF_rphi_layer%dtec",i+1);
    meErrxLFRphiTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.5);  
    sprintf(histo,"Errx_MF_rphi_layer%dtec",i+1);
    meErrxMFRphiTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.5);  

    sprintf(histo,"Res_LF_rphi_layer%dtec",i+1);
    meResLFRphiTEC[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Res_MF_rphi_layer%dtec",i+1);
    meResMFRphiTEC[i] = dbe_->book1D(histo,"RecHit Residual",100,-2,2);  

    sprintf(histo,"Pull_LF_rphi_layer%dtec",i+1);
    mePullLFRphiTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_rphi_layer%dtec",i+1);
    mePullMFRphiTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  

    sprintf(histo,"Trackangle_rphi_layer%dtec",i+1);
    meTrackangleRphiTEC[i] = dbe_->book1D(histo,"Track angle",100,-10.,10.);  

    sprintf(histo,"Trackanglebeta_rphi_layer%dtec",i+1);
    meTrackanglebetaRphiTEC[i] = dbe_->book1D(histo,"Track angle beta",100,-10.,10.);  

    sprintf(histo,"PullTrackangleProfile_rphi_layer%dtec",i+1);
    mePullTrackangleProfileRphiTEC[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -10., 10.,100, -5., 5.,"s");

    sprintf(histo,"Trackwidth_rphi_layer%dtec",i+1);
    meTrackwidthRphiTEC[i] = dbe_->book1D(histo,"Track width",100,0.,1.);  

    sprintf(histo,"Expectedwidth_rphi_layer%dtec",i+1);
    meExpectedwidthRphiTEC[i] = dbe_->book1D(histo,"Expected width",10,0.,10.);  

    sprintf(histo,"Clusterwidth_rphi_layer%dtec",i+1);
    meClusterwidthRphiTEC[i] = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

    sprintf(histo,"Category_rphi_layer%dtec",i+1);
    meCategoryRphiTEC[i] = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

    sprintf(histo,"PullTrackwidthProfile_rphi_layer%dtec",i+1);
    mePullTrackwidthProfileRphiTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category1_rphi_layer%dtec",i+1);
    mePullTrackwidthProfileCategory1RphiTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category1", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"PullTrackwidthProfile_Category2_rphi_layer%dtec",i+1);
    mePullTrackwidthProfileCategory2RphiTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category2", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category3_rphi_layer%dtec",i+1);
    mePullTrackwidthProfileCategory3RphiTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category3", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"PullTrackwidthProfile_Category4_rphi_layer%dtec",i+1);
    mePullTrackwidthProfileCategory4RphiTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category4", 100, 0., 1.,100, -2.,2.,"s");
    
    sprintf(histo,"ErrxMFTrackwidthProfile_rphi_layer%dtec",i+1);
    meErrxMFTrackwidthProfileRphiTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category1_rphi_layer%dtec",i+1);
    meErrxMFTrackwidthProfileCategory1RphiTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category2_rphi_layer%dtec",i+1);
    meErrxMFTrackwidthProfileCategory2RphiTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category3_rphi_layer%dtec",i+1);
    meErrxMFTrackwidthProfileCategory3RphiTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category4_rphi_layer%dtec",i+1);
    meErrxMFTrackwidthProfileCategory4RphiTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFAngleProfile_rphi_layer%dtec",i+1);
    meErrxMFAngleProfileRphiTEC[i] = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFClusterwidthProfile_Category1_rphi_layer%dtec",i+1);
    meErrxMFClusterwidthProfileCategory1RphiTEC[i] = dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");
    
  }

  //one histo per Layer stereo and matched hits: rings 1,2,5 are double sided
  for(int i = 0 ;i<5 ; i++) {
    if(i == 0 || i == 1 || i == 4) {
      Char_t histo[200];
      sprintf(histo,"Nstp_sas_layer%dtec",i+1);
      meNstpSasTEC[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
      sprintf(histo,"Adc_sas_layer%dtec",i+1);
      meAdcSasTEC[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
      sprintf(histo,"Posx_sas_layer%dtec",i+1);
      mePosxSasTEC[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
      sprintf(histo,"Errx_LF_sas_layer%dtec",i+1);
      meErrxLFSasTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.5);  
      sprintf(histo,"Errx_MF_sas_layer%dtec",i+1);
      meErrxMFSasTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.5);  
      sprintf(histo,"Res_LF_sas_layer%dtec",i+1);
      meResLFSasTEC[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
      sprintf(histo,"Res_MF_sas_layer%dtec",i+1);
      meResMFSasTEC[i] = dbe_->book1D(histo,"RecHit Residual",100,-2,+2);  
      sprintf(histo,"Pull_LF_sas_layer%dtec",i+1);
      mePullLFSasTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);
      sprintf(histo,"Pull_MF_sas_layer%dtec",i+1);
      mePullMFSasTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);
      sprintf(histo,"Trackangle_sas_layer%dtec",i+1);
      meTrackangleSasTEC[i] = dbe_->book1D(histo,"Track angle",100,-10.,10.);
      sprintf(histo,"Trackanglebeta_sas_layer%dtec",i+1);
      meTrackanglebetaSasTEC[i] = dbe_->book1D(histo,"Track angle beta",100,-10.,10.);

      sprintf(histo,"PullTrackangleProfile_sas_layer%dtec",i+1);
      mePullTrackangleProfileSasTEC[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -10., 10.,100, -5., 5.,"s");
     
      sprintf(histo,"Trackwidth_sas_layer%dtec",i+1);
      meTrackwidthSasTEC[i] = dbe_->book1D(histo,"Track width",100,0.,1.);  

      sprintf(histo,"Expectedwidth_sas_layer%dtec",i+1);
      meExpectedwidthSasTEC[i] = dbe_->book1D(histo,"Expected width",10,0.,10.);  

      sprintf(histo,"Clusterwidth_sas_layer%dtec",i+1);
      meClusterwidthSasTEC[i] = dbe_->book1D(histo,"Cluster width",10,0.,10.);  

      sprintf(histo,"Category_sas_layer%dtec",i+1);
      meCategorySasTEC[i] = dbe_->book1D(histo,"Cluster Category",10,0.,10.);  

      sprintf(histo,"PullTrackwidthProfile_sas_layer%dtec",i+1);
      mePullTrackwidthProfileSasTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile", 100, 0., 1.,100, -2.,2.,"s");

      sprintf(histo,"PullTrackwidthProfile_Category1_sas_layer%dtec",i+1);
      mePullTrackwidthProfileCategory1SasTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category1", 100, 0., 1.,100, -2.,2.,"s");

      sprintf(histo,"PullTrackwidthProfile_Category2_sas_layer%dtec",i+1);
      mePullTrackwidthProfileCategory2SasTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category2", 100, 0., 1.,100, -2.,2.,"s");
    
      sprintf(histo,"PullTrackwidthProfile_Category3_sas_layer%dtec",i+1);
      mePullTrackwidthProfileCategory3SasTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category3", 100, 0., 1.,100, -2.,2.,"s");
    
      sprintf(histo,"PullTrackwidthProfile_Category4_sas_layer%dtec",i+1);
      mePullTrackwidthProfileCategory4SasTEC[i] = dbe_->bookProfile(histo,"Pull Track width Profile Category4", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_sas_layer%dtec",i+1);
    meErrxMFTrackwidthProfileSasTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFTrackwidthProfile_Category1_sas_layer%dtec",i+1);
    meErrxMFTrackwidthProfileCategory1SasTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category1", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category2_sas_layer%dtec",i+1);
    meErrxMFTrackwidthProfileCategory2SasTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category2", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category3_sas_layer%dtec",i+1);
    meErrxMFTrackwidthProfileCategory3SasTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category3", 12,0.,4.,100, -2.,2.,"s");
    sprintf(histo,"ErrxMFTrackwidthProfile_Category4_sas_layer%dtec",i+1);
    meErrxMFTrackwidthProfileCategory4SasTEC[i] = dbe_->bookProfile(histo,"Resolution Track width Profile Category4", 12,0.,4.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFAngleProfile_sas_layer%dtec",i+1);
    meErrxMFAngleProfileSasTEC[i] = dbe_->bookProfile(histo,"Resolution Angle Profile", 100, 0., 1.,100, -2.,2.,"s");

    sprintf(histo,"ErrxMFClusterwidthProfile_Category1_sas_layer%dtec",i+1);
    meErrxMFClusterwidthProfileCategory1SasTEC[i] = dbe_->bookProfile(histo,"Resolution Cluster width Profile Category1", 100, 0., 10.,100, -2.,2.,"s");
    
    sprintf(histo,"Posx_matched_layer%dtec",i+1);
    mePosxMatchedTEC[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%dtec",i+1);
    mePosyMatchedTEC[i] = dbe_->book1D(histo,"RecHit y coord.",100,-8.0, +8.0);  
    sprintf(histo,"Errx_matched_layer%dtec",i+1);
    meErrxMatchedTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.02);  
    sprintf(histo,"Erry_matched_layer%dtec",i+1);
    meErryMatchedTEC[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.1);  
    sprintf(histo,"Resx_matched_layer%dtec",i+1);
    meResxMatchedTEC[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.2, +0.2);  
    sprintf(histo,"Resy_matched_layer%dtec",i+1);
    meResyMatchedTEC[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
    sprintf(histo,"Pullx_matched_layer%dtec",i+1);
    mePullxMatchedTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pully_matched_layer%dtec",i+1);
    mePullyMatchedTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    }
  }


}

void SiStripTrackingRecHitsValid::endJob() {

  /*  
  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Strip/ALL");
  
  PullvsTrackwidth->FitSlicesY();
  ErrxMFvsTrackwidth->FitSlicesY();
  PullvsExpectedwidth->FitSlicesY();
  PullvsClusterwidth->FitSlicesY();
  PullvsTrackangle->FitSlicesY();
  PullvsTrackanglebeta->FitSlicesY();

  PullvsTrackwidthTIB->FitSlicesY();
  PullvsExpectedwidthTIB->FitSlicesY();
  PullvsClusterwidthTIB->FitSlicesY();
  PullvsTrackangleTIB->FitSlicesY();
  PullvsTrackanglebetaTIB->FitSlicesY();

  PullvsTrackwidthTOB->FitSlicesY();
  PullvsExpectedwidthTOB->FitSlicesY();
  PullvsClusterwidthTOB->FitSlicesY();
  PullvsTrackangleTOB->FitSlicesY();
  PullvsTrackanglebetaTOB->FitSlicesY();

  PullvsTrackwidthTID->FitSlicesY();
  PullvsExpectedwidthTID->FitSlicesY();
  PullvsClusterwidthTID->FitSlicesY();
  PullvsTrackangleTID->FitSlicesY();
  PullvsTrackanglebetaTID->FitSlicesY();

  PullvsTrackwidthTEC->FitSlicesY();
  PullvsExpectedwidthTEC->FitSlicesY();
  PullvsClusterwidthTEC->FitSlicesY();
  PullvsTrackangleTEC->FitSlicesY();
  PullvsTrackanglebetaTEC->FitSlicesY();

  //int aaa = Pullvstrackwidth_1->GetEntries();ErrxMFvsTrackwidth

  TH1D *PullvsTrackwidth_2 = (TH1D*)gDirectory->Get("PullvsTrackwidth_2");
  TH1D *PullvsExpectedwidth_2 = (TH1D*)gDirectory->Get("PullvsExpectedwidth_2");
  TH1D *PullvsClusterwidth_2 = (TH1D*)gDirectory->Get("PullvsClusterwidth_2");
  TH1D *PullvsTrackangle_2 = (TH1D*)gDirectory->Get("PullvsTrackangle_2");
  TH1D *PullvsTrackanglebeta_2 = (TH1D*)gDirectory->Get("PullvsTrackanglebeta_2");

  TH1D *PullvsTrackwidthTIB_2 = (TH1D*)gDirectory->Get("PullvsTrackwidthTIB_2");
  TH1D *PullvsExpectedwidthTIB_2 = (TH1D*)gDirectory->Get("PullvsExpectedwidthTIB_2");
  TH1D *PullvsClusterwidthTIB_2 = (TH1D*)gDirectory->Get("PullvsClusterwidthTIB_2");
  TH1D *PullvsTrackangleTIB_2 = (TH1D*)gDirectory->Get("PullvsTrackangleTIB_2");
  TH1D *PullvsTrackanglebetaTIB_2 = (TH1D*)gDirectory->Get("PullvsTrackanglebetaTIB_2");

  TH1D *PullvsTrackwidthTOB_2 = (TH1D*)gDirectory->Get("PullvsTrackwidthTOB_2");
  TH1D *PullvsExpectedwidthTOB_2 = (TH1D*)gDirectory->Get("PullvsExpectedwidthTOB_2");
  TH1D *PullvsClusterwidthTOB_2 = (TH1D*)gDirectory->Get("PullvsClusterwidthTOB_2");
  TH1D *PullvsTrackangleTOB_2 = (TH1D*)gDirectory->Get("PullvsTrackangleTOB_2");
  TH1D *PullvsTrackanglebetaTOB_2 = (TH1D*)gDirectory->Get("PullvsTrackanglebetaTOB_2");

  TH1D *PullvsTrackwidthTID_2 = (TH1D*)gDirectory->Get("PullvsTrackwidthTID_2");
  TH1D *PullvsExpectedwidthTID_2 = (TH1D*)gDirectory->Get("PullvsExpectedwidthTID_2");
  TH1D *PullvsClusterwidthTID_2 = (TH1D*)gDirectory->Get("PullvsClusterwidthTID_2");
  TH1D *PullvsTrackangleTID_2 = (TH1D*)gDirectory->Get("PullvsTrackangleTID_2");
  TH1D *PullvsTrackanglebetaTID_2 = (TH1D*)gDirectory->Get("PullvsTrackanglebetaTID_2");

  TH1D *PullvsTrackwidthTEC_2 = (TH1D*)gDirectory->Get("PullvsTrackwidthTEC_2");
  TH1D *PullvsExpectedwidthTEC_2 = (TH1D*)gDirectory->Get("PullvsExpectedwidthTEC_2");
  TH1D *PullvsClusterwidthTEC_2 = (TH1D*)gDirectory->Get("PullvsClusterwidthTEC_2");
  TH1D *PullvsTrackangleTEC_2 = (TH1D*)gDirectory->Get("PullvsTrackangleTEC_2");
  TH1D *PullvsTrackanglebetaTEC_2 = (TH1D*)gDirectory->Get("PullvsTrackanglebetaTEC_2");

  //cout<<"h2_1->GetEntries() = "<<PullvsTrackwidth_1->GetEntries()<<endl;
  //cout<<"ddbb1"<<endl;
  unsigned int NBINSPullvsTrackwidth =PullvsTrackwidth_2->GetNbinsX();
  unsigned int NBINSPullvsClusterwidth = PullvsClusterwidth_2->GetNbinsX();
  unsigned int NBINSPullvsExpectedwidth = PullvsExpectedwidth_2->GetNbinsX();
  //cout<<"ddbb2"<<endl;
  unsigned int NBINSPullvsTrackangle = PullvsTrackangle_2->GetNbinsX();
  unsigned int NBINSPullvsTrackanglebeta = PullvsTrackanglebeta_2->GetNbinsX();
  //cout<<"ddbb3"<<endl;

  PullRMSvsTrackwidth = dbe_->book1D("PullRMSvsTrackwidth", "PullRMSvsTrackwidth",NBINSPullvsTrackwidth ,0.,4.);
  PullRMSvsClusterwidth = dbe_->book1D("PullRMSvsClusterwidth", "PullRMSvsClusterwidth",NBINSPullvsClusterwidth ,0.5,8.5);
  PullRMSvsExpectedwidth = dbe_->book1D("PullRMSvsExpectedwidth", "PullRMSvsExpectedwidth",NBINSPullvsExpectedwidth ,0.5,4.5);
  PullRMSvsTrackangle = dbe_->book1D("PullRMSvsTrackangle", "PullRMSvsTrackangle",NBINSPullvsTrackangle ,0.,90.);
  PullRMSvsTrackanglebeta = dbe_->book1D("PullRMSvsTrackanglebeta", "PullRMSvsTrackanglebeta",NBINSPullvsTrackanglebeta ,0.,90.);

  PullRMSvsTrackwidthTIB = dbe_->book1D("PullRMSvsTrackwidthTIB", "PullRMSvsTrackwidthTIB",NBINSPullvsTrackwidth ,0.,4.);
  PullRMSvsClusterwidthTIB = dbe_->book1D("PullRMSvsClusterwidthTIB", "PullRMSvsClusterwidthTIB",NBINSPullvsClusterwidth ,0.5,8.5);
  PullRMSvsExpectedwidthTIB = dbe_->book1D("PullRMSvsExpectedwidthTIB", "PullRMSvsExpectedwidthTIB",NBINSPullvsExpectedwidth ,0.5,4.5);
  PullRMSvsTrackangleTIB = dbe_->book1D("PullRMSvsTrackangleTIB", "PullRMSvsTrackangleTIB",NBINSPullvsTrackangle ,0.,90.);
  PullRMSvsTrackanglebetaTIB = dbe_->book1D("PullRMSvsTrackanglebetaTIB", "PullRMSvsTrackanglebetaTIB",NBINSPullvsTrackanglebeta ,0.,90.);

  PullRMSvsTrackwidthTOB = dbe_->book1D("PullRMSvsTrackwidthTOB", "PullRMSvsTrackwidthTOB",NBINSPullvsTrackwidth ,0.,4.);
  PullRMSvsClusterwidthTOB = dbe_->book1D("PullRMSvsClusterwidthTOB", "PullRMSvsClusterwidthTOB",NBINSPullvsClusterwidth ,0.5,8.5);
  PullRMSvsExpectedwidthTOB = dbe_->book1D("PullRMSvsExpectedwidthTOB", "PullRMSvsExpectedwidthTOB",NBINSPullvsExpectedwidth ,0.5,4.5);
  PullRMSvsTrackangleTOB = dbe_->book1D("PullRMSvsTrackangleTOB", "PullRMSvsTrackangleTOB",NBINSPullvsTrackangle ,0.,90.);
  PullRMSvsTrackanglebetaTOB = dbe_->book1D("PullRMSvsTrackanglebetaTOB", "PullRMSvsTrackanglebetaTOB",NBINSPullvsTrackanglebeta ,0.,90.);

  PullRMSvsTrackwidthTID = dbe_->book1D("PullRMSvsTrackwidthTID", "PullRMSvsTrackwidthTID",NBINSPullvsTrackwidth ,0.,4.);
  PullRMSvsClusterwidthTID = dbe_->book1D("PullRMSvsClusterwidthTID", "PullRMSvsClusterwidthTID",NBINSPullvsClusterwidth ,0.5,8.5);
  PullRMSvsExpectedwidthTID = dbe_->book1D("PullRMSvsExpectedwidthTID", "PullRMSvsExpectedwidthTID",NBINSPullvsExpectedwidth ,0.5,4.5);
  PullRMSvsTrackangleTID = dbe_->book1D("PullRMSvsTrackangleTID", "PullRMSvsTrackangleTID",NBINSPullvsTrackangle ,0.,90.);
  PullRMSvsTrackanglebetaTID = dbe_->book1D("PullRMSvsTrackanglebetaTID", "PullRMSvsTrackanglebetaTID",NBINSPullvsTrackanglebeta ,0.,90.);

  PullRMSvsTrackwidthTEC = dbe_->book1D("PullRMSvsTrackwidthTEC", "PullRMSvsTrackwidthTEC",NBINSPullvsTrackwidth ,0.,4.);
  PullRMSvsClusterwidthTEC = dbe_->book1D("PullRMSvsClusterwidthTEC", "PullRMSvsClusterwidthTEC",NBINSPullvsClusterwidth ,0.5,8.5);
  PullRMSvsExpectedwidthTEC = dbe_->book1D("PullRMSvsExpectedwidthTEC", "PullRMSvsExpectedwidthTEC",NBINSPullvsExpectedwidth ,0.5,4.5);
  PullRMSvsTrackangleTEC = dbe_->book1D("PullRMSvsTrackangleTEC", "PullRMSvsTrackangleTEC",NBINSPullvsTrackangle ,0.,90.);
  PullRMSvsTrackanglebetaTEC = dbe_->book1D("PullRMSvsTrackanglebetaTEC", "PullRMSvsTrackanglebetaTEC",NBINSPullvsTrackanglebeta ,0.,90.);

  //cout<<"ddbb5"<<endl;
  for(unsigned int i = 0; i !=NBINSPullvsTrackwidth ; ++i){
    PullRMSvsTrackwidth->setBinContent(i,PullvsTrackwidth_2 ->GetBinContent(i));
    PullRMSvsTrackwidth->setBinError(i,PullvsTrackwidth_2 ->GetBinError(i));
    PullRMSvsTrackwidthTIB->setBinContent(i,PullvsTrackwidthTIB_2 ->GetBinContent(i));
    PullRMSvsTrackwidthTIB->setBinError(i,PullvsTrackwidthTIB_2 ->GetBinError(i));
    PullRMSvsTrackwidthTOB->setBinContent(i,PullvsTrackwidthTOB_2 ->GetBinContent(i));
    PullRMSvsTrackwidthTOB->setBinError(i,PullvsTrackwidthTOB_2 ->GetBinError(i));
    PullRMSvsTrackwidthTID->setBinContent(i,PullvsTrackwidthTID_2 ->GetBinContent(i));
    PullRMSvsTrackwidthTID->setBinError(i,PullvsTrackwidthTID_2 ->GetBinError(i));
    PullRMSvsTrackwidthTEC->setBinContent(i,PullvsTrackwidthTEC_2 ->GetBinContent(i));
    PullRMSvsTrackwidthTEC->setBinError(i,PullvsTrackwidthTEC_2 ->GetBinError(i));
  }
  //cout<<"ddbb6"<<endl;
  for(unsigned int i = 0; i != NBINSPullvsClusterwidth; ++i){
    PullRMSvsClusterwidth->setBinContent(i,PullvsClusterwidth_2 ->GetBinContent(i));
    PullRMSvsClusterwidth->setBinError(i,PullvsClusterwidth_2 ->GetBinError(i));
    PullRMSvsClusterwidthTIB->setBinContent(i,PullvsClusterwidthTIB_2 ->GetBinContent(i));
    PullRMSvsClusterwidthTIB->setBinError(i,PullvsClusterwidthTIB_2 ->GetBinError(i));
    PullRMSvsClusterwidthTOB->setBinContent(i,PullvsClusterwidthTOB_2 ->GetBinContent(i));
    PullRMSvsClusterwidthTOB->setBinError(i,PullvsClusterwidthTOB_2 ->GetBinError(i));
    PullRMSvsClusterwidthTID->setBinContent(i,PullvsClusterwidthTID_2 ->GetBinContent(i));
    PullRMSvsClusterwidthTID->setBinError(i,PullvsClusterwidthTID_2 ->GetBinError(i));
    PullRMSvsClusterwidthTEC->setBinContent(i,PullvsClusterwidthTEC_2 ->GetBinContent(i));
    PullRMSvsClusterwidthTEC->setBinError(i,PullvsClusterwidthTEC_2 ->GetBinError(i));
  }
  //cout<<"ddbb7"<<endl;
  for(unsigned int i = 0; i != NBINSPullvsExpectedwidth; ++i){
    PullRMSvsExpectedwidth->setBinContent(i,PullvsExpectedwidth_2 ->GetBinContent(i));
    PullRMSvsExpectedwidth->setBinError(i,PullvsExpectedwidth_2 ->GetBinError(i));
    PullRMSvsExpectedwidthTIB->setBinContent(i,PullvsExpectedwidthTIB_2 ->GetBinContent(i));
    PullRMSvsExpectedwidthTIB->setBinError(i,PullvsExpectedwidthTIB_2 ->GetBinError(i));
    PullRMSvsExpectedwidthTOB->setBinContent(i,PullvsExpectedwidthTOB_2 ->GetBinContent(i));
    PullRMSvsExpectedwidthTOB->setBinError(i,PullvsExpectedwidthTOB_2 ->GetBinError(i));
    PullRMSvsExpectedwidthTID->setBinContent(i,PullvsExpectedwidthTID_2 ->GetBinContent(i));
    PullRMSvsExpectedwidthTID->setBinError(i,PullvsExpectedwidthTID_2 ->GetBinError(i));
    PullRMSvsExpectedwidthTEC->setBinContent(i,PullvsExpectedwidthTEC_2 ->GetBinContent(i));
    PullRMSvsExpectedwidthTEC->setBinError(i,PullvsExpectedwidthTEC_2 ->GetBinError(i));
  }
  //cout<<"ddbb8"<<endl;
  for(unsigned int i = 0; i != NBINSPullvsTrackangle; ++i){
    PullRMSvsTrackangle->setBinContent(i,PullvsTrackangle_2 ->GetBinContent(i));
    PullRMSvsTrackangle->setBinError(i,PullvsTrackangle_2 ->GetBinError(i));
    PullRMSvsTrackangleTIB->setBinContent(i,PullvsTrackangleTIB_2 ->GetBinContent(i));
    PullRMSvsTrackangleTIB->setBinError(i,PullvsTrackangleTIB_2 ->GetBinError(i));
    PullRMSvsTrackangleTOB->setBinContent(i,PullvsTrackangleTOB_2 ->GetBinContent(i));
    PullRMSvsTrackangleTOB->setBinError(i,PullvsTrackangleTOB_2 ->GetBinError(i));
    PullRMSvsTrackangleTID->setBinContent(i,PullvsTrackangleTID_2 ->GetBinContent(i));
    PullRMSvsTrackangleTID->setBinError(i,PullvsTrackangleTID_2 ->GetBinError(i));
    PullRMSvsTrackangleTEC->setBinContent(i,PullvsTrackangleTEC_2 ->GetBinContent(i));
    PullRMSvsTrackangleTEC->setBinError(i,PullvsTrackangleTEC_2 ->GetBinError(i));
  }
  //cout<<"ddbb9"<<endl;
  for(unsigned int i = 0; i !=NBINSPullvsTrackanglebeta ; ++i){
    PullRMSvsTrackanglebeta->setBinContent(i,PullvsTrackanglebeta_2 ->GetBinContent(i));
    PullRMSvsTrackanglebeta->setBinError(i,PullvsTrackanglebeta_2 ->GetBinError(i));
    PullRMSvsTrackanglebetaTIB->setBinContent(i,PullvsTrackanglebetaTIB_2 ->GetBinContent(i));
    PullRMSvsTrackanglebetaTIB->setBinError(i,PullvsTrackanglebetaTIB_2 ->GetBinError(i));
    PullRMSvsTrackanglebetaTOB->setBinContent(i,PullvsTrackanglebetaTOB_2 ->GetBinContent(i));
    PullRMSvsTrackanglebetaTOB->setBinError(i,PullvsTrackanglebetaTOB_2 ->GetBinError(i));
    PullRMSvsTrackanglebetaTID->setBinContent(i,PullvsTrackanglebetaTID_2 ->GetBinContent(i));
    PullRMSvsTrackanglebetaTID->setBinError(i,PullvsTrackanglebetaTID_2 ->GetBinError(i));
    PullRMSvsTrackanglebetaTEC->setBinContent(i,PullvsTrackanglebetaTEC_2 ->GetBinContent(i));
    PullRMSvsTrackanglebetaTEC->setBinError(i,PullvsTrackanglebetaTEC_2 ->GetBinError(i));
  }  
*/  


  
  /*
  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Strip/ALL");
  unsigned int NBINS = meErrxMFTrackwidthProfile->getNbinsX();
  float Entries = meErrxMFTrackwidthProfile->getEntries();
  cout<<"Entries = "<<Entries<<endl;
  cout<<"NBINS = "<<NBINS<<endl;
  NBINS=100;
  new_1D = dbe_->book1D("my_name", "my_title", NBINS,0.,5.);
  for(unsigned int i = 0; i != NBINS; ++i){
    cout<<"i,   getBinError(i) = "<<i<<";"<<meErrxMFTrackwidthProfile ->getBinError(i)<<endl;
   new_1D->setBinContent(i,meErrxMFTrackwidthProfile ->getBinError(i));
  }
  */

  /*
    myFile->cd();

    PositionSHx->Write();
    Diff->Write();
    SecondStrip->Write();
    ErrxMF->Write();
    ErrxMFvsTrackwidth->Write();
    ResMFvsTrackwidth->Write();
    ResMFvsTrackwidthWClus1->Write();
    ResMFvsTrackwidthWClus1Wexp1->Write();
    ResMFvsTrackwidthWClus1Wexp2->Write();
    ResMFvsTrackwidthWClus1Wexp3->Write();
    ResMFvsTrackwidthWClus1Wexp4->Write();
    ResMFvsTrackwidthWClus2->Write();
    ResMFvsTrackwidthWClus2Wexp1->Write();
    ResMFvsTrackwidthWClus2Wexp2->Write();
    ResMFvsTrackwidthWClus2Wexp3->Write();
    ResMFvsTrackwidthWClus2Wexp4->Write();
    ResMFvsTrackwidthWClus3->Write();
    ResMFvsTrackwidthWClus3Wexp1->Write();
    ResMFvsTrackwidthWClus3Wexp2->Write();
    ResMFvsTrackwidthWClus3Wexp3->Write();
    ResMFvsTrackwidthWClus3Wexp4->Write();
    ResMFvsTrackwidthWClus4->Write();
    ResMFvsTrackwidthWClus4Wexp1->Write();
    ResMFvsTrackwidthWClus4Wexp2->Write();
    ResMFvsTrackwidthWClus4Wexp3->Write();
    ResMFvsTrackwidthWClus4Wexp4->Write();
    ErrxMFvsTrackwidthWClus1->Write();
    ErrxMFvsTrackwidthWClus2->Write();
    ErrxMFvsTrackwidthWClus3->Write();
    ErrxMFvsTrackwidthWClus4->Write();

    ResMFvsTrackwidthCategory2->Write();
    ResMFvsTrackwidthCategory3->Write();
    ResMFvsTrackwidthCategory4->Write();

    ErrxMFvsTrackwidthCategory2->Write();
    ErrxMFvsTrackwidthCategory3->Write();
    ErrxMFvsTrackwidthCategory4->Write();

    //    ErrxMFvsTrackwidth_1->Write();
    //ErrxMFvsTrackwidth_2->Write();

    PullvsTrackwidth->Write();
    //PullvsTrackwidth_1 ->Write();
    //PullvsTrackwidth_2 ->Write();
    PullvsExpectedwidth->Write();
    PullvsExpectedwidth_1 ->Write();
    PullvsExpectedwidth_2 ->Write();
    PullvsClusterwidth->Write();
    PullvsClusterwidth_1 ->Write();
    PullvsClusterwidth_2 ->Write();
    PullvsTrackangle->Write();
    PullvsTrackangle_1 ->Write();
    PullvsTrackangle_2 ->Write();
    PullvsTrackanglebeta->Write();
    PullvsTrackanglebeta_1 ->Write();
    PullvsTrackanglebeta_2 ->Write();
    myFile->Close();
  */


  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

// Virtual destructor needed.
SiStripTrackingRecHitsValid::~SiStripTrackingRecHitsValid() {  

}  

// Functions that gets called by framework every event
void SiStripTrackingRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);


  
  // EventID e.id() ;

  //  float diff=0;
  //float positionshx = 0;
  //int secondstrip = 0;

  int isrechitrphi     = 0;
  int isrechitsas      = 0;
  int isrechitmatched  = 0;

  float anglealpha=0;
  float anglebeta=0;
  float Wtrack;
  int Wexp;
  int clusterWidth;
  DetId detid;
  uint32_t myid;

  LocalPoint position;
  LocalError error;
  MeasurementPoint Mposition;
  MeasurementError Merror;

  int clusiz=0;
  int totcharge=0;


  float mindist = 999999;
  float dist;
  std::vector<PSimHit> matched;
  
  TrackerHitAssociator associate(e, conf_);
  PSimHit closest;

  

  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry &tracker(*pDD);

  const TrackerGeometry * tracker2;
  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker2=&(* estracker);

  edm::ESHandle<MagneticField> magfield;
  //iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );
  es.get<IdealMagneticFieldRecord>().get(magfield );
  //  magfield_  = magfield;
  //const magfield_
  const MagneticField & magfield_ (*magfield);
  magfield2_ = &magfield_;


  edm::ESHandle<StripClusterParameterEstimator> stripcpe;
  es.get<TkStripCPERecord>().get("SimpleStripCPE",stripcpe);

    //


  // Mangano's

  edm::Handle<vector<Trajectory> > trajCollectionHandle;
  e.getByLabel(conf_.getParameter<string>("trajectoryInput"),trajCollectionHandle);

  edm::LogVerbatim("TrajectoryAnalyzer") << "trajColl->size(): " << trajCollectionHandle->size() ;

  //cout<<"trajColl->size() = "<<trajCollectionHandle->size()<<endl;
  
  for(vector<Trajectory>::const_iterator it = trajCollectionHandle->begin(); it!=trajCollectionHandle->end();it++){
     
    edm::LogVerbatim("TrajectoryAnalyzer") << "this traj has " << it->foundHits() << " valid hits"  << " , "
					    << "isValid: " << it->isValid() ;

    vector<TrajectoryMeasurement> tmColl = it->measurements();
    for(vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(); itTraj!=tmColl.end(); itTraj++){
      if(! itTraj->updatedState().isValid()) continue;
           
//        edm::LogVerbatim("TrajectoryAnalyzer") << "tm number: " << (itTraj - tmColl.begin()) + 1<< " , "
// 	  << "tm.backwardState.pt: " << itTraj->backwardPredictedState().globalMomentum().perp() << " , "
// 	  << "tm.forwardState.pt:  " << itTraj->forwardPredictedState().globalMomentum().perp() << " , "
// 	  << "tm.updatedState.pt:  " << itTraj->updatedState().globalMomentum().perp()  << " , "
// 	  << "tm.globalPos.perp: "   << itTraj->updatedState().globalPosition().perp() ;       
       
               if ( itTraj->updatedState().globalMomentum().perp() < 1.) continue;
	       
      TrajectoryStateOnSurface tsos=itTraj->updatedState();
      LocalVector trackdirection=tsos.localDirection();

      DetId  detid2 = itTraj->recHit()->geographicalId();

      const TransientTrackingRecHit::ConstRecHitPointer thit2=itTraj->recHit();
      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit2).hit());
      const SiStripRecHit2D* hit2d=dynamic_cast<const SiStripRecHit2D*>((*thit2).hit());
      const SiStripRecHit1D* hit1d=dynamic_cast<const SiStripRecHit1D*>((*thit2).hit());
      //if(matchedhit) cout<<"manganomatchedhit"<<endl;
      //if(hit) cout<<"manganosimplehit"<<endl;
      //if (hit && matchedhit) cout<<"manganosimpleandmatchedhit"<<endl;
      const TrackingRecHit *thit = (*thit2).hit();
	    
      detid = (thit)->geographicalId();
      myid=((thit)->geographicalId()).rawId();

      StripSubdetector StripSubdet = (StripSubdetector) detid;

      isrechitmatched  = 0;
      
      if(matchedhit){

	isrechitmatched = 1;
 
	position = (thit)->localPosition();
	//  Mposition = topol.measurementPosition(position);
	error = (thit)->localPositionError();
	//  Merror = topol.measurementError(position,error);
	rechitmatchedx = position.x();
	rechitmatchedy = position.y();
	rechitmatchedz = position.z();
	rechitmatchederrxx = error.xx();
	rechitmatchederrxy = error.xy();
	rechitmatchederryy = error.yy();

	//Association of the rechit to the simhit
	mindist = 999999;
	float distx, disty;
	std::pair<LocalPoint,LocalVector> closestPair;
	matched.clear();
	matched = associate.associateHit(*matchedhit);
	if(!matched.empty()){
	  //project simhit;
	  const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(matchedhit->geographicalId());
	  const StripGeomDetUnit* partnerstripdet =(StripGeomDetUnit*) gluedDet->stereoDet();
	  std::pair<LocalPoint,LocalVector> hitPair;
	  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    //project simhit;
	    hitPair= projectHit((*m),partnerstripdet,gluedDet->surface());
	    distx = fabs(rechitmatchedx - hitPair.first.x());
	    disty = fabs(rechitmatchedy - hitPair.first.y());
	    dist = distx*distx+disty*disty;
	    if(sqrt(dist)<mindist){
	      mindist = dist;
	      closestPair = hitPair;
	    }
	  }
	  rechitmatchedresx = rechitmatchedx - closestPair.first.x();
	  rechitmatchedresy = rechitmatchedy - closestPair.first.y();
	  rechitmatchedpullx = ((thit)->localPosition().x() - (closestPair.first.x()))/sqrt(error.xx());
	  rechitmatchedpully = ((thit)->localPosition().y() - (closestPair.first.y()))/sqrt(error.yy());
	}
      }
 
      //Filling Histograms for Matched hits

      if(isrechitmatched){

	if (detid.subdetId() == int(StripSubdetector::TIB)){
	  
	  int Tibisrechitmatched = isrechitmatched;
	  int ilay = tTopo->tibLayer(myid) - 1; //for histogram filling
	  if(Tibisrechitmatched>0){
	    mePosxMatchedTIB[ilay]->Fill(rechitmatchedx);
	    meErrxMatchedTIB[ilay]->Fill(sqrt(rechitmatchederrxx));
	    meErryMatchedTIB[ilay]->Fill(sqrt(rechitmatchederryy));
	    meResxMatchedTIB[ilay]->Fill(rechitmatchedresx);
	    meResyMatchedTIB[ilay]->Fill(rechitmatchedresy);
	    mePullxMatchedTIB[ilay]->Fill(rechitmatchedpullx);
	    mePullyMatchedTIB[ilay]->Fill(rechitmatchedpully);

	  }
	}
	      
	if (detid.subdetId() == int(StripSubdetector::TOB)){
	  
	  int Tobisrechitmatched = isrechitmatched;
	  int ilay = tTopo->tobLayer(myid) - 1; //for histogram filling
	  if(Tobisrechitmatched>0){
	    mePosxMatchedTOB[ilay]->Fill(rechitmatchedx);
	    mePosyMatchedTOB[ilay]->Fill(rechitmatchedy);
	    meErrxMatchedTOB[ilay]->Fill(sqrt(rechitmatchederrxx));
	    meErryMatchedTOB[ilay]->Fill(sqrt(rechitmatchederryy));
	    meResxMatchedTOB[ilay]->Fill(rechitmatchedresx);
	    meResyMatchedTOB[ilay]->Fill(rechitmatchedresy);
	    mePullxMatchedTOB[ilay]->Fill(rechitmatchedpullx);
	    mePullyMatchedTOB[ilay]->Fill(rechitmatchedpully);
	  }
	}
	if (detid.subdetId() == int(StripSubdetector::TID)){
	  
	  int Tidisrechitmatched = isrechitmatched;
	  int ilay = tTopo->tidRing(myid) - 1; //for histogram filling
	  if(Tidisrechitmatched>0){
	    mePosxMatchedTID[ilay]->Fill(rechitmatchedx);
	    mePosyMatchedTID[ilay]->Fill(rechitmatchedy);
	    meErrxMatchedTID[ilay]->Fill(sqrt(rechitmatchederrxx));
	    meErryMatchedTID[ilay]->Fill(sqrt(rechitmatchederryy));
	    meResxMatchedTID[ilay]->Fill(rechitmatchedresx);
	    meResyMatchedTID[ilay]->Fill(rechitmatchedresy);
	    mePullxMatchedTID[ilay]->Fill(rechitmatchedpullx);
	    mePullyMatchedTID[ilay]->Fill(rechitmatchedpully);
	  }
	}
	if (detid.subdetId() == int(StripSubdetector::TEC)){
	  
	  int Tecisrechitmatched = isrechitmatched;
	  int ilay = tTopo->tecRing(myid) - 1; //for histogram filling
	  if(Tecisrechitmatched>0){
	    mePosxMatchedTEC[ilay]->Fill(rechitmatchedx);
	    mePosyMatchedTEC[ilay]->Fill(rechitmatchedy);
	    meErrxMatchedTEC[ilay]->Fill(sqrt(rechitmatchederrxx));
	    meErryMatchedTEC[ilay]->Fill(sqrt(rechitmatchederryy));
	    meResxMatchedTEC[ilay]->Fill(rechitmatchedresx);
	    meResyMatchedTEC[ilay]->Fill(rechitmatchedresy);
	    mePullxMatchedTEC[ilay]->Fill(rechitmatchedpullx);
	    mePullyMatchedTEC[ilay]->Fill(rechitmatchedpully);
	  }
	}
	      
      }
      
	      
      ///////////////////////////////////////////////////////
      // simple hits from matched hits
      ///////////////////////////////////////////////////////
      // Reset variables
      
      isrechitrphi    = 0;
      isrechitsas     = 0;
      rechitrphix =0;
      rechitrphierrxLF =0;
      rechitrphierrxMF =0;
      rechitrphiy =0;
      rechitrphiz =0;
      rechitsasx =0;
      rechitsaserrxLF =0;
      rechitsaserrxMF =0;
      rechitsasy =0;
      rechitsasz =0;
      clusizrphi =0;
      clusizsas =0;
      cluchgrphi =0;
      cluchgsas =0;
      rechitrphiresLF=-999.;
      rechitrphiresMF=-999.;
      rechitrphipullLF=-999.;
      rechitrphipullMF=-999.;
      rechitrphitrackangle =0;
      rechitrphitrackanglebeta =0;
      rechitrphitrackangle2 =0;
      rechitrphitrackwidth =0;
      rechitrphiexpectedwidth =0;
      rechitrphicategory =0;
      rechitrphithickness = 0.;
      rechitsasresLF=-999.;
      rechitsasresMF=-999.;
      rechitsaspullLF=-999.;
      rechitsaspullMF=-999.;
      rechitsastrackangle =0;
      rechitsastrackanglebeta =0;
      rechitsasthickness = 0;

      GluedGeomDet * gdet;
      const GeomDetUnit * monodet;
      const SiStripRecHit2D *monohit;
      const StripGeomDetUnit * stripdet;

      if (matchedhit)
	{
          auto hm =matchedhit->monoHit();
	  monohit=&hm;
	  //	  const GeomDetUnit * monodet=gdet->monoDet();
	  gdet=(GluedGeomDet *)tracker2->idToDet(matchedhit->geographicalId());
	  monodet=gdet->monoDet();
	  GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	  LocalVector monotkdir=monodet->toLocal(gtrkdir);
	  //	  const GeomDetUnit *  det = tracker.idToDetUnit(detid);
	  //stripdet=(const StripGeomDetUnit*)(gdet);
	  stripdet=(const StripGeomDetUnit*)(monodet);
	  //	  const StripTopology &topol2=(const StripTopology&)stripdet->topology();

	  if(monotkdir.z()!=0){
	    anglealpha = atan(monotkdir.x()/monotkdir.z())*180/TMath::Pi();
	    anglebeta = atan(monotkdir.y()/monotkdir.z())*180/TMath::Pi();
	  }
	  
	  if(monohit){

	    const StripTopology &topol=(const StripTopology&)stripdet->topology();

	    position = monohit->localPosition();
	    error = monohit->localPositionError();
	    Mposition = topol.measurementPosition(position);
	    Merror = topol.measurementError(position,error);

	    LocalVector drift = stripcpe->driftDirection(stripdet);
	    float thickness=stripdet->surface().bounds().thickness();
	    rechitrphithickness = thickness;

	    //cout<<"thickness = "<<thickness<<endl;
	    float pitch = topol.localPitch(position);
	    //cout<<"Valid:pitch = "<<pitch<<endl;
	    float tanalpha = tan(anglealpha/57.3);
	    //cout<<"Valid:tanalpha = "<<tanalpha<<endl;
	    float tanalphaL = drift.x()/drift.z();
	    //float tanalphaLbis = driftbis.x()/driftbis.z();
	    //float tanalphaLter = driftter.x()/driftter.z();
	    //cout<<"Validmonofrommatched:drift.x() = "<<drift.x()<<endl;
	    //cout<<"Valid:drift.x() = "<<drift.x()<<endl;
	    //cout<<"Valid:driftbis.x() = "<<driftbis.x()<<endl;
	    //cout<<"Valid:driftter.x() = "<<driftter.x()<<endl;
	    //cout<<"Valid:driftter.z() = "<<driftter.z()<<endl;
	    //cout<<"Valid:tanalphaL = "<<tanalphaL<<endl;
	    Wtrack = fabs((thickness/pitch)*tanalpha - (thickness/pitch)*tanalphaL);
	    //cout<<"Valid1:Wtrack = "<<Wtrack<<endl;
	    float SLorentz = 0.5*(thickness/pitch)*tanalphaL;
	    //int nstrips = topol.nstrips(); 
	    //clusterWidth = cluster->amplitudes().size();
	    int Sp = int(position.x()/pitch+SLorentz+0.5*Wtrack);
	    int Sm = int(position.x()/pitch+SLorentz-0.5*Wtrack);
	    Wexp = 1+Sp-Sm;
	    //cout<<"DebugLine22"<<endl;

	    isrechitrphi = 1;
	    //cout<<"DebugLine23"<<endl;
	    //		const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	    SiStripRecHit2D::ClusterRef cluster=monohit->cluster();
	    //SiStripRecHit1D::ClusterRef cluster=monohit->cluster();
	    clusiz=0;
	    totcharge=0;
	    clusiz = cluster->amplitudes().size();
	    //	    cout<<"clusiz = "<<clusiz<<endl;
	    const std::vector<uint8_t> amplitudes=cluster->amplitudes();
	    for(size_t ia=0; ia<amplitudes.size();ia++){
	      totcharge+=amplitudes[ia];
	    }
	    rechitrphix = position.x();
	    rechitrphiy = position.y();
	    rechitrphiz = position.z();
	    rechitrphierrxLF = error.xx();
	    rechitrphierrxMF = Merror.uu();
	    //	    cout<<"rechitrphierrxMF from Matched hit= "<<sqrt(rechitrphierrxMF)<<endl;
	    clusizrphi = clusiz;
	    cluchgrphi = totcharge;

	    //Association of the rechit to the simhit
	    mindist = 999999;
	    matched.clear();  
	    //		matched = associate.associateHit(*hit);
	    matched = associate.associateHit(*monohit);
	    if(!matched.empty()){
	      //		  cout << "\t\t\tmatched  " << matched.size() << endl;
	      //	      cout<<"associatesimplehit"<<endl;
	      for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
		dist = abs((monohit)->localPosition().x() - (*m).localPosition().x());
		if(dist<mindist){
		  mindist = dist;
		  closest = (*m);
		}
		rechitrphiresLF = rechitrphix - closest.localPosition().x();
		rechitrphiresMF = Mposition.x() - (topol.measurementPosition(closest.localPosition())).x();
		rechitrphipullLF =  rechitrphiresLF/sqrt(rechitrphierrxLF);
		rechitrphipullMF = rechitrphiresMF/sqrt(rechitrphierrxMF);
		//cout<<"rechitrphiresMF == "<<rechitrphiresMF<<endl;
		//cout<<"rechitrphierrxMF == "<<rechitrphierrxMF<<endl;
		//cout<<"rechitrphierrxLF == "<<rechitrphierrxLF<<endl;
		//cout<<"rechitrphipullMF == "<<rechitrphipullMF<<endl;

	      }
	    }
	    rechitrphitrackangle = anglealpha;
	    rechitrphitrackanglebeta = anglebeta;
	    //rechitrphitrackangle = tanalphaL;
	    //cout<<"Wtrack = "<<Wtrack<<endl;
	    rechitrphitrackwidth = Wtrack;
	    rechitrphiexpectedwidth = Wexp;
	    clusterWidth = clusiz;
	    unsigned int iopt;
	    if (clusterWidth > Wexp + 2) {
	      iopt = 1;
	    } else if (Wexp == 1) {
	      iopt = 2;
	    } else if (clusterWidth <= Wexp) {
	      iopt = 3;
	    } else {
	      iopt = 4;
	    }
	    rechitrphicategory = iopt;
	  }
	   
          auto s =matchedhit->stereoHit();
	  const SiStripRecHit2D *stereohit=&s;
	  const GeomDetUnit * stereodet=gdet->stereoDet(); 
	  //	  GlobalVector 
	  gtrkdir=gdet->toGlobal(trackdirection);
	  LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	  if(stereotkdir.z()!=0){
	    anglealpha = atan(stereotkdir.x()/stereotkdir.z())*180/TMath::Pi();
	    anglebeta = atan(stereotkdir.y()/stereotkdir.z())*180/TMath::Pi();
	  }
	  
	  if (stereohit)
	    {
	      //	      cout<<"stereohit from matched hit"<<endl;
	      isrechitsas = 1;
	      SiStripRecHit2D::ClusterRef cluster=stereohit->cluster();
	    
	      //	      const GeomDetUnit *  det = tracker.idToDetUnit(detid2);
	      const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(stereodet);
	      const StripTopology &topol=(const StripTopology&)stripdet->topology();

	      position = stereohit->localPosition();
	      //  Mposition = topol.measurementPosition(position);
	      error = stereohit->localPositionError();
	      Mposition = topol.measurementPosition(position);
	      Merror = topol.measurementError(position,error);

	      //LocalVector drift= driftDirection(stripdet);
	      LocalVector drift = stripcpe->driftDirection(stripdet);
	      float thickness=stripdet->surface().bounds().thickness();
	      rechitsasthickness = thickness;
	      //cout<<"thickness = "<<thickness<<endl;
	      float pitch = topol.localPitch(position);
	      //cout<<"Valid:pitch = "<<pitch<<endl;
	      float tanalpha = tan(anglealpha/57.3);
	      //cout<<"Valid:tanalpha = "<<tanalpha<<endl;
	      float tanalphaL = drift.x()/drift.z();
	      //cout<<"Validstereofrommatched:drift.x() = "<<drift.x()<<endl;
	      //cout<<"Valid:drift.z() = "<<drift.z()<<endl;
	      //cout<<"Valid:tanalphaL = "<<tanalphaL<<endl;
	      Wtrack = fabs((thickness/pitch)*tanalpha - (thickness/pitch)*tanalphaL);
	      //cout<<"Valid:Wtrack = "<<Wtrack<<endl;
	      float SLorentz = 0.5*(thickness/pitch)*tanalphaL;
	      //int nstrips = topol.nstrips(); 
	      int Sp = int(position.x()/pitch+SLorentz+0.5*Wtrack);
	      int Sm = int(position.x()/pitch+SLorentz-0.5*Wtrack);
	      Wexp = 1+Sp-Sm;


	      clusiz=0;
	      totcharge=0;
	      clusiz = cluster->amplitudes().size();
	      const std::vector<uint8_t> amplitudes=cluster->amplitudes();
	      for(size_t ia=0; ia<amplitudes.size();ia++){
		totcharge+=amplitudes[ia];
	      }
	      rechitsasx = position.x();
	      rechitsasy = position.y();
	      rechitsasz = position.z();
	      rechitsaserrxLF = error.xx();
	      //	      cout<<"rechitsaserrxLF = "<<rechitsaserrxLF<<endl;
	      rechitsaserrxMF = Merror.uu();
	      //	      cout<<"rechitsaserrxMF from Matched hit = "<<sqrt(rechitsaserrxMF)<<endl;
	      clusizsas = clusiz;
	      cluchgsas = totcharge;

	      //Association of the rechit to the simhit
	      mindist = 999999;
	      matched.clear();  
	      matched = associate.associateHit(*stereohit);
	      if(!matched.empty()){
		//		  cout << "\t\t\tmatched  " << matched.size() << endl;
		for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
		  dist = abs((stereohit)->localPosition().x() - (*m).localPosition().x());
		  if(dist<mindist){
		    mindist = dist;
		    closest = (*m);
		  }

		  rechitsasresLF = rechitsasx - closest.localPosition().x();
		  rechitsasresMF = Mposition.x() - (topol.measurementPosition(closest.localPosition())).x();
		  rechitsaspullLF = rechitsasresLF/sqrt(rechitsaserrxLF);
		  rechitsaspullMF = rechitsasresMF/sqrt(rechitsaserrxMF);
		  
// 		  cout<<"rechitsasresMF == "<<rechitsasresMF<<endl;
// 		  cout<<"rechitsaserrxMF == "<<rechitsaserrxMF<<endl;
// 		  cout<<"rechitsaserrxLF == "<<rechitsaserrxLF<<endl;
// 		  cout<<"rechitsaspullMF == "<<rechitsaspullMF<<endl;
		  
		}
	      }
	      rechitsastrackangle = anglealpha;
	      rechitsastrackanglebeta = anglebeta;
	      rechitsastrackwidth = Wtrack;
	      rechitsasexpectedwidth = Wexp;

	      clusterWidth = clusiz;
	      unsigned int iopt;
	      if (clusterWidth > Wexp + 2) {
		iopt = 1;
	      } else if (Wexp == 1) {
		iopt = 2;
	      } else if (clusterWidth <= Wexp) {
		iopt = 3;
	      } else {
		iopt = 4;
	      }
	      rechitsascategory = iopt;
	    }
	}


      // A VIRER !!!!!!!!!!!!!!!!!!!!

      //    isrechitrphi = 0;
      //isrechitsas = 0;
      
      
      if(hit1d){
	// simple hits are mono or stereo
	//	cout<<"simple hit"<<endl;
	if (StripSubdet.stereo() == 0){
	  isrechitrphi = 1;
	  //	  cout<<"simple hit mono"<<endl;

	  const GeomDetUnit *  det = tracker.idToDetUnit(detid2);
	  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
	  const StripTopology &topol=(const StripTopology&)stripdet->topology();

	  float anglealpha = atan(trackdirection.x()/trackdirection.z())*180/TMath::Pi();
	  float anglebeta = atan(trackdirection.y()/trackdirection.z())*180/TMath::Pi();

	  //SiStripRecHit2D::ClusterRef cluster=hit->cluster();
	  SiStripRecHit1D::ClusterRef cluster=hit1d->cluster();

	  position = thit->localPosition();
	  error = thit->localPositionError();
	  Mposition = topol.measurementPosition(position);
	  Merror = topol.measurementError(position,error);

	  LocalVector drift = stripcpe->driftDirection(stripdet);
	  float thickness=stripdet->surface().bounds().thickness();
	  rechitrphithickness = thickness;
	  //cout<<"Valid:thickness = "<<thickness<<endl;
	  float pitch = topol.localPitch(position);
	  //cout<<"Valid:pitch = "<<pitch<<endl;
	  float tanalpha = tan(anglealpha/57.3);
	  //cout<<"Valid:tanalpha = "<<tanalpha<<endl;
	  float tanalphaL = drift.x()/drift.z();
	  //cout<<"Valid:tanalphaL = "<<tanalphaL<<endl;
	  //	  float tanalphaLcpe = driftcpe.x()/driftcpe.z();
	  //cout<<"Valid:tanalphaLcpe = "<<tanalphaLcpe<<endl;
	  //cout<<"Validmono:drift.x() = "<<drift.x()<<endl;
	  //cout<<"Valid:drift.z() = "<<drift.z()<<endl;
	  //cout<<"Valid:tanalphaL = "<<tanalphaL<<endl;
	  Wtrack = fabs((thickness/pitch)*tanalpha - (thickness/pitch)*tanalphaL);
	  //       fabs((thickness/pitch)*tanalpha - (thickness/pitch)*tanalphaL);
	  //cout<<"Valid2:Wtrack = "<<Wtrack<<endl;
	  float SLorentz = 0.5*(thickness/pitch)*tanalphaL;
	  //int nstrips = topol.nstrips(); 
	  int Sp = int(position.x()/pitch+SLorentz+0.5*Wtrack);
	  int Sm = int(position.x()/pitch+SLorentz-0.5*Wtrack);
	  Wexp = 1+Sp-Sm;

	  clusiz=0;
	  totcharge=0;
	  clusiz = cluster->amplitudes().size();
	  //cout<<"cluster->firstStrip() = "<<cluster->firstStrip()<<endl;
	  const std::vector<uint8_t> amplitudes=cluster->amplitudes();
	  for(size_t ia=0; ia<amplitudes.size();ia++){
	    totcharge+=amplitudes[ia];
	  }
	  rechitrphix = position.x();
	  rechitrphiy = position.y();
	  rechitrphiz = position.z();
	  rechitrphierrx = error.xx();
	  rechitrphierrxLF = error.xx();
	  rechitrphierrxMF = Merror.uu();
	  //cout<<"rechitrphierrxMF simple hit= "<<sqrt(rechitrphierrxMF)<<endl;
	  clusizrphi = clusiz;
	  //cout<<"clusizrphi = "<<clusiz<<endl;
	  cluchgrphi = totcharge;

	  //Association of the rechit to the simhit
	  mindist = 999999;
	  matched.clear();  
	  matched = associate.associateHit(*hit1d);
	  if(!matched.empty()){
	    //		  cout << "\t\t\tmatched  " << matched.size() << endl;
	    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	      dist = abs((hit1d)->localPosition().x() - (*m).localPosition().x());
	      if(dist<mindist){
		mindist = dist;
		closest = (*m);
	      }
	      rechitrphiresLF = rechitrphix - closest.localPosition().x();
	      rechitrphiresMF = Mposition.x() - (topol.measurementPosition(closest.localPosition())).x();
	      rechitrphipullLF = (thit->localPosition().x() - (closest).localPosition().x())/sqrt(error.xx());
	      rechitrphipullMF = rechitrphiresMF/sqrt(rechitrphierrxMF);
	    }
	  }
	  rechitrphitrackangle = anglealpha;
	  rechitrphitrackanglebeta = anglebeta;
	  rechitrphitrackwidth = Wtrack;
	  rechitrphiexpectedwidth = Wexp;

	  clusterWidth = clusiz;
	  unsigned int iopt;
	  if (clusterWidth > Wexp + 2) {
	    iopt = 1;
	  } else if (Wexp == 1) {
	    iopt = 2;
	  } else if (clusterWidth <= Wexp) {
	    iopt = 3;
	  } else {
	    iopt = 4;
	  }
	  rechitrphicategory = iopt;
	  
// 	  if (rechitrphiexpectedwidth == 1 && clusterWidth == 3) {
// 	  //if ( clusterWidth == 3) {
// 	    cout<<"TRUE"<<endl;
// 	    cout<<"TestClus2:Position SH = "<<(closest).localPosition().x()<<" , "<<(topol.measurementPosition(closest.localPosition())).x()<<endl;
// 	    cout<<"TestClus2:Position RH = "<<thit->localPosition().x()<<" ,"<<Mposition.x()<<endl;
// 	    cout<<"TestClus2:residue = "<<rechitrphiresMF<<endl;
// 	    short firstStrip = cluster->firstStrip();
// 	    short lastStrip = firstStrip + clusterWidth - 1;
// 	    cout<<"TestClus2:firstStrip = "<<firstStrip<<endl;
// 	    cout<<"TestClus2:lastStrip = "<<lastStrip<<endl;
// 	    cout<<"TestClus2:detid = "<<detid.subdetId()<<endl;
// 	    for(size_t ia=0; ia<amplitudes.size();ia++){
// 	      cout<<"ia, TestClus2:charge= "<<ia<<" , "<<amplitudes[ia]<<endl;
// 	    }
// 	    cout<<"TestClus2:Trackwidth = "<<Wtrack<<endl;
// 	  }
	  
	
	  //cout<<"rechitrphicategory = "<<rechitrphicategory<<endl;

	  //	  if ((detid.subdetId() == int(StripSubdetector::TID)) || (detid.subdetId() == int(StripSubdetector::TEC))) {
	    //if ((detid.subdetId() == int(StripSubdetector::TIB))) {
	   
// 	    if (clusterWidth ==2 && Wexp == 1 && Wtrack<0.1) {
// 	      cout<<"TestClus:begin"<<endl;
// 	      LocalVector  drift2 = drift * fabs(thickness/drift.z());       
// 	      LocalPoint result2=LocalPoint(position.x()-drift2.x()/2,position.y()-drift2.y()/2,0);
// 	      MeasurementPoint mpoint=topol.measurementPosition(result2);
// 	      cout<<"TestClus:Position SH = "<<(closest).localPosition().x()<<" , "<<(topol.measurementPosition(closest.localPosition())).x()<<endl;
// 	      cout<<"TestClus:Position RH = "<<thit->localPosition().x()<<" ,"<<Mposition.x()<<endl;
// 	      cout<<"TestClus:Position RH no drift= "<<thit->localPosition().x() - drift2.x()/2<<" , "<<mpoint.x()<<endl;
// 	      cout<<"TestClus:Drift= "<<drift.x()<<endl;
// 	      cout<<"TestClus:residue = "<<rechitrphiresMF<<endl;
// 	      for(size_t ia=0; ia<amplitudes.size();ia++){
// 		cout<<"ia, TestClus:charge= "<<ia<<" , "<<amplitudes[ia]<<endl;
// 	      }
// 	      cout<<"TestClus:Trackwidth = "<<Wtrack<<endl;
// 	      short firstStrip = cluster->firstStrip();
// 	      short lastStrip = firstStrip + clusterWidth - 1;
// 	      cout<<"TestClus:firstStrip = "<<firstStrip<<endl;
// 	      cout<<"TestClus:lastStrip = "<<lastStrip<<endl;
// 	      cout<<"TestClus:detid = "<<detid.subdetId()<<endl;
// 	      int nstrips = topol.nstrips(); 
// 	      cout<<"TestClus:nstrips = "<<nstrips<<endl;
// 	      cout<<"TestClus:anglealpha = "<<anglealpha<<endl;
// 	      cout<<"TestClus:end"<<endl;
// 	      positionshx = (topol.measurementPosition(closest.localPosition())).x();

// 	      if ((positionshx - int(positionshx)) > 0.5) {
// 		if (lastStrip > int(positionshx)) secondstrip = 1;
// 		if (lastStrip = int(positionshx)) secondstrip = -1;
// 	      }
// 	      if ((positionshx - int(positionshx)) < 0.5) {
// 		if (lastStrip > int(positionshx)) secondstrip = -1;
// 		if (lastStrip = int(positionshx)) secondstrip = 1;
// 	      }

// 	    }

	    //}
	  
// 	  cout<<"int() = "<<int((topol.measurementPosition(closest.localPosition())).x())<<endl;
// 	  diff = int((topol.measurementPosition(closest.localPosition())).x()) -topol.measurementPosition(closest.localPosition()).x();
// 	  cout<<"diff = "<<diff<<endl;
// 	  if (clusterWidth ==2 && Wexp == 1 && Wtrack<1) {
// 	    if ((abs(1 + diff) <0.2) || (abs(diff) <0.2)) {
// 	      //	      isrechitrphi = 0;
// 	      cout<<"vire"<<endl;
// 	    }
// 	  }
// 	  positionshx = (topol.measurementPosition(closest.localPosition())).x();


	}

	if (StripSubdet.stereo() == 1){

	  //cout<<"simple hit stereo"<<endl;
	  isrechitsas = 1;

          const GeomDetUnit *  det = tracker.idToDetUnit(detid2);
          const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
          const StripTopology &topol=(const StripTopology&)stripdet->topology();

          float anglealpha = atan(trackdirection.x()/trackdirection.z())*180/TMath::Pi();
          float anglebeta = atan(trackdirection.y()/trackdirection.z())*180/TMath::Pi();

          //SiStripRecHit2D::ClusterRef cluster=hit->cluster();
	  SiStripRecHit1D::ClusterRef cluster=hit1d->cluster();


          position = thit->localPosition();
          error = thit->localPositionError();
          Mposition = topol.measurementPosition(position);
          Merror = topol.measurementError(position,error);

	  //	  LocalVector drift= driftDirection(stripdet);
	  LocalVector drift = stripcpe->driftDirection(stripdet);
	  float thickness=stripdet->surface().bounds().thickness();
	  rechitsasthickness = thickness;
	  //cout<<"thickness = "<<thickness<<endl;
	  float pitch = topol.localPitch(position);
	  //cout<<"Valid:pitch = "<<pitch<<endl;
	  float tanalpha = tan(anglealpha/57.3);
	  //cout<<"Valid:tanalpha = "<<tanalpha<<endl;
	  float tanalphaL = drift.x()/drift.z();
	  //cout<<"Validstereo:drift.x() = "<<drift.x()<<endl;
	  //cout<<"Valid:drift.z() = "<<drift.z()<<endl;
	  //cout<<"Valid:tanalphaL = "<<tanalphaL<<endl;
	  Wtrack = fabs((thickness/pitch)*tanalpha - (thickness/pitch)*tanalphaL);
	  //cout<<"Valid:Wtrack = "<<Wtrack<<endl;
	  float SLorentz = 0.5*(thickness/pitch)*tanalphaL;
	  //int nstrips = topol.nstrips(); 
	  int Sp = int(position.x()/pitch+SLorentz+0.5*Wtrack);
	  int Sm = int(position.x()/pitch+SLorentz-0.5*Wtrack);
	  Wexp = 1+Sp-Sm;

	  clusiz=0;
	  totcharge=0;
	  clusiz = cluster->amplitudes().size();
	  const std::vector<uint8_t> amplitudes=cluster->amplitudes();
	  for(size_t ia=0; ia<amplitudes.size();ia++){
	    totcharge+=amplitudes[ia];
	  }
	  rechitsasx = position.x();
	  rechitsasy = position.y();
	  rechitsasz = position.z();
	  rechitsaserrxLF = error.xx();
	  //cout<<"rechitsaserrxLF = "<<rechitsaserrxLF<<endl;
	  rechitsaserrxMF = Merror.uu();
	  //cout<<"rechitsaserrxMF simple hit= "<<sqrt(rechitsaserrxMF)<<endl;
	  clusizsas = clusiz;
	  cluchgsas = totcharge;

	  //Association of the rechit to the simhit
	  mindist = 999999;
	  matched.clear();  
	  matched = associate.associateHit(*hit1d);
	  if(!matched.empty()){
	    //		  cout << "\t\t\tmatched  " << matched.size() << endl;
	    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	      dist = abs((hit1d)->localPosition().x() - (*m).localPosition().x());
	      if(dist<mindist){
		mindist = dist;
		closest = (*m);
	      }

	      rechitsasresLF = rechitsasx - closest.localPosition().x();
	      rechitsasresMF = Mposition.x() - (topol.measurementPosition(closest.localPosition())).x();
	      rechitsaspullLF = (thit->localPosition().x() - (closest).localPosition().x())/sqrt(error.xx());
	      rechitsaspullMF = rechitsasresMF/sqrt(rechitsaserrxMF);

	    }
	  }
	  rechitsastrackangle = anglealpha;
	  rechitsastrackanglebeta = anglebeta;
	  rechitsastrackwidth = Wtrack;
	  rechitsasexpectedwidth = Wexp;

	  clusterWidth = clusiz;
	  unsigned int iopt;
	  if (clusterWidth > Wexp + 2) {
	    iopt = 1;
	  } else if (Wexp == 1) {
	    iopt = 2;
	  } else if (clusterWidth <= Wexp) {
	    iopt = 3;
	  } else {
	    iopt = 4;
	  }
	  rechitsascategory = iopt;
	}
	//isrechitsas = 0;
      }


      if(hit2d){
	// simple hits are mono or stereo
	//	cout<<"simple hit"<<endl;
	if (StripSubdet.stereo() == 0){
	  isrechitrphi = 1;
	  //	  cout<<"simple hit mono"<<endl;

	  const GeomDetUnit *  det = tracker.idToDetUnit(detid2);
	  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
	  const StripTopology &topol=(const StripTopology&)stripdet->topology();

	  float anglealpha = atan(trackdirection.x()/trackdirection.z())*180/TMath::Pi();
	  float anglebeta = atan(trackdirection.y()/trackdirection.z())*180/TMath::Pi();

	  SiStripRecHit2D::ClusterRef cluster=hit2d->cluster();

	  position = thit->localPosition();
	  error = thit->localPositionError();
	  Mposition = topol.measurementPosition(position);
	  Merror = topol.measurementError(position,error);

	  LocalVector drift = stripcpe->driftDirection(stripdet);
	  float thickness=stripdet->surface().bounds().thickness();
	  rechitrphithickness = thickness;
	  //cout<<"Valid:thickness = "<<thickness<<endl;
	  float pitch = topol.localPitch(position);
	  //cout<<"Valid:pitch = "<<pitch<<endl;
	  float tanalpha = tan(anglealpha/57.3);
	  //cout<<"Valid:tanalpha = "<<tanalpha<<endl;
	  float tanalphaL = drift.x()/drift.z();
	  //cout<<"Valid:tanalphaL = "<<tanalphaL<<endl;
	  //	  float tanalphaLcpe = driftcpe.x()/driftcpe.z();
	  //cout<<"Valid:tanalphaLcpe = "<<tanalphaLcpe<<endl;
	  //cout<<"Validmono:drift.x() = "<<drift.x()<<endl;
	  //cout<<"Valid:drift.z() = "<<drift.z()<<endl;
	  //cout<<"Valid:tanalphaL = "<<tanalphaL<<endl;
	  Wtrack = fabs((thickness/pitch)*tanalpha - (thickness/pitch)*tanalphaL);
	  //       fabs((thickness/pitch)*tanalpha - (thickness/pitch)*tanalphaL);
	  //cout<<"Valid2:Wtrack = "<<Wtrack<<endl;
	  float SLorentz = 0.5*(thickness/pitch)*tanalphaL;
	  //int nstrips = topol.nstrips(); 
	  int Sp = int(position.x()/pitch+SLorentz+0.5*Wtrack);
	  int Sm = int(position.x()/pitch+SLorentz-0.5*Wtrack);
	  Wexp = 1+Sp-Sm;

	  clusiz=0;
	  totcharge=0;
	  clusiz = cluster->amplitudes().size();
	  //cout<<"cluster->firstStrip() = "<<cluster->firstStrip()<<endl;
	  const std::vector<uint8_t> amplitudes=cluster->amplitudes();
	  for(size_t ia=0; ia<amplitudes.size();ia++){
	    totcharge+=amplitudes[ia];
	  }
	  rechitrphix = position.x();
	  rechitrphiy = position.y();
	  rechitrphiz = position.z();
	  rechitrphierrx = error.xx();
	  rechitrphierrxLF = error.xx();
	  rechitrphierrxMF = Merror.uu();
	  //cout<<"rechitrphierrxMF simple hit= "<<sqrt(rechitrphierrxMF)<<endl;
	  clusizrphi = clusiz;
	  //cout<<"clusizrphi = "<<clusiz<<endl;
	  cluchgrphi = totcharge;

	  //Association of the rechit to the simhit
	  mindist = 999999;
	  matched.clear();  
	  matched = associate.associateHit(*hit2d);
	  if(!matched.empty()){
	    //		  cout << "\t\t\tmatched  " << matched.size() << endl;
	    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	      dist = abs((hit2d)->localPosition().x() - (*m).localPosition().x());
	      if(dist<mindist){
		mindist = dist;
		closest = (*m);
	      }
	      rechitrphiresLF = rechitrphix - closest.localPosition().x();
	      rechitrphiresMF = Mposition.x() - (topol.measurementPosition(closest.localPosition())).x();
	      rechitrphipullLF = (thit->localPosition().x() - (closest).localPosition().x())/sqrt(error.xx());
	      rechitrphipullMF = rechitrphiresMF/sqrt(rechitrphierrxMF);
	    }
	  }
	  rechitrphitrackangle = anglealpha;
	  rechitrphitrackanglebeta = anglebeta;
	  rechitrphitrackwidth = Wtrack;
	  rechitrphiexpectedwidth = Wexp;

	  clusterWidth = clusiz;
	  unsigned int iopt;
	  if (clusterWidth > Wexp + 2) {
	    iopt = 1;
	  } else if (Wexp == 1) {
	    iopt = 2;
	  } else if (clusterWidth <= Wexp) {
	    iopt = 3;
	  } else {
	    iopt = 4;
	  }
	  rechitrphicategory = iopt;
	  
// 	  if (rechitrphiexpectedwidth == 1 && clusterWidth == 3) {
// 	  //if ( clusterWidth == 3) {
// 	    cout<<"TRUE"<<endl;
// 	    cout<<"TestClus2:Position SH = "<<(closest).localPosition().x()<<" , "<<(topol.measurementPosition(closest.localPosition())).x()<<endl;
// 	    cout<<"TestClus2:Position RH = "<<thit->localPosition().x()<<" ,"<<Mposition.x()<<endl;
// 	    cout<<"TestClus2:residue = "<<rechitrphiresMF<<endl;
// 	    short firstStrip = cluster->firstStrip();
// 	    short lastStrip = firstStrip + clusterWidth - 1;
// 	    cout<<"TestClus2:firstStrip = "<<firstStrip<<endl;
// 	    cout<<"TestClus2:lastStrip = "<<lastStrip<<endl;
// 	    cout<<"TestClus2:detid = "<<detid.subdetId()<<endl;
// 	    for(size_t ia=0; ia<amplitudes.size();ia++){
// 	      cout<<"ia, TestClus2:charge= "<<ia<<" , "<<amplitudes[ia]<<endl;
// 	    }
// 	    cout<<"TestClus2:Trackwidth = "<<Wtrack<<endl;
// 	  }
	  
	
	  //cout<<"rechitrphicategory = "<<rechitrphicategory<<endl;

	  //	  if ((detid.subdetId() == int(StripSubdetector::TID)) || (detid.subdetId() == int(StripSubdetector::TEC))) {
	    //if ((detid.subdetId() == int(StripSubdetector::TIB))) {
	   
// 	    if (clusterWidth ==2 && Wexp == 1 && Wtrack<0.1) {
// 	      cout<<"TestClus:begin"<<endl;
// 	      LocalVector  drift2 = drift * fabs(thickness/drift.z());       
// 	      LocalPoint result2=LocalPoint(position.x()-drift2.x()/2,position.y()-drift2.y()/2,0);
// 	      MeasurementPoint mpoint=topol.measurementPosition(result2);
// 	      cout<<"TestClus:Position SH = "<<(closest).localPosition().x()<<" , "<<(topol.measurementPosition(closest.localPosition())).x()<<endl;
// 	      cout<<"TestClus:Position RH = "<<thit->localPosition().x()<<" ,"<<Mposition.x()<<endl;
// 	      cout<<"TestClus:Position RH no drift= "<<thit->localPosition().x() - drift2.x()/2<<" , "<<mpoint.x()<<endl;
// 	      cout<<"TestClus:Drift= "<<drift.x()<<endl;
// 	      cout<<"TestClus:residue = "<<rechitrphiresMF<<endl;
// 	      for(size_t ia=0; ia<amplitudes.size();ia++){
// 		cout<<"ia, TestClus:charge= "<<ia<<" , "<<amplitudes[ia]<<endl;
// 	      }
// 	      cout<<"TestClus:Trackwidth = "<<Wtrack<<endl;
// 	      short firstStrip = cluster->firstStrip();
// 	      short lastStrip = firstStrip + clusterWidth - 1;
// 	      cout<<"TestClus:firstStrip = "<<firstStrip<<endl;
// 	      cout<<"TestClus:lastStrip = "<<lastStrip<<endl;
// 	      cout<<"TestClus:detid = "<<detid.subdetId()<<endl;
// 	      int nstrips = topol.nstrips(); 
// 	      cout<<"TestClus:nstrips = "<<nstrips<<endl;
// 	      cout<<"TestClus:anglealpha = "<<anglealpha<<endl;
// 	      cout<<"TestClus:end"<<endl;
// 	      positionshx = (topol.measurementPosition(closest.localPosition())).x();

// 	      if ((positionshx - int(positionshx)) > 0.5) {
// 		if (lastStrip > int(positionshx)) secondstrip = 1;
// 		if (lastStrip = int(positionshx)) secondstrip = -1;
// 	      }
// 	      if ((positionshx - int(positionshx)) < 0.5) {
// 		if (lastStrip > int(positionshx)) secondstrip = -1;
// 		if (lastStrip = int(positionshx)) secondstrip = 1;
// 	      }

// 	    }

	    //}
	  
// 	  cout<<"int() = "<<int((topol.measurementPosition(closest.localPosition())).x())<<endl;
// 	  diff = int((topol.measurementPosition(closest.localPosition())).x()) -topol.measurementPosition(closest.localPosition()).x();
// 	  cout<<"diff = "<<diff<<endl;
// 	  if (clusterWidth ==2 && Wexp == 1 && Wtrack<1) {
// 	    if ((abs(1 + diff) <0.2) || (abs(diff) <0.2)) {
// 	      //	      isrechitrphi = 0;
// 	      cout<<"vire"<<endl;
// 	    }
// 	  }
// 	  positionshx = (topol.measurementPosition(closest.localPosition())).x();


	}

	if (StripSubdet.stereo() == 1){

	  //cout<<"simple hit stereo"<<endl;
	  isrechitsas = 1;

          const GeomDetUnit *  det = tracker.idToDetUnit(detid2);
          const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
          const StripTopology &topol=(const StripTopology&)stripdet->topology();

          float anglealpha = atan(trackdirection.x()/trackdirection.z())*180/TMath::Pi();
          float anglebeta = atan(trackdirection.y()/trackdirection.z())*180/TMath::Pi();

          SiStripRecHit2D::ClusterRef cluster=hit2d->cluster();


          position = thit->localPosition();
          error = thit->localPositionError();
          Mposition = topol.measurementPosition(position);
          Merror = topol.measurementError(position,error);

	  //	  LocalVector drift= driftDirection(stripdet);
	  LocalVector drift = stripcpe->driftDirection(stripdet);
	  float thickness=stripdet->surface().bounds().thickness();
	  rechitsasthickness = thickness;
	  //cout<<"thickness = "<<thickness<<endl;
	  float pitch = topol.localPitch(position);
	  //cout<<"Valid:pitch = "<<pitch<<endl;
	  float tanalpha = tan(anglealpha/57.3);
	  //cout<<"Valid:tanalpha = "<<tanalpha<<endl;
	  float tanalphaL = drift.x()/drift.z();
	  //cout<<"Validstereo:drift.x() = "<<drift.x()<<endl;
	  //cout<<"Valid:drift.z() = "<<drift.z()<<endl;
	  //cout<<"Valid:tanalphaL = "<<tanalphaL<<endl;
	  Wtrack = fabs((thickness/pitch)*tanalpha - (thickness/pitch)*tanalphaL);
	  //cout<<"Valid:Wtrack = "<<Wtrack<<endl;
	  float SLorentz = 0.5*(thickness/pitch)*tanalphaL;
	  //int nstrips = topol.nstrips(); 
	  int Sp = int(position.x()/pitch+SLorentz+0.5*Wtrack);
	  int Sm = int(position.x()/pitch+SLorentz-0.5*Wtrack);
	  Wexp = 1+Sp-Sm;

	  clusiz=0;
	  totcharge=0;
	  clusiz = cluster->amplitudes().size();
	  const std::vector<uint8_t> amplitudes=cluster->amplitudes();
	  for(size_t ia=0; ia<amplitudes.size();ia++){
	    totcharge+=amplitudes[ia];
	  }
	  rechitsasx = position.x();
	  rechitsasy = position.y();
	  rechitsasz = position.z();
	  rechitsaserrxLF = error.xx();
	  //cout<<"rechitsaserrxLF = "<<rechitsaserrxLF<<endl;
	  rechitsaserrxMF = Merror.uu();
	  //cout<<"rechitsaserrxMF simple hit= "<<sqrt(rechitsaserrxMF)<<endl;
	  clusizsas = clusiz;
	  cluchgsas = totcharge;

	  //Association of the rechit to the simhit
	  mindist = 999999;
	  matched.clear();  
	  matched = associate.associateHit(*hit2d);
	  if(!matched.empty()){
	    //		  cout << "\t\t\tmatched  " << matched.size() << endl;
	    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	      dist = abs((hit2d)->localPosition().x() - (*m).localPosition().x());
	      if(dist<mindist){
		mindist = dist;
		closest = (*m);
	      }

	      rechitsasresLF = rechitsasx - closest.localPosition().x();
	      rechitsasresMF = Mposition.x() - (topol.measurementPosition(closest.localPosition())).x();
	      rechitsaspullLF = (thit->localPosition().x() - (closest).localPosition().x())/sqrt(error.xx());
	      rechitsaspullMF = rechitsasresMF/sqrt(rechitsaserrxMF);

	    }
	  }
	  rechitsastrackangle = anglealpha;
	  rechitsastrackanglebeta = anglebeta;
	  rechitsastrackwidth = Wtrack;
	  rechitsasexpectedwidth = Wexp;

	  clusterWidth = clusiz;
	  unsigned int iopt;
	  if (clusterWidth > Wexp + 2) {
	    iopt = 1;
	  } else if (Wexp == 1) {
	    iopt = 2;
	  } else if (clusterWidth <= Wexp) {
	    iopt = 3;
	  } else {
	    iopt = 4;
	  }
	  rechitsascategory = iopt;
	}
	//isrechitsas = 0;

      }

      //Filling Histograms for simple hits
      //cout<<"isrechitrphi,isrechitsas = "<<isrechitrphi<<","<<isrechitsas<<endl;

      float CutThickness=0.04;
      CutThickness=0.;

      if(isrechitrphi>0 || isrechitsas>0){

	
	if (isrechitrphi>0) {
	  
	  //cout<<"rechitrphitrackwidth,rechitrphipullMF = "<<rechitrphitrackwidth<<" "<<rechitrphipullMF<<endl;
	  /*
	  if (rechitrphithickness > CutThickness)
	    {
	      	      PullvsTrackwidth->Fill(rechitrphitrackwidth,rechitrphipullMF);
		      
	      if (clusizrphi ==2 && rechitrphiexpectedwidth == 1 && rechitrphitrackwidth<0.1) {
		//Diff->Fill(-diff);

		//		if ((detid.subdetId() == int(StripSubdetector::TID)) || (detid.subdetId() == int(StripSubdetector::TEC))) {
		//SecondStrip->Fill(secondstrip);
		//		}
	      }
	      //	      Diff->Fill(-diff);
	      
	      //PositionSHx->Fill(positionshx);

	      //    ErrxMF->Fill(sqrt(rechitrphierrxMF));
	      //cout<<"ICI1:rechitrphitrackwidth = "<<rechitrphitrackwidth<<endl;
	      //ErrxMFvsTrackwidth->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      ResMFvsTrackwidth->Fill(rechitrphitrackwidth,rechitrphiresMF);
	      
	      PullvsClusterwidth->Fill(clusizrphi,rechitrphipullMF);
	      PullvsExpectedwidth->Fill(rechitrphiexpectedwidth,rechitrphipullMF);
	      PullvsTrackangle->Fill(rechitrphitrackangle,rechitrphipullMF);
	      PullvsTrackanglebeta->Fill(rechitrphitrackanglebeta,rechitrphipullMF);
	     
	    }
	  */
	  
	  meCategory->Fill(rechitrphicategory);
	  meTrackwidth->Fill(rechitrphitrackwidth);
	  meExpectedwidth->Fill(rechitrphiexpectedwidth);
	  meClusterwidth->Fill(clusizrphi);
	  meTrackanglealpha->Fill(rechitrphitrackangle);
	  meTrackanglebeta->Fill(rechitrphitrackanglebeta);

	  meErrxMFAngleProfile->Fill(rechitrphitrackangle,sqrt(rechitrphierrxMF));
	  meErrxMFTrackwidthProfile->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));

	  if (clusizrphi == 1) {
	    meErrxMFTrackwidthProfileWClus1->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meResMFTrackwidthProfileWClus1->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    if (rechitrphithickness > CutThickness)
	      {
		//if ((detid.subdetId() == int(StripSubdetector::TIB)) || (detid.subdetId() == int(StripSubdetector::TOB)))
		//{
		/*
		ResMFvsTrackwidthWClus1->Fill(rechitrphitrackwidth,rechitrphiresMF);
		    if (rechitrphiexpectedwidth==1) ResMFvsTrackwidthWClus1Wexp1->Fill(rechitrphitrackwidth,rechitrphiresMF);
		    if (rechitrphiexpectedwidth==2) ResMFvsTrackwidthWClus1Wexp2->Fill(rechitrphitrackwidth,rechitrphiresMF);
		    if (rechitrphiexpectedwidth==3) ResMFvsTrackwidthWClus1Wexp3->Fill(rechitrphitrackwidth,rechitrphiresMF);
		    if (rechitrphiexpectedwidth==4) ResMFvsTrackwidthWClus1Wexp4->Fill(rechitrphitrackwidth,rechitrphiresMF);
		    ErrxMFvsTrackwidthWClus1->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
		*/
		    //}
	      }
	  }
	  if (clusizrphi == 2) {
	    meErrxMFTrackwidthProfileWClus2->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meResMFTrackwidthProfileWClus2->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    meResMFTrackwidthProfileWClus21->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    meResMFTrackwidthProfileWClus22->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    meResMFTrackwidthProfileWClus23->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    if (rechitrphithickness > CutThickness)
	      {
		//		if ((detid.subdetId() == int(StripSubdetector::TIB)) || (detid.subdetId() == int(StripSubdetector::TOB)))
		//{
		if ((detid.subdetId() == int(StripSubdetector::TID)) || (detid.subdetId() == int(StripSubdetector::TEC))){
		  /*	ResMFvsTrackwidthWClus2->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==1) ResMFvsTrackwidthWClus2Wexp1->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==2) ResMFvsTrackwidthWClus2Wexp2->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==3) ResMFvsTrackwidthWClus2Wexp3->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==4) ResMFvsTrackwidthWClus2Wexp4->Fill(rechitrphitrackwidth,rechitrphiresMF);*/
		}
		//	    meResMFTrackwidthProfileWClus22->Fill(rechitrphitrackwidth,rechitrphiresMF);
		//cout<<"ICI2:rechitrphitrackwidth = "<<rechitrphitrackwidth<<endl;

		//ErrxMFvsTrackwidthWClus2->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
		    // }
	      }
	  }
	  if (clusizrphi == 3) {
	    meErrxMFTrackwidthProfileWClus3->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meResMFTrackwidthProfileWClus3->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    if (rechitrphithickness > CutThickness)
	      {
		//if ((detid.subdetId() == int(StripSubdetector::TIB)) || (detid.subdetId() == int(StripSubdetector::TOB)))
		//{
		/*
		ResMFvsTrackwidthWClus3->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==1) ResMFvsTrackwidthWClus3Wexp1->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==2) ResMFvsTrackwidthWClus3Wexp2->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==3) ResMFvsTrackwidthWClus3Wexp3->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==4) ResMFvsTrackwidthWClus3Wexp4->Fill(rechitrphitrackwidth,rechitrphiresMF);
		ErrxMFvsTrackwidthWClus3->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
		*/
		//  }
	      }
	  }
	  if (clusizrphi == 4) {
	    meErrxMFTrackwidthProfileWClus4->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meResMFTrackwidthProfileWClus4->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    if (rechitrphithickness > CutThickness)
	      {
		//if ((detid.subdetId() == int(StripSubdetector::TIB)) || (detid.subdetId() == int(StripSubdetector::TOB)))
		//{
		/*	ResMFvsTrackwidthWClus4->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==1) ResMFvsTrackwidthWClus4Wexp1->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==2) ResMFvsTrackwidthWClus4Wexp2->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==3) ResMFvsTrackwidthWClus4Wexp3->Fill(rechitrphitrackwidth,rechitrphiresMF);
		if (rechitrphiexpectedwidth==4) ResMFvsTrackwidthWClus4Wexp4->Fill(rechitrphitrackwidth,rechitrphiresMF);
		ErrxMFvsTrackwidthWClus4->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));*/
		    //}
	      }
	  }
	  
	  if (rechitrphicategory == 1) {
	    meErrxMFTrackwidthProfileCategory1->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meErrxMFClusterwidthProfileCategory1->Fill(clusizrphi,sqrt(rechitrphierrxMF));
	  }
	  if (rechitrphicategory == 2) {
	    meErrxMFTrackwidthProfileCategory2->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    //ResMFvsTrackwidthCategory2->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    //  ErrxMFvsTrackwidthCategory2->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	  }
	  if (rechitrphicategory == 3) {
	    meErrxMFTrackwidthProfileCategory3->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    //ResMFvsTrackwidthCategory3->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    //ErrxMFvsTrackwidthCategory3->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	  }
	  if (rechitrphicategory == 4) {
	    meErrxMFTrackwidthProfileCategory4->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    //ResMFvsTrackwidthCategory4->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    //ErrxMFvsTrackwidthCategory4->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	  }
	  //const unsigned int NBINS = meErrxMFTrackwidthProfile->getNbinsX();
	  //cout<<"NBINS2 = "<<NBINS<<endl;
	  
          meErrxMF->Fill(sqrt(rechitrphierrxMF));
	  //const unsigned int NBINS3 = meErrxMF->getNbinsX();
	  //cout<<"NBINS3 = "<<NBINS<<endl;
	  meErrxLF->Fill(sqrt(rechitrphierrxLF));
          meResMF->Fill(rechitrphiresMF);
          meResLF->Fill(rechitrphiresLF);
          mePullMF->Fill(rechitrphipullMF);
          mePullLF->Fill(rechitrphipullLF);
	  
	}

	if (isrechitsas>0) {
	  
	  if (rechitsasthickness > CutThickness)
	    {
	      /*
	      	      PullvsTrackwidth->Fill(rechitsastrackwidth,rechitsaspullMF);
		      //cout<<"rechitsaserrxMF"<<rechitsaserrxMF<<endl;
		      // ErrxMF->Fill(sqrt(rechitsaserrxMF));
	      ErrxMFvsTrackwidth->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	      ResMFvsTrackwidth->Fill(rechitsastrackwidth,rechitsasresMF);

	      
	      PullvsClusterwidth->Fill(clusizsas,rechitsaspullMF);
	      PullvsExpectedwidth->Fill(rechitsasexpectedwidth,rechitsaspullMF);
	      PullvsTrackangle->Fill(rechitsastrackangle,rechitsaspullMF);
	      PullvsTrackanglebeta->Fill(rechitsastrackanglebeta,rechitsaspullMF);
	      */
	    }
	  
	  
	  meCategory->Fill(rechitsascategory);
	  meTrackwidth->Fill(rechitsastrackwidth);
	  meExpectedwidth->Fill(rechitsasexpectedwidth);
	  meClusterwidth->Fill(clusizsas);
	  meTrackanglealpha->Fill(rechitsastrackangle);
	  meTrackanglebeta->Fill(rechitsastrackanglebeta);
	  
	  meErrxMFAngleProfile->Fill(rechitsastrackangle,sqrt(rechitsaserrxMF));
	  meErrxMFTrackwidthProfile->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	  
	  if (clusizsas == 1) {
	    meErrxMFTrackwidthProfileWClus1->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meResMFTrackwidthProfileWClus1->Fill(rechitsastrackwidth,rechitsasresMF);
	    if (rechitsasthickness > CutThickness)
	      {
		//if ((detid.subdetId() == int(StripSubdetector::TIB)) || (detid.subdetId() == int(StripSubdetector::TOB)))
		//{
		/*  
		ResMFvsTrackwidthWClus1->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==1) ResMFvsTrackwidthWClus1Wexp1->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==2) ResMFvsTrackwidthWClus1Wexp2->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==3) ResMFvsTrackwidthWClus1Wexp3->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==4) ResMFvsTrackwidthWClus1Wexp4->Fill(rechitsastrackwidth,rechitsasresMF);
		ErrxMFvsTrackwidthWClus1->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
		*/
		       //}
	      }
	  }
	  
	  if (clusizsas == 2) {
	    meErrxMFTrackwidthProfileWClus2->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meResMFTrackwidthProfileWClus2->Fill(rechitsastrackwidth,rechitsasresMF);
	    if (rechitsasthickness > CutThickness)
	      {
		//		if ((detid.subdetId() == int(StripSubdetector::TIB)) || (detid.subdetId() == int(StripSubdetector::TOB)))
		//{
		/*
		ResMFvsTrackwidthWClus2->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==1) ResMFvsTrackwidthWClus2Wexp1->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==2) ResMFvsTrackwidthWClus2Wexp2->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==3) ResMFvsTrackwidthWClus2Wexp3->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==4) ResMFvsTrackwidthWClus2Wexp4->Fill(rechitsastrackwidth,rechitsasresMF);
		ErrxMFvsTrackwidthWClus2->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
		*/
		    //}
	      }
	  }
	  if (clusizsas == 3) {
	    meErrxMFTrackwidthProfileWClus3->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meResMFTrackwidthProfileWClus3->Fill(rechitsastrackwidth,rechitsasresMF);
	    if (rechitsasthickness > CutThickness)
	      {
		//if ((detid.subdetId() == int(StripSubdetector::TIB)) || (detid.subdetId() == int(StripSubdetector::TOB)))
		// {
		/*
		ResMFvsTrackwidthWClus3->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==1) ResMFvsTrackwidthWClus3Wexp1->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==2) ResMFvsTrackwidthWClus3Wexp2->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==3) ResMFvsTrackwidthWClus3Wexp3->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==4) ResMFvsTrackwidthWClus3Wexp4->Fill(rechitsastrackwidth,rechitsasresMF);
		ErrxMFvsTrackwidthWClus3->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
		*/
		//}
	      }
	  }
	  if (clusizsas == 4) {
	    meErrxMFTrackwidthProfileWClus4->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meResMFTrackwidthProfileWClus4->Fill(rechitsastrackwidth,rechitsasresMF);
	    if (rechitsasthickness > CutThickness)
	      {
		//if ((detid.subdetId() == int(StripSubdetector::TIB)) || (detid.subdetId() == int(StripSubdetector::TOB)))
		//{
		/*
		ResMFvsTrackwidthWClus4->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==1) ResMFvsTrackwidthWClus4Wexp1->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==2) ResMFvsTrackwidthWClus4Wexp2->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==3) ResMFvsTrackwidthWClus4Wexp3->Fill(rechitsastrackwidth,rechitsasresMF);
		if (rechitsasexpectedwidth==4) ResMFvsTrackwidthWClus4Wexp4->Fill(rechitsastrackwidth,rechitsasresMF);
		ErrxMFvsTrackwidthWClus4->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
		*/
		    // }
	      }
	  }
	  if (rechitsascategory == 1) {
	    meErrxMFTrackwidthProfileCategory1->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meErrxMFClusterwidthProfileCategory1->Fill(clusizsas,sqrt(rechitsaserrxMF));
	  }
	  if (rechitsascategory == 2) {
	    meErrxMFTrackwidthProfileCategory2->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    //ResMFvsTrackwidthCategory2->Fill(rechitsastrackwidth,rechitsasresMF);
	    //ErrxMFvsTrackwidthCategory2->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	  }
	  if (rechitsascategory == 3) {
	    meErrxMFTrackwidthProfileCategory3->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    //ResMFvsTrackwidthCategory3->Fill(rechitsastrackwidth,rechitsasresMF);
	    //ErrxMFvsTrackwidthCategory3->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	  }
	  if (rechitsascategory == 4) {
	    meErrxMFTrackwidthProfileCategory4->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    //ResMFvsTrackwidthCategory4->Fill(rechitsastrackwidth,rechitsasresMF);
	    //ErrxMFvsTrackwidthCategory4->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	  }
	  
          meErrxMF->Fill(sqrt(rechitsaserrxMF));
          meErrxLF->Fill(sqrt(rechitsaserrxLF));
          meResMF->Fill(rechitsasresMF);
          meResLF->Fill(rechitsasresLF);
          mePullMF->Fill(rechitsaspullMF);
          mePullLF->Fill(rechitsaspullLF);
	   
	}

	
	if (detid.subdetId() == int(StripSubdetector::TIB)){
	  
	  int Tibisrechitrphi    = isrechitrphi;
	  int Tibisrechitsas     = isrechitsas;
	  //cout<<"Tibisrechitrphi,Tibisrechitsas = "<<Tibisrechitrphi<<" "<<Tibisrechitsas<<endl;
	  int ilay = tTopo->tibLayer(myid) - 1; //for histogram filling
	  //cout<<"ilay1 = "<<ilay<<endl;
	  if(Tibisrechitrphi!=0){
	    if (rechitrphithickness > CutThickness)
	      {
		/*PullvsTrackwidthTIB->Fill(rechitrphitrackwidth,rechitrphipullMF);
		PullvsClusterwidthTIB->Fill(clusizrphi,rechitrphipullMF);
		PullvsExpectedwidthTIB->Fill(rechitrphiexpectedwidth,rechitrphipullMF);
		PullvsTrackangleTIB->Fill(rechitrphitrackangle,rechitrphipullMF);
		PullvsTrackanglebetaTIB->Fill(rechitrphitrackanglebeta,rechitrphipullMF);*/
	      }
	    //cout<<"TIB:rechitrphitrackwidth,rechitrphipullMF = "<<rechitrphitrackwidth<<" "<<rechitrphipullMF<<endl;
	    //cout<<"ilay2 = "<<ilay<<endl;
	    //cout<<"je suis la RPHI"<<endl;
	    meNstpRphiTIB[ilay]->Fill(clusizrphi);
	    meAdcRphiTIB[ilay]->Fill(cluchgrphi);
	    mePosxRphiTIB[ilay]->Fill(rechitrphix);
	    meErrxLFRphiTIB[ilay]->Fill(sqrt(rechitrphierrxLF));
	    meErrxMFRphiTIB[ilay]->Fill(sqrt(rechitrphierrxMF));
	    meResLFRphiTIB[ilay]->Fill(rechitrphiresLF);
	    meResMFRphiTIB[ilay]->Fill(rechitrphiresMF);
	    mePullLFRphiTIB[ilay]->Fill(rechitrphipullLF);
	    mePullMFRphiTIB[ilay]->Fill(rechitrphipullMF);
	    meTrackangleRphiTIB[ilay]->Fill(rechitrphitrackangle);
	    mePullTrackangleProfileRphiTIB[ilay]->Fill(rechitrphitrackangle,rechitrphipullMF);
	    mePullTrackwidthProfileRphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	    if (clusizrphi == 1) {
	      meErrxMFTrackwidthProfileWclus1RphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meResMFTrackwidthProfileWclus1RphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    }
	    if (clusizrphi == 2) {
	      meErrxMFTrackwidthProfileWclus2RphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meResMFTrackwidthProfileWclus2RphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    }
	    if (clusizrphi == 3) {
	      meErrxMFTrackwidthProfileWclus3RphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meResMFTrackwidthProfileWclus3RphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    }
	    if (clusizrphi == 4) {
	      meErrxMFTrackwidthProfileWclus4RphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meResMFTrackwidthProfileWclus4RphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    }


	    if (rechitrphicategory == 1) {
	      mePullTrackwidthProfileCategory1RphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory1RphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meErrxMFClusterwidthProfileCategory1RphiTIB[ilay]->Fill(clusizrphi,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 2) {
	      mePullTrackwidthProfileCategory2RphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory2RphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 3) {
	      mePullTrackwidthProfileCategory3RphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory3RphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 4) {
	      mePullTrackwidthProfileCategory4RphiTIB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory4RphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    meTrackwidthRphiTIB[ilay]->Fill(rechitrphitrackwidth);
	    meExpectedwidthRphiTIB[ilay]->Fill(rechitrphiexpectedwidth);
	    meClusterwidthRphiTIB[ilay]->Fill(clusizrphi);
	    meCategoryRphiTIB[ilay]->Fill(rechitrphicategory);
	    meErrxMFTrackwidthProfileRphiTIB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meErrxMFAngleProfileRphiTIB[ilay]->Fill(rechitrphitrackangle,sqrt(rechitrphierrxMF));
	  }
	  if(Tibisrechitsas!=0){
	    if (rechitsasthickness > CutThickness)
	      {
		/*	PullvsTrackwidthTIB->Fill(rechitsastrackwidth,rechitsaspullMF);
		PullvsClusterwidthTIB->Fill(clusizsas,rechitsaspullMF);
		PullvsExpectedwidthTIB->Fill(rechitsasexpectedwidth,rechitsaspullMF);
		PullvsTrackangleTIB->Fill(rechitsastrackangle,rechitsaspullMF);
		PullvsTrackanglebetaTIB->Fill(rechitsastrackanglebeta,rechitsaspullMF);*/
	      }
	    meNstpSasTIB[ilay]->Fill(clusizsas);
	    meAdcSasTIB[ilay]->Fill(cluchgsas);
	    mePosxSasTIB[ilay]->Fill(rechitsasx);
	    meErrxLFSasTIB[ilay]->Fill(sqrt(rechitsaserrxLF));
	    meResLFSasTIB[ilay]->Fill(rechitsasresLF);
	    mePullLFSasTIB[ilay]->Fill(rechitsaspullLF);
	    meErrxMFSasTIB[ilay]->Fill(sqrt(rechitsaserrxMF));
	    meResMFSasTIB[ilay]->Fill(rechitsasresMF);
	    mePullMFSasTIB[ilay]->Fill(rechitsaspullMF);
	    meTrackangleSasTIB[ilay]->Fill(rechitsastrackangle);
	    mePullTrackangleProfileSasTIB[ilay]->Fill(rechitsastrackangle,rechitsaspullMF);
	    mePullTrackwidthProfileSasTIB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	    if (rechitsascategory == 1) {
	      mePullTrackwidthProfileCategory1SasTIB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory1SasTIB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	      meErrxMFClusterwidthProfileCategory1SasTIB[ilay]->Fill(clusizsas,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 2) {
	      mePullTrackwidthProfileCategory2SasTIB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory2SasTIB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 3) {
	      mePullTrackwidthProfileCategory3SasTIB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory3SasTIB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 4) {
	      mePullTrackwidthProfileCategory4SasTIB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory4SasTIB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    meTrackwidthSasTIB[ilay]->Fill(rechitsastrackwidth);
	    meExpectedwidthSasTIB[ilay]->Fill(rechitsasexpectedwidth);
	    meClusterwidthSasTIB[ilay]->Fill(clusizsas);
	    meCategorySasTIB[ilay]->Fill(rechitsascategory);
	    meErrxMFTrackwidthProfileSasTIB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meErrxMFAngleProfileSasTIB[ilay]->Fill(rechitsastrackangle,rechitsaserrxMF);
	  }
	}
	
	if (detid.subdetId() == int(StripSubdetector::TOB)){
	  
	  int Tobisrechitrphi    = isrechitrphi;
	  int Tobisrechitsas     = isrechitsas;
	  int ilay = tTopo->tobLayer(myid) - 1; //for histogram filling
	  if(Tobisrechitrphi!=0){
	    if (rechitrphithickness > CutThickness)
	      {
		/*PullvsTrackwidthTOB->Fill(rechitrphitrackwidth,rechitrphipullMF);
		PullvsClusterwidthTOB->Fill(clusizrphi,rechitrphipullMF);
		PullvsExpectedwidthTOB->Fill(rechitrphiexpectedwidth,rechitrphipullMF);
		PullvsTrackangleTOB->Fill(rechitrphitrackangle,rechitrphipullMF);
		PullvsTrackanglebetaTOB->Fill(rechitrphitrackanglebeta,rechitrphipullMF);*/
	      }
	    //cout<<"TOB:rechitrphitrackwidth,rechitrphipullMF = "<<rechitrphitrackwidth<<" "<<rechitrphipullMF<<endl;
	    meNstpRphiTOB[ilay]->Fill(clusizrphi);
	    meAdcRphiTOB[ilay]->Fill(cluchgrphi);
	    mePosxRphiTOB[ilay]->Fill(rechitrphix);
	    meErrxLFRphiTOB[ilay]->Fill(sqrt(rechitrphierrxLF));
	    meResLFRphiTOB[ilay]->Fill(rechitrphiresLF);
	    mePullLFRphiTOB[ilay]->Fill(rechitrphipullLF);
	    meErrxMFRphiTOB[ilay]->Fill(sqrt(rechitrphierrxMF));
	    meResMFRphiTOB[ilay]->Fill(rechitrphiresMF);
	    mePullMFRphiTOB[ilay]->Fill(rechitrphipullMF);
	    meTrackangleRphiTOB[ilay]->Fill(rechitrphitrackangle);
	    mePullTrackangleProfileRphiTOB[ilay]->Fill(rechitrphitrackangle,rechitrphipullMF);
	    mePullTrackwidthProfileRphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);

	    if (clusizrphi == 1) {
	      meErrxMFTrackwidthProfileWclus1RphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meResMFTrackwidthProfileWclus1RphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    }
	    if (clusizrphi == 2) {
	      meErrxMFTrackwidthProfileWclus2RphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meResMFTrackwidthProfileWclus2RphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    }
	    if (clusizrphi == 3) {
	      meErrxMFTrackwidthProfileWclus3RphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meResMFTrackwidthProfileWclus3RphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    }
	    if (clusizrphi == 4) {
	      meErrxMFTrackwidthProfileWclus4RphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meResMFTrackwidthProfileWclus4RphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphiresMF);
	    }


	    if (rechitrphicategory == 1) {
	      mePullTrackwidthProfileCategory1RphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory1RphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meErrxMFClusterwidthProfileCategory1RphiTOB[ilay]->Fill(clusizrphi,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 2) {
	      mePullTrackwidthProfileCategory2RphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory2RphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 3) {
	      mePullTrackwidthProfileCategory3RphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory3RphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 4) {
	      mePullTrackwidthProfileCategory4RphiTOB[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory4RphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    meTrackwidthRphiTOB[ilay]->Fill(rechitrphitrackwidth);
	    meExpectedwidthRphiTOB[ilay]->Fill(rechitrphiexpectedwidth);
	    meClusterwidthRphiTOB[ilay]->Fill(clusizrphi);
	    meCategoryRphiTOB[ilay]->Fill(rechitrphicategory);
	    meErrxMFTrackwidthProfileRphiTOB[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meErrxMFAngleProfileRphiTOB[ilay]->Fill(rechitrphitrackangle,sqrt(rechitrphierrxMF));
	  } 
	  if(Tobisrechitsas!=0){
	    if (rechitsasthickness > CutThickness)
	      {/*
		PullvsTrackwidthTOB->Fill(rechitsastrackwidth,rechitsaspullMF);
		PullvsClusterwidthTOB->Fill(clusizsas,rechitsaspullMF);
		PullvsExpectedwidthTOB->Fill(rechitsasexpectedwidth,rechitsaspullMF);
		PullvsTrackangleTOB->Fill(rechitsastrackangle,rechitsaspullMF);
		PullvsTrackanglebetaTOB->Fill(rechitsastrackanglebeta,rechitsaspullMF);
	       */
	      }
	    meNstpSasTOB[ilay]->Fill(clusizsas);
	    meAdcSasTOB[ilay]->Fill(cluchgsas);
	    mePosxSasTOB[ilay]->Fill(rechitsasx);
	    meErrxLFSasTOB[ilay]->Fill(sqrt(rechitsaserrxLF));
	    meResLFSasTOB[ilay]->Fill(rechitsasresLF);
	    mePullLFSasTOB[ilay]->Fill(rechitsaspullLF);
	    meErrxMFSasTOB[ilay]->Fill(sqrt(rechitsaserrxMF));
	    meResMFSasTOB[ilay]->Fill(rechitsasresMF);
	    mePullMFSasTOB[ilay]->Fill(rechitsaspullMF);
	    meTrackangleSasTOB[ilay]->Fill(rechitsastrackangle);
	    mePullTrackangleProfileSasTOB[ilay]->Fill(rechitsastrackangle,rechitsaspullMF);
	    mePullTrackwidthProfileSasTOB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	    if (rechitsascategory == 1) {
	      mePullTrackwidthProfileCategory1SasTOB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory1SasTOB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	      meErrxMFClusterwidthProfileCategory1SasTOB[ilay]->Fill(clusizsas,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 2) {
	      mePullTrackwidthProfileCategory2SasTOB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory2SasTOB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 3) {
	      mePullTrackwidthProfileCategory3SasTOB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory3SasTOB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 4) {
	      mePullTrackwidthProfileCategory4SasTOB[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory4SasTOB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    meTrackwidthSasTOB[ilay]->Fill(rechitsastrackwidth);
	    meExpectedwidthSasTOB[ilay]->Fill(rechitsasexpectedwidth);
	    meClusterwidthSasTOB[ilay]->Fill(clusizsas);
	    meCategorySasTOB[ilay]->Fill(rechitsascategory);
	    meErrxMFTrackwidthProfileSasTOB[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meErrxMFAngleProfileSasTOB[ilay]->Fill(rechitsastrackangle,rechitsaserrxMF);
	  }
	}
	
	if (detid.subdetId() == int(StripSubdetector::TID)){
	  
	  int Tidisrechitrphi    = isrechitrphi;
	  int Tidisrechitsas     = isrechitsas;
	  int ilay = tTopo->tidRing(myid) - 1; //for histogram filling
	  if(Tidisrechitrphi!=0){
	    if (rechitrphithickness > CutThickness)
	      {
		/*PullvsTrackwidthTID->Fill(rechitrphitrackwidth,rechitrphipullMF);
		PullvsClusterwidthTID->Fill(clusizrphi,rechitrphipullMF);
		PullvsExpectedwidthTID->Fill(rechitrphiexpectedwidth,rechitrphipullMF);
		PullvsTrackangleTID->Fill(rechitrphitrackangle,rechitrphipullMF);
		PullvsTrackanglebetaTID->Fill(rechitrphitrackanglebeta,rechitrphipullMF);*/
	      }
	    //cout<<"TID:rechitrphitrackwidth,rechitrphipullMF = "<<rechitrphitrackwidth<<" "<<rechitrphipullMF<<endl;
	    meNstpRphiTID[ilay]->Fill(clusizrphi);
	    meAdcRphiTID[ilay]->Fill(cluchgrphi);
	    mePosxRphiTID[ilay]->Fill(rechitrphix);
	    meErrxLFRphiTID[ilay]->Fill(sqrt(rechitrphierrxLF));
	    meResLFRphiTID[ilay]->Fill(rechitrphiresLF);
	    mePullLFRphiTID[ilay]->Fill(rechitrphipullLF);
	    meErrxMFRphiTID[ilay]->Fill(sqrt(rechitrphierrxMF));
	    meResMFRphiTID[ilay]->Fill(rechitrphiresMF);
	    mePullMFRphiTID[ilay]->Fill(rechitrphipullMF);
	    meTrackangleRphiTID[ilay]->Fill(rechitrphitrackangle);
	    mePullTrackangleProfileRphiTID[ilay]->Fill(rechitrphitrackangle,rechitrphipullMF);
	    mePullTrackwidthProfileRphiTID[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	    if (rechitrphicategory == 1) {
	      mePullTrackwidthProfileCategory1RphiTID[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory1RphiTID[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meErrxMFClusterwidthProfileCategory1RphiTID[ilay]->Fill(clusizrphi,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 2) {
	      mePullTrackwidthProfileCategory2RphiTID[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory2RphiTID[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 3) {
	      mePullTrackwidthProfileCategory3RphiTID[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory3RphiTID[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 4) {
	      mePullTrackwidthProfileCategory4RphiTID[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory4RphiTID[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    meTrackwidthRphiTID[ilay]->Fill(rechitrphitrackwidth);
	    meExpectedwidthRphiTID[ilay]->Fill(rechitrphiexpectedwidth);
	    meClusterwidthRphiTID[ilay]->Fill(clusizrphi);
	    meCategoryRphiTID[ilay]->Fill(rechitrphicategory);
	    meErrxMFTrackwidthProfileRphiTID[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meErrxMFAngleProfileRphiTID[ilay]->Fill(rechitrphitrackangle,sqrt(rechitrphierrxMF));
	  } 
	  if(Tidisrechitsas!=0){
	    if (rechitsasthickness > CutThickness)
	      {
		/*PullvsTrackwidthTID->Fill(rechitsastrackwidth,rechitsaspullMF);
		PullvsClusterwidthTID->Fill(clusizsas,rechitsaspullMF);
		PullvsExpectedwidthTID->Fill(rechitsasexpectedwidth,rechitsaspullMF);
		PullvsTrackangleTID->Fill(rechitsastrackangle,rechitsaspullMF);
		PullvsTrackanglebetaTID->Fill(rechitsastrackanglebeta,rechitsaspullMF);*/
	      }
	    meNstpSasTID[ilay]->Fill(clusizsas);
	    meAdcSasTID[ilay]->Fill(cluchgsas);
	    mePosxSasTID[ilay]->Fill(rechitsasx);
	    meErrxLFSasTID[ilay]->Fill(sqrt(rechitsaserrxLF));
	    meResLFSasTID[ilay]->Fill(rechitsasresLF);
	    mePullLFSasTID[ilay]->Fill(rechitsaspullLF);
	    meErrxMFSasTID[ilay]->Fill(sqrt(rechitsaserrxMF));
	    meResMFSasTID[ilay]->Fill(rechitsasresMF);
	    mePullMFSasTID[ilay]->Fill(rechitsaspullMF);
	    meTrackangleSasTID[ilay]->Fill(rechitsastrackangle);
	    mePullTrackangleProfileSasTID[ilay]->Fill(rechitsastrackangle,rechitsaspullMF);
	    mePullTrackwidthProfileSasTID[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	    if (rechitsascategory == 1) {
	      mePullTrackwidthProfileCategory1SasTID[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory1SasTID[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	      meErrxMFClusterwidthProfileCategory1SasTID[ilay]->Fill(clusizsas,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 2) {
	      mePullTrackwidthProfileCategory2SasTID[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory2SasTID[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 3) {
	      mePullTrackwidthProfileCategory3SasTID[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory3SasTID[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 4) {
	      mePullTrackwidthProfileCategory4SasTID[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory4SasTID[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    meTrackwidthSasTID[ilay]->Fill(rechitsastrackwidth);
	    meExpectedwidthSasTID[ilay]->Fill(rechitsasexpectedwidth);
	    meClusterwidthSasTID[ilay]->Fill(clusizsas);
	    meCategorySasTID[ilay]->Fill(rechitsascategory);
	    meErrxMFTrackwidthProfileSasTID[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meErrxMFAngleProfileSasTID[ilay]->Fill(rechitsastrackangle,rechitsaserrxMF);
	  }
	}
	      
	if (detid.subdetId() == int(StripSubdetector::TEC)){
	  
	  int Tecisrechitrphi    = isrechitrphi;
	  int Tecisrechitsas     = isrechitsas;
	  int ilay = tTopo->tecRing(myid) - 1; //for histogram filling
	  if(Tecisrechitrphi!=0){
	    if (rechitrphithickness > CutThickness)
	      {
		/*PullvsTrackwidthTEC->Fill(rechitrphitrackwidth,rechitrphipullMF);
		PullvsClusterwidthTEC->Fill(clusizrphi,rechitrphipullMF);
		PullvsExpectedwidthTEC->Fill(rechitrphiexpectedwidth,rechitrphipullMF);
		PullvsTrackangleTEC->Fill(rechitrphitrackangle,rechitrphipullMF);
		PullvsTrackanglebetaTEC->Fill(rechitrphitrackanglebeta,rechitrphipullMF);*/
	      }
	    //cout<<"TEC:rechitrphitrackwidth,rechitrphipullMF = "<<rechitrphitrackwidth<<" "<<rechitrphipullMF<<endl;
	    meNstpRphiTEC[ilay]->Fill(clusizrphi);
	    meAdcRphiTEC[ilay]->Fill(cluchgrphi);
	    mePosxRphiTEC[ilay]->Fill(rechitrphix);
	    meErrxLFRphiTEC[ilay]->Fill(sqrt(rechitrphierrxLF));
	    meResLFRphiTEC[ilay]->Fill(rechitrphiresLF);
	    mePullLFRphiTEC[ilay]->Fill(rechitrphipullLF);
	    meErrxMFRphiTEC[ilay]->Fill(sqrt(rechitrphierrxMF));
	    meResMFRphiTEC[ilay]->Fill(rechitrphiresMF);
	    mePullMFRphiTEC[ilay]->Fill(rechitrphipullMF);
	    meTrackangleRphiTEC[ilay]->Fill(rechitrphitrackangle);
	    mePullTrackangleProfileRphiTEC[ilay]->Fill(rechitrphitrackangle,rechitrphipullMF);
	    mePullTrackwidthProfileRphiTEC[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	    if (rechitrphicategory == 1) {
	      mePullTrackwidthProfileCategory1RphiTEC[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory1RphiTEC[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	      meErrxMFClusterwidthProfileCategory1RphiTEC[ilay]->Fill(clusizrphi,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 2) {
	      mePullTrackwidthProfileCategory2RphiTEC[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory2RphiTEC[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 3) {
	      mePullTrackwidthProfileCategory3RphiTEC[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory3RphiTEC[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    if (rechitrphicategory == 4) {
	      mePullTrackwidthProfileCategory4RphiTEC[ilay]->Fill(rechitrphitrackwidth,rechitrphipullMF);
	      meErrxMFTrackwidthProfileCategory4RphiTEC[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    }
	    meTrackwidthRphiTEC[ilay]->Fill(rechitrphitrackwidth);
	    meExpectedwidthRphiTEC[ilay]->Fill(rechitrphiexpectedwidth);
	    meClusterwidthRphiTEC[ilay]->Fill(clusizrphi);
	    meCategoryRphiTEC[ilay]->Fill(rechitrphicategory);
	    meErrxMFTrackwidthProfileRphiTEC[ilay]->Fill(rechitrphitrackwidth,sqrt(rechitrphierrxMF));
	    meErrxMFAngleProfileRphiTEC[ilay]->Fill(rechitrphitrackangle,sqrt(rechitrphierrxMF));
	  } 
	  if(Tecisrechitsas!=0){
	    if (rechitsasthickness > CutThickness)
	      {
		/*PullvsTrackwidthTEC->Fill(rechitsastrackwidth,rechitsaspullMF);
		PullvsClusterwidthTEC->Fill(clusizsas,rechitsaspullMF);
		PullvsExpectedwidthTEC->Fill(rechitsasexpectedwidth,rechitsaspullMF);
		PullvsTrackangleTEC->Fill(rechitsastrackangle,rechitsaspullMF);
		PullvsTrackanglebetaTEC->Fill(rechitsastrackanglebeta,rechitsaspullMF);*/
	      }
	    meNstpSasTEC[ilay]->Fill(clusizsas);
	    meAdcSasTEC[ilay]->Fill(cluchgsas);
	    mePosxSasTEC[ilay]->Fill(rechitsasx);
	    meErrxLFSasTEC[ilay]->Fill(sqrt(rechitsaserrxLF));
	    meResLFSasTEC[ilay]->Fill(rechitsasresLF);
	    mePullLFSasTEC[ilay]->Fill(rechitsaspullLF);
	    meErrxMFSasTEC[ilay]->Fill(sqrt(rechitsaserrxMF));
	    meResMFSasTEC[ilay]->Fill(rechitsasresMF);
	    mePullMFSasTEC[ilay]->Fill(rechitsaspullMF);
	    meTrackangleSasTEC[ilay]->Fill(rechitsastrackangle);
	    mePullTrackangleProfileSasTEC[ilay]->Fill(rechitsastrackangle,rechitsaspullMF);
	    mePullTrackwidthProfileSasTEC[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	    if (rechitsascategory == 1) {
	      mePullTrackwidthProfileCategory1SasTEC[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory1SasTEC[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	      meErrxMFClusterwidthProfileCategory1SasTEC[ilay]->Fill(clusizsas,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 2) {
	      mePullTrackwidthProfileCategory2SasTEC[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory2SasTEC[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 3) {
	      mePullTrackwidthProfileCategory3SasTEC[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory3SasTEC[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    if (rechitsascategory == 4) {
	      mePullTrackwidthProfileCategory4SasTEC[ilay]->Fill(rechitsastrackwidth,rechitsaspullMF);
	      meErrxMFTrackwidthProfileCategory4SasTEC[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    }
	    meTrackwidthSasTEC[ilay]->Fill(rechitsastrackwidth);
	    meExpectedwidthSasTEC[ilay]->Fill(rechitsasexpectedwidth);
	    meClusterwidthSasTEC[ilay]->Fill(clusizsas);
	    meCategorySasTEC[ilay]->Fill(rechitsascategory);
	    meErrxMFTrackwidthProfileSasTEC[ilay]->Fill(rechitsastrackwidth,sqrt(rechitsaserrxMF));
	    meErrxMFAngleProfileSasTEC[ilay]->Fill(rechitsastrackangle,rechitsaserrxMF);
	  }
	
	}
	
      } //simplehits
      //cout<<"DebugLine301"<<endl;
      
    }
    //cout<<"DebugLine302"<<endl;
    
  }
  //cout<<"DebugLine303"<<endl;

}



  //needed by to do the residual for matched hits
std::pair<LocalPoint,LocalVector> SiStripTrackingRecHitsValid::projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet, const BoundPlane& plane) 
{
  //  const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(hit.det());
  //if (stripDet == 0) throw MeasurementDetException("HitMatcher hit is not on StripGeomDetUnit");
  
  const StripTopology& topol = stripDet->specificTopology();
  GlobalPoint globalpos= stripDet->surface().toGlobal(hit.localPosition());
  LocalPoint localHit = plane.toLocal(globalpos);
  //track direction
  LocalVector locdir=hit.localDirection();
  //rotate track in new frame
  
  GlobalVector globaldir= stripDet->surface().toGlobal(locdir);
  LocalVector dir=plane.toLocal(globaldir);
  float scale = -localHit.z() / dir.z();
  
  LocalPoint projectedPos = localHit + scale*dir;
  
  //  std::cout << "projectedPos " << projectedPos << std::endl;
  
  float selfAngle = topol.stripAngle( topol.strip( hit.localPosition()));
  
  LocalVector stripDir( sin(selfAngle), cos(selfAngle), 0); // vector along strip in hit frame
  
  LocalVector localStripDir( plane.toLocal(stripDet->surface().toGlobal( stripDir)));
  
  return std::pair<LocalPoint,LocalVector>( projectedPos, localStripDir);
}
