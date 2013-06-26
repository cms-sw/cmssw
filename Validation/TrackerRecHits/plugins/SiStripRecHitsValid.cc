// File: SiStripRecHitsValid.cc
// Description:  see SiStripRecHitsValid.h
// Author:  P. Azzi
// Creation Date:  PA May 2006 Initial version.
//
//--------------------------------------------
#include "Validation/TrackerRecHits/interface/SiStripRecHitsValid.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h" 

//needed for the geometry: 
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"


//--- for RecHit
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h" 
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h" 
#include "DataFormats/Common/interface/OwnVector.h" 
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

namespace helper { 
    struct GetDetId { 
        template<typename X> 
        DetId operator()(const X &x) { return DetId(x.detId()); }
    };

    template<typename T>
    std::pair<typename T::DetSet::const_iterator, typename T::DetSet::const_iterator> 
    getRange(const T &detset, const DetId &id) {
        typedef std::pair<typename T::DetSet::const_iterator, typename T::DetSet::const_iterator> return_type;
        typename T::const_iterator match = detset.find(id);
        if (match == detset.end()) return return_type();
        typename T::DetSet hits = *match;
        return return_type(hits.begin(), hits.end());
    } 
}



//Constructor
SiStripRecHitsValid::SiStripRecHitsValid(const ParameterSet& ps) :
  dbe_(0),	
  conf_(ps),
  matchedRecHits_( ps.getParameter<edm::InputTag>("matchedRecHits") ),
  rphiRecHits_( ps.getParameter<edm::InputTag>("rphiRecHits") ),
  stereoRecHits_( ps.getParameter<edm::InputTag>("stereoRecHits") ) {

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "sistriprechitshisto.root");
  dbe_ = Service<DQMStore>().operator->();
  //dbe_->showDirStructure();
  dbe_->setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Strip/SISTRIP");

  meNumTotRphi = dbe_->book1D("NumTotRphi","Num of RecHits rphi",100, 0, 10000);
  meNumTotSas = dbe_->book1D("NumTotSas","Num of RecHits sas",100, 0, 10000);
  meNumTotMatched = dbe_->book1D("NumTotMatched","Num of RecHits rmatched",100, 0, 10000);

  dbe_->setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Strip/TIB");
  meNumRphiTIB = dbe_->book1D("NumRphiTIB","Num of RecHits rphi", 100, 0, 1000.);
  meNumSasTIB = dbe_->book1D("NumSasTIB","Num of RecHits sas", 100, 0, 1000.);
  meNumMatchedTIB = dbe_->book1D("NumMatchedTIB","Num of RecHits matched", 100, 0, 1000.);

  //one histo per Layer rphi hits
  for(int i = 0 ;i<4 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtib",i+1);
    meNstpRphiTIB[i] = dbe_->book1D(histo,"RecHit Cluster Size",10,0.5,10.5);  
    sprintf(histo,"Adc_rphi_layer%dtib",i+1);
    meAdcRphiTIB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtib",i+1);
    mePosxRphiTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%dtib",i+1);
    meErrxRphiTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.01); //<error>~20micron  
    sprintf(histo,"Res_rphi_layer%dtib",i+1);
    meResRphiTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Pull_LF_rphi_layer%dtib",i+1);
    mePullLFRphiTIB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_rphi_layer%dtib",i+1);
    mePullMFRphiTIB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Chi2_rphi_layer%dtib",i+1);
    meChi2RphiTIB[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
  }

  //one histo per Layer stereo and matched hits
  for(int i = 0 ;i<2 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_sas_layer%dtib",i+1);
    meNstpSasTIB[i] = dbe_->book1D(histo,"RecHit Cluster Size",10,0.5,10.5);  
    sprintf(histo,"Adc_sas_layer%dtib",i+1);
    meAdcSasTIB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_sas_layer%dtib",i+1);
    mePosxSasTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_sas_layer%dtib",i+1);
    meErrxSasTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.01);  
    sprintf(histo,"Res_sas_layer%dtib",i+1);
    meResSasTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Pull_LF_sas_layer%dtib",i+1);
    mePullLFSasTIB[i] = dbe_->book1D(histo,"Pull",100,-4.,4.);  
    sprintf(histo,"Pull_MF_sas_layer%dtib",i+1);
    mePullMFSasTIB[i] = dbe_->book1D(histo,"Pull",100,-4.,4.);  
    sprintf(histo,"Chi2_sas_layer%dtib",i+1);
    meChi2SasTIB[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  

    sprintf(histo,"Posx_matched_layer%dtib",i+1);
    mePosxMatchedTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%dtib",i+1);
    mePosyMatchedTIB[i] = dbe_->book1D(histo,"RecHit y coord.",100,-6.0, +6.0);  
    sprintf(histo,"Errx_matched_layer%dtib",i+1);
    meErrxMatchedTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.01);  
    sprintf(histo,"Erry_matched_layer%dtib",i+1);
    meErryMatchedTIB[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.05);  
    sprintf(histo,"Resx_matched_layer%dtib",i+1);
    meResxMatchedTIB[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.02, +0.02);  
    sprintf(histo,"Resy_matched_layer%dtib",i+1);
    meResyMatchedTIB[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
    sprintf(histo,"Chi2_matched_layer%dtib",i+1);
    meChi2MatchedTIB[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
  }

  dbe_->setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Strip/TOB");
  meNumRphiTOB = dbe_->book1D("NumRphiTOB","Num of RecHits rphi", 100, 0, 1000.);
  meNumSasTOB = dbe_->book1D("NumSasTOB","Num of RecHits sas", 100, 0, 1000.);
  meNumMatchedTOB = dbe_->book1D("NumMatchedTOB","Num of RecHits matched", 100, 0, 1000.);
  //one histo per Layer rphi hits
  for(int i = 0 ;i<6 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtob",i+1);
    meNstpRphiTOB[i] = dbe_->book1D(histo,"RecHit Cluster Size",10,0.5,10.5);  
    sprintf(histo,"Adc_rphi_layer%dtob",i+1);
    meAdcRphiTOB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtob",i+1);
    mePosxRphiTOB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%dtob",i+1);
    meErrxRphiTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.01);  
    sprintf(histo,"Res_rphi_layer%dtob",i+1);
    meResRphiTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Pull_LF_rphi_layer%dtob",i+1);
    mePullLFRphiTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_rphi_layer%dtob",i+1);
    mePullMFRphiTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Chi2_rphi_layer%dtob",i+1);
    meChi2RphiTOB[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
  }

  //one histo per Layer stereo and matched hits
  for(int i = 0 ;i<2 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_sas_layer%dtob",i+1);
    meNstpSasTOB[i] = dbe_->book1D(histo,"RecHit Cluster Size",10,0.5,10.5);  
    sprintf(histo,"Adc_sas_layer%dtob",i+1);
    meAdcSasTOB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_sas_layer%dtob",i+1);
    mePosxSasTOB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_sas_layer%dtob",i+1);
    meErrxSasTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.01);  
    sprintf(histo,"Res_sas_layer%dtob",i+1);
    meResSasTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Pull_LF_sas_layer%dtob",i+1);
    mePullLFSasTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_sas_layer%dtob",i+1);
    mePullMFSasTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Chi2_sas_layer%dtob",i+1);
    meChi2SasTOB[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  

    sprintf(histo,"Posx_matched_layer%dtob",i+1);
    mePosxMatchedTOB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%dtob",i+1);
    mePosyMatchedTOB[i] = dbe_->book1D(histo,"RecHit y coord.",100,-10.0, +10.0);  
    sprintf(histo,"Errx_matched_layer%dtob",i+1);
    meErrxMatchedTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.01);  
    sprintf(histo,"Erry_matched_layer%dtob",i+1);
    meErryMatchedTOB[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.1);  
    sprintf(histo,"Resx_matched_layer%dtob",i+1);
    meResxMatchedTOB[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.02, +0.02);  
    sprintf(histo,"Resy_matched_layer%dtob",i+1);
    meResyMatchedTOB[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
    sprintf(histo,"Chi2_matched_layer%dtob",i+1);
    meChi2MatchedTOB[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
  }

  dbe_->setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Strip/TID");
  meNumRphiTID = dbe_->book1D("NumRphiTID","Num of RecHits rphi", 100, 0, 1000.);
  meNumSasTID = dbe_->book1D("NumSasTID","Num of RecHits sas", 100, 0, 1000.);
  meNumMatchedTID = dbe_->book1D("NumMatchedTID","Num of RecHits matched", 100, 0, 1000.);

  //one histo per Ring rphi hits: 3 rings, 6 disks, 2 inner rings are glued 
  for(int i = 0 ;i<3 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtid",i+1);
    meNstpRphiTID[i] = dbe_->book1D(histo,"RecHit Cluster Size",10,0.5,10.5);  
    sprintf(histo,"Adc_rphi_layer%dtid",i+1);
    meAdcRphiTID[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtid",i+1);
    mePosxRphiTID[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%dtid",i+1);
    meErrxRphiTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.5);  
    sprintf(histo,"Res_rphi_layer%dtid",i+1);
    meResRphiTID[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Pull_LF_rphi_layer%dtid",i+1);
    mePullLFRphiTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_rphi_layer%dtid",i+1);
    mePullMFRphiTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Chi2_rphi_layer%dtid",i+1);
    meChi2RphiTID[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
  }

  //one histo per Ring stereo and matched hits
  for(int i = 0 ;i<2 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_sas_layer%dtid",i+1);
    meNstpSasTID[i] = dbe_->book1D(histo,"RecHit Cluster Size",10,0.5,10.5);  
    sprintf(histo,"Adc_sas_layer%dtid",i+1);
    meAdcSasTID[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_sas_layer%dtid",i+1);
    mePosxSasTID[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_sas_layer%dtid",i+1);
    meErrxSasTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.5);  
    sprintf(histo,"Res_sas_layer%dtid",i+1);
    meResSasTID[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Pull_LF_sas_layer%dtid",i+1);
    mePullLFSasTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_sas_layer%dtid",i+1);
    mePullMFSasTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Chi2_sas_layer%dtid",i+1);
    meChi2SasTID[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  

    sprintf(histo,"Posx_matched_layer%dtid",i+1);
    mePosxMatchedTID[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%dtid",i+1);
    mePosyMatchedTID[i] = dbe_->book1D(histo,"RecHit y coord.",100,-6.0, +6.0);  
    sprintf(histo,"Errx_matched_layer%dtid",i+1);
    meErrxMatchedTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.05);  
    sprintf(histo,"Erry_matched_layer%dtid",i+1);
    meErryMatchedTID[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.1);  
    sprintf(histo,"Resx_matched_layer%dtid",i+1);
    meResxMatchedTID[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.2, +0.2);  
    sprintf(histo,"Resy_matched_layer%dtid",i+1);
    meResyMatchedTID[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
    sprintf(histo,"Chi2_matched_layer%dtid",i+1);
    meChi2MatchedTID[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
  }

  dbe_->setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Strip/TEC");
  meNumRphiTEC = dbe_->book1D("NumRphiTEC","Num of RecHits rphi", 100, 0, 1000.);
  meNumSasTEC = dbe_->book1D("NumSasTEC","Num of RecHits sas", 100, 0, 1000.);
  meNumMatchedTEC = dbe_->book1D("NumMatchedTEC","Num of RecHits matched", 100, 0, 1000.);

  //one histo per Ring rphi hits: 7 rings, 18 disks. Innermost 3 rings are same as TID above.  
  for(int i = 0 ;i<7 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtec",i+1);
    meNstpRphiTEC[i] = dbe_->book1D(histo,"RecHit Cluster Size",10,0.5,10.5);  
    sprintf(histo,"Adc_rphi_layer%dtec",i+1);
    meAdcRphiTEC[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtec",i+1);
    mePosxRphiTEC[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%dtec",i+1);
    meErrxRphiTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.5);  
    sprintf(histo,"Res_rphi_layer%dtec",i+1);
    meResRphiTEC[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Pull_LF_rphi_layer%dtec",i+1);
    mePullLFRphiTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Pull_MF_rphi_layer%dtec",i+1);
    mePullMFRphiTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Chi2_rphi_layer%dtec",i+1);
    meChi2RphiTEC[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
  }

  //one histo per Layer stereo and matched hits: rings 1,2,5 are double sided
  for(int i = 0 ;i<5 ; i++) {
    if(i == 0 || i == 1 || i == 4) {
      Char_t histo[200];
      sprintf(histo,"Nstp_sas_layer%dtec",i+1);
      meNstpSasTEC[i] = dbe_->book1D(histo,"RecHit Cluster Size",10,0.5,10.5);  
      sprintf(histo,"Adc_sas_layer%dtec",i+1);
      meAdcSasTEC[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
      sprintf(histo,"Posx_sas_layer%dtec",i+1);
      mePosxSasTEC[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
      sprintf(histo,"Errx_sas_layer%dtec",i+1);
      meErrxSasTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.5);  
      sprintf(histo,"Res_sas_layer%dtec",i+1);
      meResSasTEC[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
      sprintf(histo,"Pull_LF_sas_layer%dtec",i+1);
      mePullLFSasTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);
      sprintf(histo,"Pull_MF_sas_layer%dtec",i+1);
      mePullMFSasTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);
      sprintf(histo,"Chi2_sas_layer%dtec",i+1);
      meChi2SasTEC[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
      
      sprintf(histo,"Posx_matched_layer%dtec",i+1);
      mePosxMatchedTEC[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
      sprintf(histo,"Posy_matched_layer%dtec",i+1);
      mePosyMatchedTEC[i] = dbe_->book1D(histo,"RecHit y coord.",100,-8.0, +8.0);  
      sprintf(histo,"Errx_matched_layer%dtec",i+1);
      meErrxMatchedTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.05);  
      sprintf(histo,"Erry_matched_layer%dtec",i+1);
      meErryMatchedTEC[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.1);  
      sprintf(histo,"Resx_matched_layer%dtec",i+1);
      meResxMatchedTEC[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.2, +0.2);  
      sprintf(histo,"Resy_matched_layer%dtec",i+1);
      meResyMatchedTEC[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
      sprintf(histo,"Chi2_matched_layer%dtec",i+1);
      meChi2MatchedTEC[i] = dbe_->book1D(histo,"RecHit Chi2 test",100,0., 50);  
    }
  }
}


SiStripRecHitsValid::~SiStripRecHitsValid(){
 //if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void SiStripRecHitsValid::beginJob(){

}

void SiStripRecHitsValid::endJob() {
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void SiStripRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);



  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();  
  //cout  << " Run = " << e.id().run() << " Event = " << e.id().event() << endl;  
  
  //--- get RecHits
  
  std::string rechitProducer = "SiStripRecHits2D";
  
  // Step A: Get Inputs 
  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
  edm::Handle<SiStripRecHit2DCollection> rechitsrphi;
  edm::Handle<SiStripRecHit2DCollection> rechitsstereo;
  e.getByLabel(matchedRecHits_, rechitsmatched);
  e.getByLabel(rphiRecHits_, rechitsrphi);
  e.getByLabel(stereoRecHits_, rechitsstereo);

  int numrechitrphi   =0;
  int numrechitsas    =0;
  int numrechitmatched=0;

  int totTibnumrechitrphi=0;
  int totTibnumrechitsas=0;
  int totTibnumrechitmatched=0;
  int totTobnumrechitrphi=0;
  int totTobnumrechitsas=0;
  int totTobnumrechitmatched=0;
  int totTidnumrechitrphi=0;
  int totTidnumrechitsas=0;
  int totTidnumrechitmatched=0;
  int totTecnumrechitrphi=0;
  int totTecnumrechitsas=0;
  int totTecnumrechitmatched=0;
  int totrechitrphi =0;
  int totrechitsas =0;
  int totrechitmatched =0;
  
  TrackerHitAssociator associate(e, conf_);
  
  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry &tracker(*pDD);
  // FIXME: this using of vector<DetId> is suboptimal, but I don't want to re-write the full class now
  std::vector<DetId> IDs; 
  IDs.reserve(rechitsrphi->size() + rechitsmatched->size() + rechitsstereo->size());
  std::transform(rechitsrphi->begin(), rechitsrphi->end(), std::back_inserter(IDs), helper::GetDetId() );
  std::transform(rechitsstereo->begin(), rechitsstereo->end(), std::back_inserter(IDs), helper::GetDetId() );
  std::transform(rechitsmatched->begin(), rechitsmatched->end(), std::back_inserter(IDs), helper::GetDetId() );
  // loop over detunits
  //  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
  for(std::vector<DetId>::const_iterator it = IDs.begin(); it != IDs.end(); ++it ){//loop on rphi detector with hits
    uint32_t myid=((*it).rawId());       
    DetId detid = ((*it));
    
    // initialize here
    for(int i=0; i<MAXHIT; i++){
      rechitrphix[i] =0;
      rechitrphierrx[i] =0;
      rechitrphiy[i] =0;
      rechitrphiz[i] =0;
      rechitsasx[i] =0;
      rechitsaserrx[i] =0;
      rechitsasy[i] =0;
      rechitsasz[i] =0;
      clusizrphi[i] =0;
      clusizsas[i] =0;
      cluchgrphi[i] =0;
      cluchgsas[i] =0;
      rechitrphires[i]=-999.;
      rechitsasres[i]=-999.;
      rechitrphipullMF[i]=-999.;
      rechitsaspullMF[i]=-999.;
      chi2rphi[i] =0;
      chi2sas[i]=0;
      rechitmatchedx[i] =0;
      rechitmatchedy[i] =0;
      rechitmatchedz[i] =0;
      rechitmatchederrxx[i] =0;
      rechitmatchederrxy[i] =0;
      rechitmatchederryy[i] =0;
      rechitmatchedresx[i]=-999;
      rechitmatchedresy[i]=-999;
      chi2matched[i]=0;
    }
    
    numrechitrphi =0;
    //loop over rechits-rphi in the same subdetector
    std::pair<SiStripRecHit2DCollection::DetSet::const_iterator,SiStripRecHit2DCollection::DetSet::const_iterator> rechitrphiRange = helper::getRange(*rechitsrphi, detid);
    SiStripRecHit2DCollection::DetSet::const_iterator rechitrphiRangeIteratorBegin = rechitrphiRange.first;
    SiStripRecHit2DCollection::DetSet::const_iterator rechitrphiRangeIteratorEnd   = rechitrphiRange.second;
    SiStripRecHit2DCollection::DetSet::const_iterator iterrphi=rechitrphiRangeIteratorBegin;
    
    numrechitrphi = rechitrphiRangeIteratorEnd - rechitrphiRangeIteratorBegin;   
         

    int i=0;
    int i2=0;

    if(numrechitrphi > 0 ){
      totrechitrphi+=numrechitrphi;
      for(iterrphi=rechitrphiRangeIteratorBegin; iterrphi!=rechitrphiRangeIteratorEnd;++iterrphi){
	const GeomDetUnit *  det = tracker.idToDetUnit(detid);
	const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
	const StripTopology &topol=(StripTopology&)stripdet->topology();
	SiStripRecHit2D const rechit=*iterrphi;
	LocalPoint position=rechit.localPosition();
	LocalError error=rechit.localPositionError();
	MeasurementPoint Mposition;
	MeasurementError Merror;
	Mposition = topol.measurementPosition(position);
	Merror = topol.measurementError(position,error);
	SiStripRecHit2D::ClusterRef clust=rechit.cluster();
	int clusiz=0;
	int totcharge=0;
	clusiz = clust->amplitudes().size();
	const std::vector<uint8_t> amplitudes=clust->amplitudes();
	  for(size_t ia=0; ia<amplitudes.size();ia++){
	    totcharge+=amplitudes[ia];
	  }
	rechitrphix[i] = position.x();
	rechitrphiy[i] = position.y();
	rechitrphiz[i] = position.z();
	rechitrphierrx[i] = error.xx();
	clusizrphi[i] = clusiz;
	cluchgrphi[i] = totcharge;

	matched.clear();
	matched = associate.associateHit(rechit);
	float mindist = 999999;
	float dist = 999999;
	PSimHit closest;
	if(!matched.empty()){
	  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    dist = fabs(rechitrphix[i] - (*m).localPosition().x());
	    if(dist<mindist){
	      mindist = dist;
	      closest = (*m);
	    }
	    rechitrphires[i] = rechitrphix[i] - closest.localPosition().x();
	  }  
	  rechitrphipullMF[i] = (Mposition.x() - (topol.measurementPosition(closest.localPosition())).x())/sqrt(Merror.uu());

	  //chi2test compare rechit errors with the simhit position ( using null matrix for the simhit). 
	  //Can spot problems in the geometry better than a simple residual. (thanks to BorisM)
	  AlgebraicVector  rhparameters(2);//= rechit.parameters();
	  rhparameters[0] = position.x(); 
	  rhparameters[1] = position.y();
	  AlgebraicVector shparameters(2);
	  shparameters[0] = closest.localPosition().x();
	  shparameters[1] = closest.localPosition().y();
	  AlgebraicVector r(rhparameters - shparameters);
	  AlgebraicSymMatrix R(2);//  = rechit.parametersError();
	  R[0][0] = error.xx();
	  R[0][1] = error.xy();
	  R[1][1] = error.yy();
	  int ierr; 
	  R.invert(ierr); // if (ierr != 0) throw exception;
	  double est = R.similarity(r);
	  // 	  std::cout << " ====== Chi2 test rphi hits ====== " << std::endl;
	  // 	  std::cout << "RecHit param. = " << rhparameters << std::endl;
	  // 	  std::cout << "RecHit errors = " << R << std::endl;
	  //	  std::cout << "SimHit param. = " << shparameters << std::endl;
	  //	  std::cout << " chi2  = " << est << std::endl;
	  //	  std::cout << "DEBUG BORIS,filling chi2rphi[i],i: " << i << std::endl;
	  chi2rphi[i2] = est;
	  i2++;
	}
	i++;
      }
    }
    
    //loop over rechits-sas in the same subdetector
    int j=0;
    int j2=0;
    numrechitsas=0;
    std::pair<SiStripRecHit2DCollection::DetSet::const_iterator,SiStripRecHit2DCollection::DetSet::const_iterator> rechitsasRange = helper::getRange(*rechitsstereo, detid);
    SiStripRecHit2DCollection::DetSet::const_iterator rechitsasRangeIteratorBegin = rechitsasRange.first;
    SiStripRecHit2DCollection::DetSet::const_iterator rechitsasRangeIteratorEnd   = rechitsasRange.second;
    SiStripRecHit2DCollection::DetSet::const_iterator itersas=rechitsasRangeIteratorBegin;
    numrechitsas = rechitsasRangeIteratorEnd - rechitsasRangeIteratorBegin;   
    if(numrechitsas > 0){
      totrechitsas+=numrechitsas;
      for(itersas=rechitsasRangeIteratorBegin; itersas!=rechitsasRangeIteratorEnd;++itersas){
	const GeomDetUnit *  det = tracker.idToDetUnit(detid);
	const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
	const StripTopology &topol=(StripTopology&)stripdet->topology();
	SiStripRecHit2D const rechit=*itersas;
	LocalPoint position=rechit.localPosition();
	LocalError error=rechit.localPositionError();
	MeasurementPoint Mposition;
	MeasurementError Merror;
	Mposition = topol.measurementPosition(position);
	Merror = topol.measurementError(position,error);
	SiStripRecHit2D::ClusterRef clust=rechit.cluster();	int clusiz=0;
	int totcharge=0;
	clusiz = clust->amplitudes().size();
	const std::vector<uint8_t> amplitudes=clust->amplitudes();
	for(size_t ia=0; ia<amplitudes.size();ia++){
	  totcharge+=amplitudes[ia];
	}
	
	rechitsasx[j] = position.x();
	rechitsasy[j] = position.y();
	rechitsasz[j] = position.z();
	rechitsaserrx[j] = error.xx();
	clusizsas[j] = clusiz;
	cluchgsas[j] = totcharge;
	
	float mindist = 999999;
	float dist = 999999;
	PSimHit closest;
	matched.clear();
	matched = associate.associateHit(rechit);
	if(!matched.empty()){
	  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    dist = fabs(rechitsasx[j] - (*m).localPosition().x());
	    if(dist<mindist){
	      mindist = dist;
	      closest = (*m);
	    }
	    rechitsasres[j] = rechitsasx[j] - closest.localPosition().x();
	  }  
	  rechitsaspullMF[j] = (Mposition.x() - (topol.measurementPosition(closest.localPosition())).x())/sqrt(Merror.uu());
	  //chi2test compare rechit errors with the simhit position ( using null matrix for the simhit). 
	  //Can spot problems in the geometry better than a simple residual. (thanks to BorisM)
// 	  AlgebraicVector rhparameters = rechit.parameters();
	  AlgebraicVector  rhparameters(2);//= rechit.parameters();
	  rhparameters[0] = position.x(); 
	  rhparameters[1] = position.y();
	  AlgebraicVector shparameters(2);
	  shparameters[0] = closest.localPosition().x();
	  shparameters[1] = closest.localPosition().y();
	  AlgebraicVector r(rhparameters - shparameters);
// 	  AlgebraicSymMatrix R  = rechit.parametersError();
	  AlgebraicSymMatrix R(2);//  = rechit.parametersError();
	  R[0][0] = error.xx();
	  R[0][1] = error.xy();
	  R[1][1] = error.yy();
	  int ierr; 
	  R.invert(ierr); // if (ierr != 0) throw exception;
	  double est = R.similarity(r);
	  // 	  std::cout << " ====== Chi2 test sas hits ====== " << std::endl;
	  //	  std::cout << "RecHit param. = " << rhparameters << std::endl;
	  // 	  std::cout << "RecHit errors = " << R << std::endl;
	  // 	  std::cout << "SimHit param. = " << shparameters << std::endl;
	  // 	  std::cout << " chi2  = " << est << std::endl;
	  //	  std::cout << "DEBUG BORIS,filling chi2sas[j],j: " << i << std::endl;
	  chi2sas[j2] = est;
	  j2++;
	}
	j++;
      }
    }
    
    //now matched hits

    int k=0;
    int k2=0;
    
    //loop over rechits-matched in the same subdetector
    numrechitmatched=0;
    std::pair<SiStripMatchedRecHit2DCollection::DetSet::const_iterator,SiStripMatchedRecHit2DCollection::DetSet::const_iterator> rechitmatchedRange = helper::getRange(*rechitsmatched, detid);
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitmatchedRangeIteratorBegin = rechitmatchedRange.first;
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitmatchedRangeIteratorEnd   = rechitmatchedRange.second;
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator itermatched=rechitmatchedRangeIteratorBegin;
    numrechitmatched = rechitmatchedRangeIteratorEnd - rechitmatchedRangeIteratorBegin;   
    if(numrechitmatched > 0){
      totrechitmatched +=numrechitmatched;

      for(itermatched=rechitmatchedRangeIteratorBegin; itermatched!=rechitmatchedRangeIteratorEnd;++itermatched){
	SiStripMatchedRecHit2D const rechit=*itermatched;
	LocalPoint position=rechit.localPosition();
	LocalError error=rechit.localPositionError();
	
	float mindist = 999999;
	float distx = 999999;
	float disty = 999999;
	float dist = 999999;
	std::pair<LocalPoint,LocalVector> closestPair;
	matched.clear();
	//	const SiStripRecHit2D *mono = rechit.monoHit();
	//	const SiStripRecHit2D *st = rechit.stereoHit();
	//	LocalPoint monopos = mono->localPosition();
	//	LocalPoint stpos   = st->localPosition();
	
	rechitmatchedx[k] = position.x();
	rechitmatchedy[k] = position.y();
	rechitmatchedz[k] = position.z();
	rechitmatchederrxx[k] = error.xx();
	rechitmatchederrxy[k] = error.xy();
	rechitmatchederryy[k] = error.yy();

	//	std::cout << " before association " << std::endl;
	matched = associate.associateHit(rechit);
	//std::cout << " after association size = " << matched.size() << std::endl;

	if(!matched.empty()){
	  //project simhit;
	  const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(rechit.geographicalId());
	  const StripGeomDetUnit* partnerstripdet =(StripGeomDetUnit*) gluedDet->stereoDet();
	  std::pair<LocalPoint,LocalVector> hitPair;

	  //std::cout << " RECHIT position = " << position << std::endl;	  
	  
	  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    //project simhit;
	    hitPair= projectHit((*m),partnerstripdet,gluedDet->surface());
	    distx = fabs(rechitmatchedx[k] - hitPair.first.x());
	    disty = fabs(rechitmatchedy[k] - hitPair.first.y());
	    dist = sqrt(distx*distx+disty*disty);
	    // std::cout << " Simhit position x = " << hitPair.first.x() 
	    //      << " y = " << hitPair.first.y() << " dist = " << dist << std::endl;	  
	    if(dist<mindist){
	      mindist = dist;
	      closestPair = hitPair;
	    }
	  }
	  //std::cout << " Closest position x = " << closestPair.first.x() 
	  //      << " y = " << closestPair.first.y() << " dist = " << dist << std::endl;	  
	  rechitmatchedresx[k] = rechitmatchedx[k] - closestPair.first.x();
	  rechitmatchedresy[k] = rechitmatchedy[k] - closestPair.first.y();

 	  //chi2test compare rechit errors with the simhit position ( using null matrix for the simhit). 
 	  //Can spot problems in the geometry better than a simple residual. (thanks to BorisM)

	  AlgebraicVector  rhparameters(2);//= rechit.parameters();
	  rhparameters[0] = position.x(); 
	  rhparameters[1] = position.y();
 	  LocalPoint sh = closestPair.first;
 	  AlgebraicVector shparameters(2);
 	  shparameters[0] = sh.x();
 	  shparameters[1] = sh.y();
 	  AlgebraicVector r(rhparameters - shparameters);
	  AlgebraicSymMatrix R(2);//  = rechit.parametersError();
	  R[0][0] = error.xx();
	  R[0][1] = error.xy();
	  R[1][1] = error.yy();
	  int ierr; 
	  R.invert(ierr); // if (ierr != 0) throw exception;
	  double est = R.similarity(r);
	  // 	  std::cout << " ====== Chi2 test matched ====== " << std::endl;
	  // 	  std::cout << "RecHit param. = " << rhparameters << std::endl;
	  // 	  std::cout << "RecHit errors = " << R << std::endl;
	  // 	  std::cout << "SimHit param. = " << shparameters << std::endl;
	  // 	  std::cout << " chi2  = " << est << std::endl;
	  chi2matched[k2] = est;
	  k2++;
	}
	
	k++;
      }
    }
    
    //for each detid
    if(numrechitrphi>0 || numrechitsas>0 || numrechitmatched){
      if (detid.subdetId() == int(StripSubdetector::TIB)){
	
	int Tibnumrechitrphi    = numrechitrphi;
	int Tibnumrechitsas     = numrechitsas;
	int Tibnumrechitmatched = numrechitmatched;

	int Tibnumrechitrphi2    = i2;
	int Tibnumrechitsas2     = j2;
	int Tibnumrechitmatched2 = k2;

	totTibnumrechitrphi +=numrechitrphi;
	totTibnumrechitsas +=numrechitsas;
	totTibnumrechitmatched +=numrechitmatched;

	int ilay = tTopo->tibLayer(myid) - 1; //for histogram filling
	
	if(tTopo->tibStereo(myid)==0){
	  for(int k = 0; k<Tibnumrechitrphi; k++){
	    meNstpRphiTIB[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTIB[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTIB[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTIB[ilay]->Fill(sqrt(rechitrphierrx[k]));
	    meResRphiTIB[ilay]->Fill(rechitrphires[k]);
	    mePullLFRphiTIB[ilay]->Fill(rechitrphires[k]/sqrt(rechitrphierrx[k]));
	    mePullMFRphiTIB[ilay]->Fill(rechitrphipullMF[k]);

	  }

	  for(int k = 0; k<Tibnumrechitrphi2; k++){
	    meChi2RphiTIB[ilay]->Fill(chi2rphi[k]);
	  }

	} else  if(tTopo->tibStereo(myid)==1){
	  for(int kk = 0; kk < Tibnumrechitsas; kk++)	    
	    {
	      meNstpSasTIB[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTIB[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTIB[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTIB[ilay]->Fill(sqrt(rechitsaserrx[kk]));
	      meResSasTIB[ilay]->Fill(rechitsasres[kk]);
	      mePullLFSasTIB[ilay]->Fill(rechitsasres[kk]/sqrt(rechitsaserrx[kk]));
	      mePullMFSasTIB[ilay]->Fill(rechitsaspullMF[kk]);
	    }	  
	  for(int l = 0; l<Tibnumrechitsas2; l++){
	    meChi2SasTIB[ilay]->Fill(chi2sas[l]);
	  }

	}
	if(Tibnumrechitmatched>0){
	  for(int kkk = 0; kkk<Tibnumrechitmatched; kkk++)
	    {
	      mePosxMatchedTIB[ilay]->Fill(rechitmatchedx[kkk]);
	      mePosyMatchedTIB[ilay]->Fill(rechitmatchedy[kkk]);
	      meErrxMatchedTIB[ilay]->Fill(sqrt(rechitmatchederrxx[kkk]));
	      meErryMatchedTIB[ilay]->Fill(sqrt(rechitmatchederryy[kkk]));
	      meResxMatchedTIB[ilay]->Fill(rechitmatchedresx[kkk]);
	      meResyMatchedTIB[ilay]->Fill(rechitmatchedresy[kkk]);
	    }	  
	  for(int l = 0; l<Tibnumrechitmatched2; l++){
	    meChi2MatchedTIB[ilay]->Fill(chi2matched[l]);
	  }

	}
      }


      if (detid.subdetId() == int(StripSubdetector::TOB)){
	
	int Tobnumrechitrphi    = numrechitrphi;
	int Tobnumrechitsas     = numrechitsas;
	int Tobnumrechitmatched = numrechitmatched;
	totTobnumrechitrphi +=numrechitrphi;
	totTobnumrechitsas +=numrechitsas;
	totTobnumrechitmatched +=numrechitmatched;

	int Tobnumrechitrphi2    = i2;
	int Tobnumrechitsas2     = j2;
	int Tobnumrechitmatched2 = k2;

	int ilay = tTopo->tobLayer(myid) - 1; //for histogram filling
	
	if(tTopo->tobStereo(myid)==0){
	  for(int k = 0; k<Tobnumrechitrphi; k++){
	    meNstpRphiTOB[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTOB[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTOB[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTOB[ilay]->Fill(sqrt(rechitrphierrx[k]));
	    meResRphiTOB[ilay]->Fill(rechitrphires[k]);
	    mePullLFRphiTOB[ilay]->Fill(rechitrphires[k]/sqrt(rechitrphierrx[k]));
	    mePullMFRphiTOB[ilay]->Fill(rechitrphipullMF[k]);
	  }
	  for(int l = 0; l<Tobnumrechitrphi2; l++){
	    meChi2RphiTOB[ilay]->Fill(chi2rphi[l]);
	  }
	} else  if(tTopo->tobStereo(myid)==1){
	  for(int kk = 0; kk < Tobnumrechitsas; kk++)	    
	    {
	      meNstpSasTOB[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTOB[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTOB[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTOB[ilay]->Fill(sqrt(rechitsaserrx[kk]));
	      meResSasTOB[ilay]->Fill(rechitsasres[kk]);
	      mePullLFSasTOB[ilay]->Fill(rechitsasres[kk]/sqrt(rechitsaserrx[kk]));
	      mePullMFSasTOB[ilay]->Fill(rechitsaspullMF[kk]);
	    }
	  for(int l = 0; l<Tobnumrechitsas2; l++){
	    meChi2SasTOB[ilay]->Fill(chi2sas[l]);
	  }
	  
	}
	if(Tobnumrechitmatched>0){
	  for(int kkk = 0; kkk<Tobnumrechitmatched; kkk++)
	    {
	      mePosxMatchedTOB[ilay]->Fill(rechitmatchedx[kkk]);
	      mePosyMatchedTOB[ilay]->Fill(rechitmatchedy[kkk]);
	      meErrxMatchedTOB[ilay]->Fill(sqrt(rechitmatchederrxx[kkk]));
	      meErryMatchedTOB[ilay]->Fill(sqrt(rechitmatchederryy[kkk]));
	      meResxMatchedTOB[ilay]->Fill(rechitmatchedresx[kkk]);
	      meResyMatchedTOB[ilay]->Fill(rechitmatchedresy[kkk]);
	    }	  
	  for(int l = 0; l<Tobnumrechitmatched2; l++){
	    meChi2MatchedTOB[ilay]->Fill(chi2matched[l]);
	  }

	}
      }
      if (detid.subdetId() == int(StripSubdetector::TID)){
	
	int Tidnumrechitrphi    = numrechitrphi;
	int Tidnumrechitsas     = numrechitsas;
	int Tidnumrechitmatched = numrechitmatched;
	totTidnumrechitrphi +=numrechitrphi;
	totTidnumrechitsas +=numrechitsas;
	totTidnumrechitmatched +=numrechitmatched;

	int Tidnumrechitrphi2    = i2;
	int Tidnumrechitsas2     = j2;
	int Tidnumrechitmatched2 = k2;

	int ilay = tTopo->tidRing(myid) - 1; //for histogram filling
	
	if(tTopo->tidStereo(myid)==0){
	  for(int k = 0; k<Tidnumrechitrphi; k++){
	    meNstpRphiTID[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTID[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTID[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTID[ilay]->Fill(sqrt(rechitrphierrx[k]));
	    meResRphiTID[ilay]->Fill(rechitrphires[k]);
	    mePullLFRphiTID[ilay]->Fill(rechitrphires[k]/sqrt(rechitrphierrx[k]));
	    mePullMFRphiTID[ilay]->Fill(rechitrphipullMF[k]);
	  }
	  for(int l = 0; l<Tidnumrechitrphi2; l++){
	    meChi2RphiTID[ilay]->Fill(chi2rphi[l]);
	  }

	} else  if(tTopo->tidStereo(myid)==1){
	  for(int kk = 0; kk < Tidnumrechitsas; kk++)	    
	    {
	      meNstpSasTID[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTID[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTID[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTID[ilay]->Fill(sqrt(rechitsaserrx[kk]));
	      meResSasTID[ilay]->Fill(rechitsasres[kk]);
	      mePullLFSasTID[ilay]->Fill(rechitsasres[kk]/sqrt(rechitsaserrx[kk]));
	      mePullMFSasTID[ilay]->Fill(rechitsaspullMF[kk]);
	    }	  
	  for(int l = 0; l<Tidnumrechitsas2; l++){
	    meChi2SasTID[ilay]->Fill(chi2sas[l]);
	  }

	}
	if(Tidnumrechitmatched>0){
	  for(int kkk = 0; kkk<Tidnumrechitmatched; kkk++)
	    {
	      mePosxMatchedTID[ilay]->Fill(rechitmatchedx[kkk]);
	      mePosyMatchedTID[ilay]->Fill(rechitmatchedy[kkk]);
	      meErrxMatchedTID[ilay]->Fill(sqrt(rechitmatchederrxx[kkk]));
	      meErryMatchedTID[ilay]->Fill(sqrt(rechitmatchederryy[kkk]));
	      meResxMatchedTID[ilay]->Fill(rechitmatchedresx[kkk]);
	      meResyMatchedTID[ilay]->Fill(rechitmatchedresy[kkk]);
	    }
	  for(int l = 0; l<Tidnumrechitmatched2; l++){
	    meChi2MatchedTID[ilay]->Fill(chi2matched[l]);
	  }
	  
	}
      }
      if (detid.subdetId() == int(StripSubdetector::TEC)){
	
	int Tecnumrechitrphi    = numrechitrphi;
	int Tecnumrechitsas     = numrechitsas;
	int Tecnumrechitmatched = numrechitmatched;
	totTecnumrechitrphi +=numrechitrphi;
	totTecnumrechitsas +=numrechitsas;
	totTecnumrechitmatched +=numrechitmatched;

	int Tecnumrechitrphi2    = i2;
	int Tecnumrechitsas2     = j2;
	int Tecnumrechitmatched2 = k2;

	int ilay = tTopo->tecRing(myid) - 1; //for histogram filling
	
	if(tTopo->tecStereo(myid)==0){
	  for(int k = 0; k<Tecnumrechitrphi; k++){
	    meNstpRphiTEC[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTEC[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTEC[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTEC[ilay]->Fill(sqrt(rechitrphierrx[k]));
	    meResRphiTEC[ilay]->Fill(rechitrphires[k]);
	    mePullLFRphiTEC[ilay]->Fill(rechitrphires[k]/sqrt(rechitrphierrx[k]));
	    mePullMFRphiTEC[ilay]->Fill(rechitrphipullMF[k]);
	  }
	  for(int l = 0; l<Tecnumrechitrphi2; l++){
	    meChi2RphiTEC[ilay]->Fill(chi2rphi[l]);
	  }

	} else  if(tTopo->tecStereo(myid)==1){
	  for(int kk = 0; kk < Tecnumrechitsas; kk++)	    
	    {
	      meNstpSasTEC[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTEC[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTEC[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTEC[ilay]->Fill(sqrt(rechitsaserrx[kk]));
	      meResSasTEC[ilay]->Fill(rechitsasres[kk]);
	      mePullLFSasTEC[ilay]->Fill(rechitsasres[kk]/sqrt(rechitsaserrx[kk]));
	      mePullMFSasTEC[ilay]->Fill(rechitsaspullMF[kk]);
	    }	  
	  for(int l = 0; l<Tecnumrechitsas2; l++){
	    meChi2SasTEC[ilay]->Fill(chi2sas[l]);
	  }

	}
	if(Tecnumrechitmatched>0){
	  for(int kkk = 0; kkk<Tecnumrechitmatched; kkk++)
	    {
	      mePosxMatchedTEC[ilay]->Fill(rechitmatchedx[kkk]);
	      mePosyMatchedTEC[ilay]->Fill(rechitmatchedy[kkk]);
	      meErrxMatchedTEC[ilay]->Fill(sqrt(rechitmatchederrxx[kkk]));
	      meErryMatchedTEC[ilay]->Fill(sqrt(rechitmatchederryy[kkk]));
	      meResxMatchedTEC[ilay]->Fill(rechitmatchedresx[kkk]);
	      meResyMatchedTEC[ilay]->Fill(rechitmatchedresy[kkk]);
	    }	  
	  for(int l = 0; l<Tecnumrechitmatched2; l++){
	    meChi2MatchedTEC[ilay]->Fill(chi2matched[l]);
	  }

	}
      }

    }
  }

  //now fill the cumulative histograms of the hits

  meNumRphiTIB->Fill(totTibnumrechitrphi);
  meNumSasTIB->Fill(totTibnumrechitsas);
  meNumMatchedTIB->Fill(totTibnumrechitmatched);
  meNumRphiTOB->Fill(totTobnumrechitrphi);
  meNumSasTOB->Fill(totTobnumrechitsas);
  meNumMatchedTOB->Fill(totTobnumrechitmatched);
  meNumRphiTID->Fill(totTidnumrechitrphi);
  meNumSasTID->Fill(totTidnumrechitsas);
  meNumMatchedTID->Fill(totTidnumrechitmatched);
  meNumRphiTEC->Fill(totTecnumrechitrphi);
  meNumSasTEC->Fill(totTecnumrechitsas);
  meNumMatchedTEC->Fill(totTecnumrechitmatched);

  meNumTotRphi->Fill(totrechitrphi);
  meNumTotSas->Fill(totrechitsas);
  meNumTotMatched->Fill(totrechitmatched);

 
}

  
//needed by to do the residual for matched hits
std::pair<LocalPoint,LocalVector> SiStripRecHitsValid::projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,
								   const BoundPlane& plane) 
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


