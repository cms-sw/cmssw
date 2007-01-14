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
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

using namespace std;
using namespace edm;

SiStripTrackingRecHitsValid::SiStripTrackingRecHitsValid(const ParameterSet& ps):dbe_(0)
{

  //Read config file
  MTCCtrack_ = ps.getParameter<bool>("MTCCtrack");
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "striptrackingrechitshisto.root");
  src_ = ps.getUntrackedParameter<std::string>( "src" );
  builderName_ = ps.getParameter<std::string>("TTRHBuilder");   

  // Book histograms
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  dbe_->showDirStructure();

  dbe_->setCurrentFolder("TIB");
  //one histo per Layer rphi hits
  for(int i = 0 ;i<4 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtib",i+1);
    meNstpRphiTIB[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%dtib",i+1);
    meAdcRphiTIB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtib",i+1);
    mePosxRphiTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%dtib",i+1);
    meErrxRphiTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.05);  
    sprintf(histo,"Res_rphi_layer%dtib",i+1);
    meResRphiTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Pull_rphi_layer%dtib",i+1);
    mePullRphiTIB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Trackangle_rphi_layer%dtib",i+1);
    meTrackangleRphiTIB[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  
    sprintf(histo,"PullTrackangleProfile_rphi_layer%dtib",i+1);
    mePullTrackangleProfileRphiTIB[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -20., 20.,100, -2.,2.,"s");
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
    sprintf(histo,"Errx_sas_layer%dtib",i+1);
    meErrxSasTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.05);  
    sprintf(histo,"Res_sas_layer%dtib",i+1);
    meResSasTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Pull_sas_layer%dtib",i+1);
    mePullSasTIB[i] = dbe_->book1D(histo,"Pull",100,-4.,4.);  
    sprintf(histo,"Trackangle_sas_layer%dtib",i+1);
    meTrackangleSasTIB[i] = dbe_->book1D(histo,"Track angle",100,-40.,40.);  
    sprintf(histo,"PullTrackangleProfile_sas_layer%dtib",i+1);
    mePullTrackangleProfileSasTIB[i] = dbe_->bookProfile(histo,"Pull Track angle Profile",  100, -40., 40.,100,-4.,4.,"s");

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

  dbe_->setCurrentFolder("TOB");
  //one histo per Layer rphi hits
  for(int i = 0 ;i<6 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtob",i+1);
    meNstpRphiTOB[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%dtob",i+1);
    meAdcRphiTOB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtob",i+1);
    mePosxRphiTOB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%dtob",i+1);
    meErrxRphiTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.05);  
    sprintf(histo,"Res_rphi_layer%dtob",i+1);
    meResRphiTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Pull_rphi_layer%dtob",i+1);
    mePullRphiTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Trackangle_rphi_layer%dtob",i+1);
    meTrackangleRphiTOB[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  
    sprintf(histo,"PullTrackangleProfile_rphi_layer%dtob",i+1);
    mePullTrackangleProfileRphiTOB[i] = dbe_->bookProfile(histo,"Pull Track angle Profile",  100, -20., 20.,100,-5.,5.,"s");
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
    sprintf(histo,"Errx_sas_layer%dtob",i+1);
    meErrxSasTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.05);  
    sprintf(histo,"Res_sas_layer%dtob",i+1);
    meResSasTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
    sprintf(histo,"Pull_sas_layer%dtob",i+1);
    mePullSasTOB[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Trackangle_sas_layer%dtob",i+1);
    meTrackangleSasTOB[i] = dbe_->book1D(histo,"Track angle",100,-25.,25.);  
    sprintf(histo,"PullTrackangleProfile_sas_layer%dtob",i+1);
    mePullTrackangleProfileSasTOB[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -25., 25. ,100 , -5., 5.,"s");

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

  dbe_->setCurrentFolder("TID");
  //one histo per Ring rphi hits: 3 rings, 6 disks, 2 inner rings are glued 
  for(int i = 0 ;i<3 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtid",i+1);
    meNstpRphiTID[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%dtid",i+1);
    meAdcRphiTID[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtid",i+1);
    mePosxRphiTID[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%dtid",i+1);
    meErrxRphiTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.1);  
    sprintf(histo,"Res_rphi_layer%dtid",i+1);
    meResRphiTID[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Pull_rphi_layer%dtid",i+1);
    mePullRphiTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Trackangle_rphi_layer%dtid",i+1);
    meTrackangleRphiTID[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  
    sprintf(histo,"PullTrackangleProfile_rphi_layer%dtid",i+1);
    mePullTrackangleProfileRphiTID[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -20., 20.,100, -5., 5.,"s");
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
    sprintf(histo,"Errx_sas_layer%dtid",i+1);
    meErrxSasTID[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.1);  
    sprintf(histo,"Res_sas_layer%dtid",i+1);
    meResSasTID[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Pull_sas_layer%dtid",i+1);
    mePullSasTID[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Trackangle_sas_layer%dtid",i+1);
    meTrackangleSasTID[i] = dbe_->book1D(histo,"Track angle",100,-20.,20.);  
    sprintf(histo,"PullTrackangleProfile_sas_layer%dtid",i+1);
    mePullTrackangleProfileSasTID[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -20., 20.,100, -5., 5.,"s");

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

  dbe_->setCurrentFolder("TEC");
  //one histo per Ring rphi hits: 7 rings, 18 disks. Innermost 3 rings are same as TID above.  
  for(int i = 0 ;i<7 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%dtec",i+1);
    meNstpRphiTEC[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%dtec",i+1);
    meAdcRphiTEC[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%dtec",i+1);
    mePosxRphiTEC[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%dtec",i+1);
    meErrxRphiTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.1);  
    sprintf(histo,"Res_rphi_layer%dtec",i+1);
    meResRphiTEC[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
    sprintf(histo,"Pull_rphi_layer%dtec",i+1);
    mePullRphiTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);  
    sprintf(histo,"Trackangle_rphi_layer%dtec",i+1);
    meTrackangleRphiTEC[i] = dbe_->book1D(histo,"Track angle",100,-10.,10.);  
    sprintf(histo,"PullTrackangleProfile_rphi_layer%dtec",i+1);
    mePullTrackangleProfileRphiTEC[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -10., 10.,100, -5., 5.,"s");
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
      sprintf(histo,"Errx_sas_layer%dtec",i+1);
      meErrxSasTEC[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.05);  
      sprintf(histo,"Res_sas_layer%dtec",i+1);
      meResSasTEC[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
      sprintf(histo,"Pull_sas_layer%dtec",i+1);
      mePullSasTEC[i] = dbe_->book1D(histo,"Pull",100,-5.,5.);
      sprintf(histo,"Trackangle_sas_layer%dtec",i+1);
      meTrackangleSasTEC[i] = dbe_->book1D(histo,"Track angle",100,-10.,10.);
      sprintf(histo,"PullTrackangleProfile_sas_layer%dtec",i+1);
      mePullTrackangleProfileSasTEC[i] = dbe_->bookProfile(histo,"Pull Track angle Profile", 100, -10., 10.,100, -5., 5.,"s");
     
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

  // Instantiate angle finder
  anglefinder_ = new  TrackLocalAngle(ps);

}

// Virtual destructor needed.
SiStripTrackingRecHitsValid::~SiStripTrackingRecHitsValid() {  

  delete anglefinder_;
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}  

// Functions that gets called by framework every event
void SiStripTrackingRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es)
{

  int isrechitrphi     = 0;
  int isrechitsas      = 0;
  int isrechitmatched  = 0;

  DetId detid;
  uint32_t myid;

  LocalPoint position;
  LocalError error;

  int clusiz=0;
  int totcharge=0;

  float mindist = 999999;
  float dist;
  std::vector<PSimHit> matched;
  TrackerHitAssociator associate(e);
  PSimHit closest;

  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry &tracker(*pDD);

   // Initialize angle finder
  anglefinder_->init(e,es);

  if(!MTCCtrack_){
    // Get tracks
    edm::Handle<reco::TrackCollection> trackCollection;
    e.getByLabel(src_, trackCollection);
    const reco::TrackCollection *tracks=trackCollection.product();
    reco::TrackCollection::const_iterator tciter;

    if(tracks->size()>0){
      // Loop on tracks
      for(tciter=tracks->begin();tciter!=tracks->end();tciter++){

	/////////////////////////////////////////
	// First loop on hits: find matched hits
	/////////////////////////////////////////
        for (trackingRecHit_iterator it = tciter->recHitsBegin();  it != tciter->recHitsEnd(); it++)
	  {
	    isrechitmatched = 0;
	    const TrackingRecHit &thit = **it;
	    
            // subdetector infos
	    detid = (*it)->geographicalId();
	    myid=((*it)->geographicalId()).rawId();

	    // Is it a matched hit?
	    const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&thit);

	    if(matchedhit){

              isrechitmatched = 1;
 
	      position = (*it)->localPosition();
	      error = (*it)->localPositionError();
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
		rechitmatchedpullx = ((*it)->localPosition().x() - (closestPair.first.x()))/sqrt(error.xx());
		rechitmatchedpully = ((*it)->localPosition().y() - (closestPair.first.y()))/sqrt(error.yy());
	      }
	    }
 
            //Filling Histograms for Matched hits

	    if(isrechitmatched){

	      if (detid.subdetId() == int(StripSubdetector::TIB)){
		TIBDetId tibid(myid);
		int Tibisrechitmatched = isrechitmatched;
		int ilay = tibid.layer() - 1; //for histogram filling
		if(Tibisrechitmatched>0){
		  mePosxMatchedTIB[ilay]->Fill(rechitmatchedx);
		  mePosyMatchedTIB[ilay]->Fill(rechitmatchedy);
		  meErrxMatchedTIB[ilay]->Fill(sqrt(rechitmatchederrxx));
		  meErryMatchedTIB[ilay]->Fill(sqrt(rechitmatchederryy));
		  meResxMatchedTIB[ilay]->Fill(rechitmatchedresx);
		  meResyMatchedTIB[ilay]->Fill(rechitmatchedresy);
		  mePullxMatchedTIB[ilay]->Fill(rechitmatchedpullx);
		  mePullyMatchedTIB[ilay]->Fill(rechitmatchedpully);
		}
	      }
	      if (detid.subdetId() == int(StripSubdetector::TOB)){
		TOBDetId tobid(myid);
		int Tobisrechitmatched = isrechitmatched;
		int ilay = tobid.layer() - 1; //for histogram filling
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
		TIDDetId tidid(myid);
		int Tidisrechitmatched = isrechitmatched;
		int ilay = tidid.ring() - 1; //for histogram filling
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
		TECDetId tecid(myid);
		int Tecisrechitmatched = isrechitmatched;
		int ilay = tecid.ring() - 1; //for histogram filling
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
	  }


	///////////////////////////////////////////////////////
	// Second loop on hits: find simple hits
	///////////////////////////////////////////////////////

        // Get the track angle
	std::vector<std::pair<const TrackingRecHit *,float> > hitangle=anglefinder_->findtrackangle(*tciter);
	std::vector<std::pair<const TrackingRecHit *,float> >::iterator iter;

        // Loop on the track hits
	for(iter=hitangle.begin();iter!=hitangle.end();iter++){

          // Reset variables
	  isrechitrphi    = 0;
	  isrechitsas     = 0;
	  rechitrphix =0;
	  rechitrphierrx =0;
	  rechitrphiy =0;
	  rechitrphiz =0;
	  rechitsasx =0;
	  rechitsaserrx =0;
	  rechitsasy =0;
	  rechitsasz =0;
	  clusizrphi =0;
	  clusizsas =0;
	  cluchgrphi =0;
	  cluchgsas =0;
	  rechitrphires=-999.;
	  rechitrphipull=-999.;
	  rechitrphitrackangle =0;
	  rechitsasres=-999.;
	  rechitsaspull=-999.;
	  rechitsastrackangle =0;

          // Track angle
          float angle=iter->second;

	  // subdetector infos
	  detid = (iter->first)->geographicalId();
	  StripSubdetector StripSubdet = (StripSubdetector) detid;
	  myid=((iter->first)->geographicalId()).rawId();

	  const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(iter->first);

	  if(hit){
	    // simple hits are mono or stereo
	    if (StripSubdet.stereo() == 0){

		isrechitrphi = 1;
		const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();

		position = (iter->first)->localPosition();
		error = (iter->first)->localPositionError();
		clusiz=0;
		totcharge=0;
		clusiz = cluster->amplitudes().size();
		const std::vector<uint16_t> amplitudes=cluster->amplitudes();
		for(size_t ia=0; ia<amplitudes.size();ia++){
		  totcharge+=amplitudes[ia];
		}
		rechitrphix = position.x();
		rechitrphiy = position.y();
		rechitrphiz = position.z();
		rechitrphierrx = error.xx();
		clusizrphi = clusiz;
		cluchgrphi = totcharge;

		//Association of the rechit to the simhit
		mindist = 999999;
		matched.clear();  
		matched = associate.associateHit(*hit);
		if(!matched.empty()){
		  //		  cout << "\t\t\tmatched  " << matched.size() << endl;
		  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
		    dist = abs((hit)->localPosition().x() - (*m).localPosition().x());
		    if(dist<mindist){
		      mindist = dist;
		      closest = (*m);
		    }
		    rechitrphires = rechitrphix - closest.localPosition().x();
		    rechitrphipull = (((iter->first))->localPosition().x() - (closest).localPosition().x())/sqrt(error.xx());
		  }
		}
		rechitrphitrackangle = angle;

	    }
	    
	    if (StripSubdet.stereo() == 1){

		isrechitsas = 1;
		const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();

		position = (iter->first)->localPosition();
		error = (iter->first)->localPositionError();
		clusiz=0;
		totcharge=0;
		clusiz = cluster->amplitudes().size();
		const std::vector<uint16_t> amplitudes=cluster->amplitudes();
		for(size_t ia=0; ia<amplitudes.size();ia++){
		  totcharge+=amplitudes[ia];
		}
		rechitsasx = position.x();
		rechitsasy = position.y();
		rechitsasz = position.z();
		rechitsaserrx = error.xx();
		clusizsas = clusiz;
		cluchgsas = totcharge;

		//Association of the rechit to the simhit
		mindist = 999999;
		matched.clear();  
		matched = associate.associateHit(*hit);
		if(!matched.empty()){
		  //		  cout << "\t\t\tmatched  " << matched.size() << endl;
		  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
		    dist = abs((hit)->localPosition().x() - (*m).localPosition().x());
		    if(dist<mindist){
		      mindist = dist;
		      closest = (*m);
		    }
		    rechitsasres = rechitsasx - closest.localPosition().x();
		    rechitsaspull = (((iter->first))->localPosition().x() - (closest).localPosition().x())/sqrt(error.xx());
		  }
		}
		  rechitsastrackangle = angle;

	    }

	  }

          //Filling Histograms for simple hits

	  if(isrechitrphi>0 || isrechitsas>0){

	      if (detid.subdetId() == int(StripSubdetector::TIB)){
		TIBDetId tibid(myid);
		int Tibisrechitrphi    = isrechitrphi;
		int Tibisrechitsas     = isrechitsas;
		int ilay = tibid.layer() - 1; //for histogram filling
		if(Tibisrechitrphi!=0){
		  meNstpRphiTIB[ilay]->Fill(clusizrphi);
		  meAdcRphiTIB[ilay]->Fill(cluchgrphi);
		  mePosxRphiTIB[ilay]->Fill(rechitrphix);
		  meErrxRphiTIB[ilay]->Fill(sqrt(rechitrphierrx));
		  meResRphiTIB[ilay]->Fill(rechitrphires);
		  mePullRphiTIB[ilay]->Fill(rechitrphipull);
		  meTrackangleRphiTIB[ilay]->Fill(rechitrphitrackangle);
		  mePullTrackangleProfileRphiTIB[ilay]->Fill(rechitrphitrackangle,rechitrphipull);
		} else  if(Tibisrechitsas!=0){
		  meNstpSasTIB[ilay]->Fill(clusizsas);
		  meAdcSasTIB[ilay]->Fill(cluchgsas);
		  mePosxSasTIB[ilay]->Fill(rechitsasx);
		  meErrxSasTIB[ilay]->Fill(sqrt(rechitsaserrx));
		  meResSasTIB[ilay]->Fill(rechitsasres);
		  mePullSasTIB[ilay]->Fill(rechitsaspull);
		  meTrackangleSasTIB[ilay]->Fill(rechitsastrackangle);
		  mePullTrackangleProfileSasTIB[ilay]->Fill(rechitsaspull,rechitsastrackangle);
		}
	      }
	      if (detid.subdetId() == int(StripSubdetector::TOB)){
		TOBDetId tobid(myid);
		int Tobisrechitrphi    = isrechitrphi;
		int Tobisrechitsas     = isrechitsas;
		int ilay = tobid.layer() - 1; //for histogram filling
		if(Tobisrechitrphi!=0){
		  meNstpRphiTOB[ilay]->Fill(clusizrphi);
		  meAdcRphiTOB[ilay]->Fill(cluchgrphi);
		  mePosxRphiTOB[ilay]->Fill(rechitrphix);
		  meErrxRphiTOB[ilay]->Fill(sqrt(rechitrphierrx));
		  meResRphiTOB[ilay]->Fill(rechitrphires);
		  mePullRphiTOB[ilay]->Fill(rechitrphipull);
		  meTrackangleRphiTOB[ilay]->Fill(rechitrphitrackangle);
		  mePullTrackangleProfileRphiTOB[ilay]->Fill(rechitrphipull,rechitrphitrackangle);
		} else  if(Tobisrechitsas!=0){
		  meNstpSasTOB[ilay]->Fill(clusizsas);
		  meAdcSasTOB[ilay]->Fill(cluchgsas);
		  mePosxSasTOB[ilay]->Fill(rechitsasx);
		  meErrxSasTOB[ilay]->Fill(sqrt(rechitsaserrx));
		  meResSasTOB[ilay]->Fill(rechitsasres);
		  mePullSasTOB[ilay]->Fill(rechitsaspull);
		  meTrackangleSasTOB[ilay]->Fill(rechitsastrackangle);
		  mePullTrackangleProfileSasTOB[ilay]->Fill(rechitsaspull,rechitsastrackangle);
		}
	      }
	      if (detid.subdetId() == int(StripSubdetector::TID)){
		TIDDetId tidid(myid);
		int Tidisrechitrphi    = isrechitrphi;
		int Tidisrechitsas     = isrechitsas;
		int ilay = tidid.ring() - 1; //for histogram filling
		if(Tidisrechitrphi!=0){
		  meNstpRphiTID[ilay]->Fill(clusizrphi);
		  meAdcRphiTID[ilay]->Fill(cluchgrphi);
		  mePosxRphiTID[ilay]->Fill(rechitrphix);
		  meErrxRphiTID[ilay]->Fill(sqrt(rechitrphierrx));
		  meResRphiTID[ilay]->Fill(rechitrphires);
		  mePullRphiTID[ilay]->Fill(rechitrphipull);
		  meTrackangleRphiTID[ilay]->Fill(rechitrphitrackangle);
		  mePullTrackangleProfileRphiTID[ilay]->Fill(rechitrphipull,rechitrphitrackangle);
		} else  if(Tidisrechitsas!=0){
		  meNstpSasTID[ilay]->Fill(clusizsas);
		  meAdcSasTID[ilay]->Fill(cluchgsas);
		  mePosxSasTID[ilay]->Fill(rechitsasx);
		  meErrxSasTID[ilay]->Fill(sqrt(rechitsaserrx));
		  meResSasTID[ilay]->Fill(rechitsasres);
		  mePullSasTID[ilay]->Fill(rechitsaspull);
		  meTrackangleSasTID[ilay]->Fill(rechitsastrackangle);
		  mePullTrackangleProfileSasTID[ilay]->Fill(rechitsaspull,rechitsastrackangle);
		}
	      }
	      if (detid.subdetId() == int(StripSubdetector::TEC)){
		TECDetId tecid(myid);
		int Tecisrechitrphi    = isrechitrphi;
		int Tecisrechitsas     = isrechitsas;
		int ilay = tecid.ring() - 1; //for histogram filling
		if(Tecisrechitrphi!=0){
		  meNstpRphiTEC[ilay]->Fill(clusizrphi);
		  meAdcRphiTEC[ilay]->Fill(cluchgrphi);
		  mePosxRphiTEC[ilay]->Fill(rechitrphix);
		  meErrxRphiTEC[ilay]->Fill(sqrt(rechitrphierrx));
		  meResRphiTEC[ilay]->Fill(rechitrphires);
		  mePullRphiTEC[ilay]->Fill(rechitrphipull);
		  meTrackangleRphiTEC[ilay]->Fill(rechitrphitrackangle);
		  mePullTrackangleProfileRphiTEC[ilay]->Fill(rechitrphipull,rechitrphitrackangle);
		} else  if(Tecisrechitsas!=0){
		  meNstpSasTEC[ilay]->Fill(clusizsas);
		  meAdcSasTEC[ilay]->Fill(cluchgsas);
		  mePosxSasTEC[ilay]->Fill(rechitsasx);
		  meErrxSasTEC[ilay]->Fill(sqrt(rechitsaserrx));
		  meResSasTEC[ilay]->Fill(rechitsasres);
		  mePullSasTEC[ilay]->Fill(rechitsaspull);
		  meTrackangleSasTEC[ilay]->Fill(rechitsastrackangle);
		  mePullTrackangleProfileSasTEC[ilay]->Fill(rechitsaspull,rechitsastrackangle);
		}
	      }
	  }
	}
      }
    }
  }
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
