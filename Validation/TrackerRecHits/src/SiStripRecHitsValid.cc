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
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"


//--- for RecHit
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h" 
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h" 
#include "DataFormats/Common/interface/OwnVector.h" 


//Constructor
SiStripRecHitsValid::SiStripRecHitsValid(const ParameterSet& ps):dbe_(0){

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "striprechitshisto.root");
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  dbe_->showDirStructure();
  dbe_->setCurrentFolder("TIB");
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
    meErrxRphiTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.05);  
    sprintf(histo,"Res_rphi_layer%dtib",i+1);
    meResRphiTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
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
    meErrxSasTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.05);  
    sprintf(histo,"Res_sas_layer%dtib",i+1);
    meResSasTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  

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
  }

  dbe_->setCurrentFolder("TOB");
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
    meErrxRphiTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0,0.05);  
    sprintf(histo,"Res_rphi_layer%dtob",i+1);
    meResRphiTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  
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
    meErrxSasTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0.,0.05);  
    sprintf(histo,"Res_sas_layer%dtob",i+1);
    meResSasTOB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.02,+0.02);  

    sprintf(histo,"Posx_matched_layer%dtob",i+1);
    mePosxMatchedTOB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%dtob",i+1);
    mePosyMatchedTOB[i] = dbe_->book1D(histo,"RecHit y coord.",100,-10.0, +10.0);  
    sprintf(histo,"Errx_matched_layer%dtob",i+1);
    meErrxMatchedTOB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,0., 0.05);  
    sprintf(histo,"Erry_matched_layer%dtob",i+1);
    meErryMatchedTOB[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,0., 0.1);  
    sprintf(histo,"Resx_matched_layer%dtob",i+1);
    meResxMatchedTOB[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-0.02, +0.02);  
    sprintf(histo,"Resy_matched_layer%dtob",i+1);
    meResyMatchedTOB[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-1., +1.);  
  }

  dbe_->setCurrentFolder("TID");
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
  }

  dbe_->setCurrentFolder("TEC");
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
    }
  }
}


SiStripRecHitsValid::~SiStripRecHitsValid(){
 //if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void SiStripRecHitsValid::beginJob(const EventSetup& c){

}

void SiStripRecHitsValid::endJob() {
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void SiStripRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es) {

  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();  
  cout  << " Run = " << e.id().run() << " Event = " << e.id().event() << endl;  
  
  //--- get RecHits
  
  //  std::string rechitProducer = conf_.getParameter<std::string>("RecHitProducer");
  std::string rechitProducer = "LocalMeasurementConverter";
  
  // Step A: Get Inputs 
  edm::Handle<SiStripRecHit2DMatchedLocalPosCollection> rechitsmatched;
  edm::Handle<SiStripRecHit2DLocalPosCollection> rechitsrphi;
  edm::Handle<SiStripRecHit2DLocalPosCollection> rechitsstereo;
  e.getByLabel(rechitProducer,"matchedRecHit", rechitsmatched);
  e.getByLabel(rechitProducer,"rphiRecHit", rechitsrphi);
  e.getByLabel(rechitProducer,"stereoRecHit", rechitsstereo);
  
  int numrechitrphi   =0;
  int numrechitsas    =0;
  int numrechitmatched=0;
  
  TrackerHitAssociator associate(e);
  
  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry &tracker(*pDD);
  
  // loop over detunits
  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    uint32_t myid=((*it)->geographicalId()).rawId();       
    DetId detid = ((*it)->geographicalId());
    
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
      rechitmatchedx[i] =0;
      rechitmatchedy[i] =0;
      rechitmatchedz[i] =0;
      rechitmatchederrxx[i] =0;
      rechitmatchederrxy[i] =0;
      rechitmatchederryy[i] =0;
      rechitmatchedresx[i]=-999;
      rechitmatchedresy[i]=-999;
    }
    
    numrechitrphi =0;
    //loop over rechits-rphi in the same subdetector
    SiStripRecHit2DLocalPosCollection::range          rechitrphiRange = rechitsrphi->get(detid);
    SiStripRecHit2DLocalPosCollection::const_iterator rechitrphiRangeIteratorBegin = rechitrphiRange.first;
    SiStripRecHit2DLocalPosCollection::const_iterator rechitrphiRangeIteratorEnd   = rechitrphiRange.second;
    SiStripRecHit2DLocalPosCollection::const_iterator iterrphi=rechitrphiRangeIteratorBegin;
    
    numrechitrphi = rechitrphiRangeIteratorEnd - rechitrphiRangeIteratorBegin;   
    if(numrechitrphi > 0 ){
      int i=0;
      for(iterrphi=rechitrphiRangeIteratorBegin; iterrphi!=rechitrphiRangeIteratorEnd;++iterrphi){
	SiStripRecHit2DLocalPos const rechit=*iterrphi;
	LocalPoint position=rechit.localPosition();
	LocalError error=rechit.localPositionError();
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > clust=rechit.cluster();
	int clusiz=0;
	int totcharge=0;
	clusiz = clust->amplitudes().size();
	const std::vector<short> amplitudes=clust->amplitudes();
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
	float dist;
	PSimHit closest;
	if(!matched.empty()){
	  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    dist = rechitrphix[i] - (*m).localPosition().x();
	    if(dist<mindist){
	      mindist = dist;
	      closest = (*m);
	    }
	    rechitrphires[i] = rechitrphix[i] - closest.localPosition().x();
	  }  
	}
	i++;
      }
    }
    
    //loop over rechits-sas in the same subdetector
    numrechitsas=0;
    SiStripRecHit2DLocalPosCollection::range rechitsasRange = rechitsstereo->get(detid);
    SiStripRecHit2DLocalPosCollection::const_iterator rechitsasRangeIteratorBegin = rechitsasRange.first;
    SiStripRecHit2DLocalPosCollection::const_iterator rechitsasRangeIteratorEnd   = rechitsasRange.second;
    SiStripRecHit2DLocalPosCollection::const_iterator itersas=rechitsasRangeIteratorBegin;
    numrechitsas = rechitsasRangeIteratorEnd - rechitsasRangeIteratorBegin;   
    if(numrechitsas > 0){
      int j=0;
      for(itersas=rechitsasRangeIteratorBegin; itersas!=rechitsasRangeIteratorEnd;++itersas){
	SiStripRecHit2DLocalPos const rechit=*itersas;
	LocalPoint position=rechit.localPosition();
	LocalError error=rechit.localPositionError();
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > clust=rechit.cluster();	int clusiz=0;
	int totcharge=0;
	clusiz = clust->amplitudes().size();
	const std::vector<short> amplitudes=clust->amplitudes();
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
	float dist;
	PSimHit closest;
	matched.clear();
	matched = associate.associateHit(rechit);
	if(!matched.empty()){
	  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    dist = rechitsasx[j] - (*m).localPosition().x();
	    if(dist<mindist){
	      mindist = dist;
	      closest = (*m);
	    }
	    rechitsasres[j] = rechitsasx[j] - closest.localPosition().x();
	  }  
	}
	j++;
      }
    }
    
    //now matched hits
    
    //loop over rechits-matched in the same subdetector
    numrechitmatched=0;
    SiStripRecHit2DMatchedLocalPosCollection::range rechitmatchedRange = rechitsmatched->get(detid);
    SiStripRecHit2DMatchedLocalPosCollection::const_iterator rechitmatchedRangeIteratorBegin = rechitmatchedRange.first;
    SiStripRecHit2DMatchedLocalPosCollection::const_iterator rechitmatchedRangeIteratorEnd   = rechitmatchedRange.second;
    SiStripRecHit2DMatchedLocalPosCollection::const_iterator itermatched=rechitmatchedRangeIteratorBegin;
    numrechitmatched = rechitmatchedRangeIteratorEnd - rechitmatchedRangeIteratorBegin;   
    if(numrechitmatched > 0){
      int j=0;
      for(itermatched=rechitmatchedRangeIteratorBegin; itermatched!=rechitmatchedRangeIteratorEnd;++itermatched){
	SiStripRecHit2DMatchedLocalPos const rechit=*itermatched;
	LocalPoint position=rechit.localPosition();
	LocalError error=rechit.localPositionError();
	
	float mindistx = 999999;
	float distx, disty;
	std::pair<LocalPoint,LocalVector> closestPair;
	matched.clear();
	const SiStripRecHit2DLocalPos *mono = rechit.monoHit();
	const SiStripRecHit2DLocalPos *st = rechit.stereoHit();
	LocalPoint monopos = mono->localPosition();
	LocalPoint stpos   = st->localPosition();
	
	rechitmatchedx[j] = position.x();
	rechitmatchedy[j] = position.y();
	rechitmatchedz[j] = position.z();
	rechitmatchederrxx[j] = error.xx();
	rechitmatchederrxy[j] = error.xy();
	rechitmatchederryy[j] = error.yy();
	//	cout << " errors = " << sqrt(error.xx()) << ", " << error.xy() << ", " << sqrt(error.yy()) <<  endl;
	matched = associate.associateHit(*st);
	if(!matched.empty()){
	  //project simhit;
	  const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(rechit.geographicalId());
	  const StripGeomDetUnit* partnerstripdet =(StripGeomDetUnit*) gluedDet->stereoDet();
	  std::pair<LocalPoint,LocalVector> hitPair;
	  
	  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    //project simhit;
	    hitPair= projectHit((*m),partnerstripdet,gluedDet->surface());
	    distx = rechitmatchedx[j] - hitPair.first.x();
	    disty = rechitmatchedy[j] - hitPair.first.y();
	    if(distx<mindistx){
	      mindistx = distx;
	      closestPair = hitPair;
	    }
	  }
	  rechitmatchedresx[j] = rechitmatchedx[j] - closestPair.first.x();
	  rechitmatchedresy[j] = rechitmatchedy[j] - closestPair.first.y();
	  
	  // 	  cout << " res x = " << rechitmatchedresx[j] << " rec(x) = " <<  rechitmatchedx[j] 
 	  //     << " sim(x) = " << hitPair.first.x() << endl;
 	  //cout << " res y = " << rechitmatchedresy[j] << " rec(y) = " <<  rechitmatchedy[j] 
 	  //     << " sim(y) = " <<  hitPair.first.y()<< endl;
	}
	
	j++;
      }
    }
    

    if(numrechitrphi>0 || numrechitsas>0 || numrechitmatched){
      if (detid.subdetId() == int(StripSubdetector::TIB)){
	TIBDetId tibid(myid);
	int Tibnumrechitrphi    = numrechitrphi;
	int Tibnumrechitsas     = numrechitsas;
	int Tibnumrechitmatched = numrechitmatched;

	int ilay = tibid.layer() - 1; //for histogram filling
	
	if(tibid.stereo()==0){
	  for(int k = 0; k<Tibnumrechitrphi; k++){
	    meNstpRphiTIB[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTIB[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTIB[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTIB[ilay]->Fill(sqrt(rechitrphierrx[k]));
	    meResRphiTIB[ilay]->Fill(rechitrphires[k]);
	  }
	} else  if(tibid.stereo()==1){
	  for(int kk = 0; kk < Tibnumrechitsas; kk++)	    
	    {
	      meNstpSasTIB[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTIB[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTIB[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTIB[ilay]->Fill(sqrt(rechitsaserrx[kk]));
	      meResSasTIB[ilay]->Fill(rechitsasres[kk]);
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
	}
      }


      if (detid.subdetId() == int(StripSubdetector::TOB)){
	TOBDetId tobid(myid);
	int Tobnumrechitrphi    = numrechitrphi;
	int Tobnumrechitsas     = numrechitsas;
	int Tobnumrechitmatched = numrechitmatched;

	int ilay = tobid.layer() - 1; //for histogram filling
	
	if(tobid.stereo()==0){
	  for(int k = 0; k<Tobnumrechitrphi; k++){
	    meNstpRphiTOB[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTOB[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTOB[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTOB[ilay]->Fill(sqrt(rechitrphierrx[k]));
	    meResRphiTOB[ilay]->Fill(rechitrphires[k]);
	  }
	} else  if(tobid.stereo()==1){
	  for(int kk = 0; kk < Tobnumrechitsas; kk++)	    
	    {
	      meNstpSasTOB[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTOB[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTOB[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTOB[ilay]->Fill(sqrt(rechitsaserrx[kk]));
	      meResSasTOB[ilay]->Fill(rechitsasres[kk]);
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
	}
      }
      if (detid.subdetId() == int(StripSubdetector::TID)){
	TIDDetId tidid(myid);
	int Tidnumrechitrphi    = numrechitrphi;
	int Tidnumrechitsas     = numrechitsas;
	int Tidnumrechitmatched = numrechitmatched;

	int ilay = tidid.ring() - 1; //for histogram filling
	
	if(tidid.stereo()==0){
	  for(int k = 0; k<Tidnumrechitrphi; k++){
	    meNstpRphiTID[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTID[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTID[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTID[ilay]->Fill(sqrt(rechitrphierrx[k]));
	    meResRphiTID[ilay]->Fill(rechitrphires[k]);
	  }
	} else  if(tidid.stereo()==1){
	  for(int kk = 0; kk < Tidnumrechitsas; kk++)	    
	    {
	      meNstpSasTID[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTID[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTID[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTID[ilay]->Fill(sqrt(rechitsaserrx[kk]));
	      meResSasTID[ilay]->Fill(rechitsasres[kk]);
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
	}
      }
      if (detid.subdetId() == int(StripSubdetector::TEC)){
	TECDetId tecid(myid);
	int Tecnumrechitrphi    = numrechitrphi;
	int Tecnumrechitsas     = numrechitsas;
	int Tecnumrechitmatched = numrechitmatched;

	int ilay = tecid.ring() - 1; //for histogram filling
	
	if(tecid.stereo()==0){
	  for(int k = 0; k<Tecnumrechitrphi; k++){
	    meNstpRphiTEC[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTEC[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTEC[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTEC[ilay]->Fill(sqrt(rechitrphierrx[k]));
	    meResRphiTEC[ilay]->Fill(rechitrphires[k]);
	  }
	} else  if(tecid.stereo()==1){
	  for(int kk = 0; kk < Tecnumrechitsas; kk++)	    
	    {
	      meNstpSasTEC[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTEC[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTEC[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTEC[ilay]->Fill(sqrt(rechitsaserrx[kk]));
	      meResSasTEC[ilay]->Fill(rechitsasres[kk]);
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
	}
      }




    }
  }

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


