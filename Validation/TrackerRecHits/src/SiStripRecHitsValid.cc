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
  
  //one histo per Layer rphi hits
  for(int i = 0 ;i<4 ; i++) {
    Char_t histo[200];
    sprintf(histo,"Nstp_rphi_layer%tib",i+1);
    meNstpRphiTIB[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_rphi_layer%tib",i+1);
    meAdcRphiTIB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_rphi_layer%tib",i+1);
    mePosxRphiTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_rphi_layer%tib",i+1);
    meErrxRphiTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100, -6.0,+6.0);  
    sprintf(histo,"Res_rphi_layer%tib",i+1);
    meResRphiTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  
  }

  //one histo per Layer stereo and matched hits
  for(int i = 0 ;i<2 ; i++) {

    Char_t histo[200];
    sprintf(histo,"Nstp_sas_layer%tib",i+1);
    meNstpSasTIB[i] = dbe_->book1D(histo,"RecHit Cluster Size",20,0.5,20.5);  
    sprintf(histo,"Adc_sas_layer%tib",i+1);
    meAdcSasTIB[i] = dbe_->book1D(histo,"RecHit Cluster Charge",100,0.,300.);  
    sprintf(histo,"Posx_sas_layer%tib",i+1);
    mePosxSasTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0,+6.0);  
    sprintf(histo,"Errx_sas_layer%tib",i+1);
    meErrxSasTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100, -6.0,+6.0);  
    sprintf(histo,"Res_sas_layer%tib",i+1);
    meResSasTIB[i] = dbe_->book1D(histo,"RecHit Residual",100,-0.5,+0.5);  

    sprintf(histo,"Posx_matched_layer%tib",i+1);
    mePosxMatchedTIB[i] = dbe_->book1D(histo,"RecHit x coord.",100,-6.0, +6.0);  
    sprintf(histo,"Posy_matched_layer%tib",i+1);
    mePosyMatchedTIB[i] = dbe_->book1D(histo,"RecHit y coord.",100,-6.0, +6.0);  
    sprintf(histo,"Errx_matched_layer%tib",i+1);
    meErrxMatchedTIB[i] = dbe_->book1D(histo,"RecHit err(x) coord.",100,-6.0, +6.0);  
    sprintf(histo,"Erry_matched_layer%tib",i+1);
    meErryMatchedTIB[i] = dbe_->book1D(histo,"RecHit err(y) coord.",100,-6.0, +6.0);  
    sprintf(histo,"Resx_matched_layer%tib",i+1);
    meResxMatchedTIB[i] = dbe_->book1D(histo,"RecHit Res(x) coord.",100,-6.0, +6.0);  
    sprintf(histo,"Resy_matched_layer%tib",i+1);
    meResyMatchedTIB[i] = dbe_->book1D(histo,"RecHit Res(y) coord.",100,-6.0, +6.0);  
  }

  
  //etc add other histograms... 
  
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
  
  //--- get SimHits
  
  // Step A: Get Inputs
  theStripHits.clear();
  edm::Handle<edm::PSimHitContainer> TIBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TIBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TIDHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TIDHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TECHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TECHitsHighTof;
  
  e.getByLabel("SimG4Object","TrackerHitsTIBLowTof", TIBHitsLowTof);
  e.getByLabel("SimG4Object","TrackerHitsTIBHighTof", TIBHitsHighTof);
  e.getByLabel("SimG4Object","TrackerHitsTIDLowTof", TIDHitsLowTof);
  e.getByLabel("SimG4Object","TrackerHitsTIDHighTof", TIDHitsHighTof);
  e.getByLabel("SimG4Object","TrackerHitsTOBLowTof", TOBHitsLowTof);
  e.getByLabel("SimG4Object","TrackerHitsTOBHighTof", TOBHitsHighTof);
  e.getByLabel("SimG4Object","TrackerHitsTECLowTof", TECHitsLowTof);
  e.getByLabel("SimG4Object","TrackerHitsTECHighTof", TECHitsHighTof);
  
  theStripHits.insert(theStripHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end()); 
  theStripHits.insert(theStripHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
  theStripHits.insert(theStripHits.end(), TIDHitsLowTof->begin(), TIDHitsLowTof->end()); 
  theStripHits.insert(theStripHits.end(), TIDHitsHighTof->begin(), TIDHitsHighTof->end());
  theStripHits.insert(theStripHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end()); 
  theStripHits.insert(theStripHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
  theStripHits.insert(theStripHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end()); 
  theStripHits.insert(theStripHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());
  
  SimHitMap.clear();
  for (std::vector<PSimHit>::iterator isim = theStripHits.begin();
       isim != theStripHits.end(); ++isim){
    SimHitMap[(*isim).detUnitId()].push_back((*isim));
  }
  
  int numrechitrphi   =0;
  int numrechitsas    =0;
  int numsimhit       =0;
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
      simhitx[i] =0.;
      simhity[i] =0.;
      simhitz[i] =0.;
      simhitphi[i]=0;
      simhiteta[i]=0;
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
      rechitmatchedresx[i]=0;
      rechitmatchedresy[i]=0;
      
    }
    
    //---get simhit
    numsimhit=0;
    std::map<unsigned int, std::vector<PSimHit> >::const_iterator it = SimHitMap.find(myid);
    vector<PSimHit> simHit; 
    simHit.clear();
    if (it!= SimHitMap.end()){
      simHit = it->second;
      vector<PSimHit>::const_iterator simHitIter = simHit.begin();
      vector<PSimHit>::const_iterator simHitIterEnd = simHit.end();
      numsimhit = simHit.size();
      int i=0;
      for (;simHitIter != simHitIterEnd; ++simHitIter) {
	const PSimHit ihit = *simHitIter;
	LocalPoint simhitpos = ihit.localPosition();
	simhitx[i] = simhitpos.x();
	simhity[i] = simhitpos.y();
	simhitz[i] = simhitpos.z();
	simhitphi[i] = simhitpos.phi();
	simhiteta[i] = simhitpos.eta();
	i++;
      }
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
	const std::vector<const SiStripCluster*> clust=rechit.cluster();
	int clusiz=0;
	int totcharge=0;
	for(vector<const SiStripCluster*>::const_iterator ic = clust.begin(); ic!=clust.end(); ic++) {
	  clusiz = (*ic)->amplitudes().size();
	  const std::vector<short> amplitudes=(*ic)->amplitudes();
	  for(size_t i=0; i<amplitudes.size();i++){
	    totcharge+=amplitudes[i];
	  }
	}
	rechitrphix[i] = position.x();
	rechitrphiy[i] = position.y();
	rechitrphiz[i] = position.z();
	rechitrphierrx[i] = error.xx();
	clusizrphi[i] = clusiz;
	cluchgrphi[i] = totcharge;

	matched.clear();
	matched = associate.associateHit(rechit);
	if(!matched.empty()){
	  rechitrphires[i] = rechitrphix[i] - matched[0].localPosition().x();
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
	std::vector<const SiStripCluster*> clust=rechit.cluster();
	int clusiz=0;
	int totcharge=0;
	for(vector<const SiStripCluster*>::const_iterator ic = clust.begin(); ic!=clust.end(); ic++) {
	  clusiz = (*ic)->amplitudes().size();
	  const std::vector<short> amplitudes=(*ic)->amplitudes();
	  for(size_t i=0; i<amplitudes.size();i++){
	    totcharge+=amplitudes[i];
	  }
	}
	rechitsasx[j] = position.x();
	rechitsasy[j] = position.y();
	rechitsasz[j] = position.z();
	rechitsaserrx[j] = error.xx();
	clusizsas[j] = clusiz;
	cluchgsas[j] = totcharge;
	
	matched.clear();
	matched = associate.associateHit(rechit);
	if(!matched.empty()){
	  rechitsasres[j] = rechitsasx[j] - matched[0].localPosition().x();
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
	
	//	cout << "SiStripRecHitsValid ---> try association matched! " << endl;
	matched.clear();
	const SiStripRecHit2DLocalPos *mono = rechit.monoHit();
	const SiStripRecHit2DLocalPos *st = rechit.stereoHit();
	LocalPoint monopos = mono->localPosition();
	LocalPoint stpos   = st->localPosition();
	
	rechitmatchedx[j] = position.x();
	rechitmatchedy[j] = position.y();
	rechitmatchedz[j] = position.z();
	//rechitmatchedphi[j] = position.phi();
	rechitmatchederrxx[j] = error.xx();
	rechitmatchederrxy[j] = error.xy();
	rechitmatchederryy[j] = error.yy();
	matched = associate.associateHit(*st);
	if(!matched.empty()){
	  cout << " detector = " << myid << " #match = " << matched.size() << endl;
	  cout << " Matched x = " << position.x() << " y = "<< position.y() << " z = " << position.z() << endl;
	  cout << " Mono    x = " << monopos.x() << " y = "<< monopos.y() << " z = " << monopos.z() << endl;
	  cout << " Stereo  x = " << stpos.x() << " y = "<< stpos.y() << " z = " << stpos.z() << endl;
	  
	  for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	    cout << " hit  ID = " << (*m).trackId() << " Simhit x = " << (*m).localPosition().x() 
		 << " y = " <<  (*m).localPosition().y() << " z = " <<  (*m).localPosition().x() << endl;
	  }
	  //project simhit;
	  const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(rechit.geographicalId());
	  const StripGeomDetUnit* partnerstripdet =(StripGeomDetUnit*) gluedDet->stereoDet();
	  std::pair<LocalPoint,LocalVector> hitPair= projectHit(matched[0],partnerstripdet,gluedDet->surface());
	  //	    rechitmatchedresx[j] = rechitmatchedx[j] - matched[0].localPosition().x();
	  // rechitmatchedresy[j] = rechitmatchedy[j] - matched[0].localPosition().y();
	  rechitmatchedresx[j] = rechitmatchedx[j] - hitPair.first.x();
	  rechitmatchedresy[j] = rechitmatchedy[j] - hitPair.first.y();
	  
	  cout << " res x = " << rechitmatchedresx[j] << " rec(x) = " <<  rechitmatchedx[j] 
	    //		 << " sim(x) = " << matched[0].localPosition().x() << endl;
	       << " sim(x) = " << hitPair.first.x() << endl;
	  cout << " res y = " << rechitmatchedresy[j] << " rec(y) = " <<  rechitmatchedy[j] 
	    //		 << " sim(x) = " << matched[0].localPosition().y() << endl;
	       << " sim(y) = " <<  hitPair.first.y()<< endl;
	}
	
	j++;
      }
    }
    

    if(numrechitrphi>0 || numrechitsas>0 ){
      cout << " N(simhit)= " << numsimhit << "  N(rphi) = " << numrechitrphi << "  N(sas) = " << numrechitsas 
	   << " N(matched) = " << numrechitmatched << endl; 
      // 	if(numrechitmatched>0){ 
      // 	  std::cout << " det id= " << myid << " N(simhit) = " << numsimhit 
      // 		    << " N(rechitrphi) = " << numrechitrphi << " N(rechitsas)= " << numrechitsas 
      // 		    << " N(matched) = " << numrechitmatched << std::endl;      
      // 	}
      
      if (detid.subdetId() == int(StripSubdetector::TIB)){
	TIBDetId tibid(myid);
	int Tibnumrechitrphi    = numrechitrphi;
	int Tibnumrechitsas     = numrechitsas;
	int Tibnumrechitmatched = numrechitmatched;
	int ilay = tibid.layer();
	cout << " inside tib layer = " <<ilay << " stereo = " << tibid.stereo() << endl;
	
	if(tibid.stereo()==0){
	  for(int k = 0; k<Tibnumrechitrphi; k++){
	    meNstpRphiTIB[ilay]->Fill(clusizrphi[k]);
	    meAdcRphiTIB[ilay]->Fill(cluchgrphi[k]);
	    mePosxRphiTIB[ilay]->Fill(rechitrphix[k]);
	    meErrxRphiTIB[ilay]->Fill(rechitrphierrx[k]);
	    meResRphiTIB[ilay]->Fill(rechitrphires[k]);
	  }
	} else  if(tibid.stereo()==1){
	  for(int kk = 0; kk<Tibnumrechitsas; kk++)
	    {
	      meNstpSasTIB[ilay]->Fill(clusizsas[kk]);
	      meAdcSasTIB[ilay]->Fill(cluchgsas[kk]);
	      mePosxSasTIB[ilay]->Fill(rechitsasx[kk]);
	      meErrxSasTIB[ilay]->Fill(rechitsaserrx[kk]);
	      meResSasTIB[ilay]->Fill(rechitsasres[kk]);
	    }	  
	  for(int kkk = 0; kkk<Tibnumrechitmatched; kkk++)
	    {
	      mePosxMatchedTIB[ilay]->Fill(rechitmatchedx[kkk]);
	      mePosyMatchedTIB[ilay]->Fill(rechitmatchedy[kkk]);
	      meErrxMatchedTIB[ilay]->Fill(rechitmatchederrxx[kkk]);
	      meErryMatchedTIB[ilay]->Fill(rechitmatchederryy[kkk]);
	      meResxMatchedTIB[ilay]->Fill(rechitmatchedresx[kkk]);
	      meResyMatchedTIB[ilay]->Fill(rechitmatchedresy[kkk]);
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


