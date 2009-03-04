#include "RecoTracker/SiTrackerMRHTools/test/DAFValidator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
/*
An analyzer suitable for multitrack 
fitting algorithm like DAF, MTF and EA
*/
using namespace std;
using namespace edm;
typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
 

int event=0;
DAFValidator::DAFValidator(const edm::ParameterSet& conf): theConf(conf){
	LogVerbatim("DAFValidator") << "Constructed a DAFValidator";
	string outfilename = theConf.getParameter<string>("OutputFileName");
	output = new TFile(outfilename.c_str(), "RECREATE");
}

void DAFValidator::beginRun(edm::Run & run, const edm::EventSetup& c){
       
  const bool oldAddDir = TH1::AddDirectoryStatus();
  TH1::AddDirectory(true);
  //histos 
  histo_weight = new TH1F("Weight", "weight of the mrh components ", 110, 0, 1.1);
  processtype_withassociatedsimhit_merged = 
    new TH1F("Process_Type_merged", "type of the process(merged_simhit)", 20, 0, 20);
  processtype_withassociatedsimhit=new TH1F("Process_Type", "type of the process", 20, 0, 20);
  Hit_Histo = new TH1F("Momentum_of_Hit", "momentum of the hit", 100, 0, 100); 
  MergedHisto = new TH1F("Histogram_Hit_Merged", "type of the hit", 5, 0, 5);
  NotMergedHisto = new TH1F("Histogram_Hit_NotMerged", "type of the hit", 5, 0, 5);
  weight_vs_processtype_merged = 
    new TH2F("WeightVsProcessType", "weight vs proc type", 20, 0, 20,  110, 0, 1.1 ); 
  weight_vs_processtype_notmerged = 
    new TH2F("WeightVsProcessTypeNotMerged", "weight vs proc type not merged", 20, 0, 20,  110, 0, 1.1 );
  pull_vs_weight = new TH2F("PullVsWeight", "pull vs weight", 100, 0, 20,  110, 0, 1.1 ); 
  Merged_vs_weight = new TH2F("HitsMergedVsWeight", "hits vs weight", 5, 0, 5,  110, 0, 1.1 );
  NotMerged_vs_weight = new TH2F("HitsNotMergedVsWeight", "hits vs weight", 5, 0, 5,  110, 0, 1.1 );
  NotMergedPos = new TH2F("BadHitsPositionNotMerged", "badposition", 600, -30, 30,  2400, -1200, 1200);
  MergedPos = new TH2F("BadHitsPositionMerged()", "badposition", 600, -30, 30,  2400, -1200, 1200);

  int bufsize = 64000;
  mrhit = new TTree("Ntuple","Ntuple");
  mrhit->Branch("mergedtype" , &mergedtype , "mergedtype/I" , bufsize);
  mrhit->Branch("notmergedtype" , &notmergedtype , "notmergedtype/I" , bufsize);
  mrhit->Branch("weight" , &weight , "weight/F" , bufsize);
  mrhit->Branch("detId" , &detId , "detId/I" , bufsize);
  mrhit->Branch("r" , &r , "r/F" , bufsize);
  mrhit->Branch("zeta" , &zeta , "zeta/F" , bufsize);
  mrhit->Branch("phi" , &phi , "phi/F" , bufsize);
  mrhit->Branch("hittyipe" , &hittype , "hittype/F" , bufsize);
  mrhit->Branch("event" , &nevent , "nevent/I" , bufsize);
   
  
  
  mrhit->Branch("hitlocalx" , &hitlocalx , "hitlocalx/F" , bufsize);
  mrhit->Branch("hitlocaly" , &hitlocaly , "hitlocaly/F" , bufsize);
  mrhit->Branch("hitlocalsigmax" , &hitlocalsigmax , "hitlocalsigmax/F" , bufsize);
  mrhit->Branch("hitlocalsigmay" , &hitlocalsigmay , "hitlocalsigmay/F" , bufsize);
  mrhit->Branch("hitlocalcov" , &hitlocalcov , "hitlocalcov/F" , bufsize);
  mrhit->Branch("tsoslocalx" , &tsoslocalx , "tsoslocalx/F" , bufsize);
  mrhit->Branch("tsoslocaly" , &tsoslocaly , "tsoslocaly/F" , bufsize);
  mrhit->Branch("tsoslocalsigmax" , &tsoslocalsigmax , "tsoslocalsigmax/F" , bufsize);
  mrhit->Branch("tsoslocalsigmay" , &tsoslocalsigmay , "tsoslocalsigmay/F" , bufsize);
  mrhit->Branch("tsoslocalcov" , &tsoslocalcov , "tsoslocalcov/F" , bufsize);
  mrhit->Branch("RecoTracknum" , &tsoslocalcov , "tsoslocalcov/F" , bufsize);
  mrhit->Branch("SimTracknum" , &tsoslocalcov , "tsoslocalcov/F" , bufsize);
  
  event=0;
  mergedtype=0;
  notmergedtype=0;

  TH1::AddDirectory(oldAddDir); 
  
}

DAFValidator::~DAFValidator(){}


void DAFValidator::analyze(const edm::Event& e, const edm::EventSetup& c){
  
  event++;
  //get tracker geometry
  edm::ESHandle<TrackerGeometry> tkgeom;
  c.get<TrackerDigiGeometryRecord>().get(tkgeom);
  //track associator
  string associatorName = theConf.getParameter<string>("TrackAssociator"); 
  edm::ESHandle<TrackAssociatorBase> associatorHandle;
  c.get<TrackAssociatorRecord>().get(associatorName,associatorHandle);
  
  //get the track collection
  edm::Handle<View<reco::Track> >  trackCollection;
  
  //get the association map between Traj and Track
  edm::Handle<TrajTrackAssociationCollection> assoMap;
  

  //get the trajectory collection
  edm::Handle<std::vector<Trajectory> >  trajectoryCollection;
  
  InputTag tracktag = theConf.getParameter<InputTag>("TrackCollection");
  e.getByLabel(tracktag, trackCollection);
  
  //  float RecoTracknum=trackCollection->size();
  
  e.getByLabel(tracktag, trajectoryCollection);
  //get the association map by label
  e.getByLabel(tracktag,assoMap);
  //get the trajectorycollection by label
 
  //get the tracking particle collection
  InputTag trackingParticleTag = theConf.getParameter<InputTag>("TrackingParticleCollection");	
  Handle<TrackingParticleCollection> trackingParticleCollection;
  
  e.getByLabel(trackingParticleTag, trackingParticleCollection);
  
  //  float SimTracknum=trackingParticleCollection->size();
  
  //hit associator
  TrackerHitAssociator hitAssociate(e,theConf.getParameter<ParameterSet>("HitAssociatorPSet"));

  //associate the tracking particles to the reco track
  reco::RecoToSimCollection RecsimColl=
    associatorHandle.product()->associateRecoToSim(trackCollection, 
						   trackingParticleCollection, &e);
    
  //loop over the recotrack looking for corresponding trackingparticle
  
  int i=0;   
  int nSimu=0;
  
  for(TrajTrackAssociationCollection::const_iterator it = assoMap->begin();it != assoMap->end(); ++it){ 
    
    std::map<reco::TrackRef,unsigned int> trackid;
    
    const edm::Ref<std::vector<Trajectory> > traj = it->key;
        
    const reco::TrackRef trackref = it->val;
    trackid.insert(make_pair(trackref,i));
    edm::RefToBase<reco::Track> track(trackCollection,i);
    i++;
    vector<pair<TrackingParticleRef, double> > simTracks;
    TrackingParticleRef matchedSimTrack;
    int nSim = 0;
    
    if(RecsimColl.find(track) != RecsimColl.end()){
      
      simTracks=RecsimColl[track];
      float fractionmax=0;
      for(vector<pair<TrackingParticleRef, double> >::const_iterator it = simTracks.begin(); it != simTracks.end(); ++it)
	{
	  TrackingParticleRef simTrack = it->first;
	  float fraction = it->second;
	  
	  //pick the trackingparticle with the highest fraction of hishared hits 
	  if(fraction > fractionmax)
	    { 
	      matchedSimTrack = simTrack; 
	      fractionmax=fraction;	    
	      nSim++; }
	}
      
      analyzeHits(matchedSimTrack.get(), track.get(), hitAssociate, traj, tkgeom.product(), event);
      nSimu++;
    }
  }
}

void DAFValidator::analyzeHits(const TrackingParticle* tpref, 
			       const reco::Track* rtref, 
			       TrackerHitAssociator& hitassociator, 
			       const edm::Ref<std::vector<Trajectory> > traj_iterator,
			       const TrackerGeometry* geom,
			       int event) {
  
  
  if (!tpref || !rtref) {
    cout << "something wrong: tpref = " << tpref << " rtref = " << rtref << endl;
    return;
  }
  
  //loop over the reco track rec hits, associate the simhits
  trackingRecHit_iterator iter;
  std::vector<TrajectoryMeasurement> measurements =traj_iterator->measurements();
  std::vector<TrajectoryMeasurement>::iterator traj_mes_iterator;
  
  for(traj_mes_iterator=measurements.begin();traj_mes_iterator!=measurements.end();traj_mes_iterator++){
    const TrackingRecHit* ttrh=traj_mes_iterator->recHit()->hit();
    
    const SiTrackerMultiRecHit* mrh = dynamic_cast<const SiTrackerMultiRecHit*>(ttrh);
    const TrackingRecHit* rechit=0;
    float maxweight = 0;
    if (mrh){
      vector<const TrackingRecHit*> components = mrh->recHits();
      vector<const TrackingRecHit*>::const_iterator icomp;
      int hitcounter=0;
      for (icomp = components.begin(); icomp != components.end(); icomp++)
	{
	  if((*icomp)->isValid())
	    {
	      cout << "weight: " << mrh->weight(hitcounter) << endl;
	      //extract the hit with the max weight from the multirechit 
	      weight = mrh->weight(hitcounter);
	      if(weight > maxweight) {
		rechit=*icomp;
		maxweight=weight;
	      }
	    }
	  
	  hitcounter++;
	}
    }
    
    else{
      if(ttrh->isValid()){
	rechit=ttrh;
	maxweight=1;
      }
    }
    
    if(rechit){
      
      if (getType(rechit)==2.)
	{
	  std::vector<const TrackingRecHit*> hits = rechit->recHits();
	  
	  for(std::vector<const TrackingRecHit*>::iterator iterhits=hits.begin();iterhits!=hits.end();iterhits++)
	    {
	      const TrackingRecHit* rechit1 = *iterhits;
	      
	      LocalPoint pos;
	      if(rechit1->isValid()) pos=rechit1->localPosition();
	      //unsigned int detid=rechit->geographicalId().rawId();
	      
	      TrajectoryStateOnSurface tsos = traj_mes_iterator->updatedState();
	      
	      AlgebraicVector tsospos(2);
	      tsospos[0]=tsos.localPosition().x();
	      tsospos[1]=tsos.localPosition().y();
	      
	      AlgebraicVector hitposition(2);
	      hitposition[0]=pos.x();
	      hitposition[1]=pos.y();
	      
	      AlgebraicVector tsoserr(3); 
	      tsoserr[0] = tsos.localError().positionError().xx();
	      tsoserr[1] = tsos.localError().positionError().yy();
	      tsoserr[2] = tsos.localError().positionError().xy();
	      
	      AlgebraicVector hiterr(3); 
	      hiterr[0] = rechit1->localPositionError().xx();
	      hiterr[1] = rechit1->localPositionError().yy();
	      hiterr[2] = rechit1->localPositionError().xy();
	      
	      tsoslocalx = tsospos[0];
	      tsoslocaly = tsospos[1];
	      
	      hitlocalx = hitposition[0];
	      hitlocaly = hitposition[1];
	      
	      tsoslocalsigmax = tsoserr[0];
	      tsoslocalsigmay = tsoserr[1];
	      tsoslocalcov = tsoserr[2]; 
	      
	      hitlocalsigmax = hiterr[0];
	      hitlocalsigmay = hiterr[1];
	      hitlocalcov = hiterr[2];	
	      
	      nevent=event;
	      weight=maxweight;
	      GlobalPoint point=getGlobalPosition(rechit1,geom);
	      r=point.perp();
	      zeta=point.z();
	      phi=point.phi();
	      hittype=getType(rechit1);
	      detId=rechit->geographicalId().rawId();
	      //do the association between rechits and simhits 
	      vector<PSimHit> matchedhits = hitassociator.associateHit(*rechit1);     
	      vector<SimHitIdpr> simhitids = hitassociator.associateHitId(*rechit1);      
	      fillDAFHistos(matchedhits, maxweight, rechit1, geom);
	      fillPHistos(matchedhits);
	    
	      if(matchedhits.size()!=1){
		notmergedtype=0;
		mergedtype=fillMergedHisto(simhitids,matchedhits,tpref,maxweight,geom);
		
	      }
	      
	      else{
		
		mergedtype=0;
		notmergedtype=fillNotMergedHisto(simhitids,matchedhits,tpref,maxweight,geom);
		
	      }
	      
	      mrhit->Fill();
	    }
	}
      
      else {
	LocalPoint pos;
	if(rechit->isValid()) pos=rechit->localPosition();
	//unsigned int detid=rechit->geographicalId().rawId();
	
	TrajectoryStateOnSurface tsos = traj_mes_iterator->updatedState();
	
	AlgebraicVector tsospos(2);
	tsospos[0]=tsos.localPosition().x();
	tsospos[1]=tsos.localPosition().y();
	
	AlgebraicVector hitposition(2);
	hitposition[0]=pos.x();
	hitposition[1]=pos.y();
      
	AlgebraicVector tsoserr(3); 
	tsoserr[0] = tsos.localError().positionError().xx();
	tsoserr[1] = tsos.localError().positionError().yy();
	tsoserr[2] = tsos.localError().positionError().xy();
	
	AlgebraicVector hiterr(3); 
	hiterr[0] = rechit->localPositionError().xx();
	hiterr[1] = rechit->localPositionError().yy();
	hiterr[2] = rechit->localPositionError().xy();
	
	tsoslocalx = tsospos[0];
	tsoslocaly = tsospos[1];
	
	hitlocalx = hitposition[0];
	hitlocaly = hitposition[1];
	
	tsoslocalsigmax = tsoserr[0];
	tsoslocalsigmay = tsoserr[1];
	tsoslocalcov = tsoserr[2]; 
	
	hitlocalsigmax = hiterr[0];
	hitlocalsigmay = hiterr[1];
	hitlocalcov = hiterr[2];	
	
	nevent=event;
	weight=maxweight;
	GlobalPoint point=getGlobalPosition(rechit,geom);
	r=point.perp();
	zeta=point.z();
	phi=point.phi();
	hittype=getType(rechit);
	detId=rechit->geographicalId().rawId();
	//do the association between rechits and simhits 
	vector<PSimHit> matchedhits = hitassociator.associateHit(*rechit);
	vector<SimHitIdpr> simhitids = hitassociator.associateHitId(*rechit);
	
	fillDAFHistos(matchedhits, maxweight, rechit, geom);
	fillPHistos(matchedhits);
	
	if(matchedhits.size()!=1){
	  notmergedtype=0;
	  mergedtype=fillMergedHisto(simhitids,matchedhits,tpref,maxweight,geom);
	  
	}
	
	else{
	  
	  mergedtype=0;
	  notmergedtype=fillNotMergedHisto(simhitids,matchedhits,tpref,maxweight,geom);
	  
	}
	
	mrhit->Fill();
	
      }
      
    }
    
  }
  
}

int DAFValidator::fillNotMergedHisto(const vector<SimHitIdpr>& simhitids,
				     const vector<PSimHit>& simhits,
				     const TrackingParticle* tpref,
				     float weight,
				     const TrackerGeometry* geom) const 
{
  
  
  if (simhitids.empty()) {cout << "something wrong" << endl;}
  
  vector<PSimHit>::const_iterator isimid = simhits.begin();
  //int simcount=0; 
  if ( isimid->processType() == 2)
    {
      
      for (TrackingParticle::g4t_iterator g4T = tpref -> g4Track_begin(); g4T !=  tpref -> g4Track_end(); ++g4T){
	//vector<SimHitIdpr>::const_iterator isimid;
	//	int simcount=0;
	if ((*g4T).trackId()==isimid->trackId()) return 3;
	
      }

    }
  
  else {
    for (TrackingParticle::g4t_iterator g4T = tpref -> g4Track_begin(); g4T !=  tpref -> g4Track_end(); ++g4T){
      //vector<SimHitIdpr>::const_iterator isimid;
      //      int simcount=0;
      if ((*g4T).trackId()==isimid->trackId()) return 2;
      
    }
  }
  
  return 1;
}


int DAFValidator::fillMergedHisto(const vector<SimHitIdpr>& simhitids,
				  const vector<PSimHit>& simhits,
				  const TrackingParticle* tpref,
				  float weight,
				  const TrackerGeometry* geom) const 
{
  
  if (simhitids.empty()) {cout << "something wrong" << endl;}
  GlobalPoint point;
  //unsigned int simcount=0;
  for (TrackingParticle::g4t_iterator g4T = tpref -> g4Track_begin(); g4T !=  tpref -> g4Track_end(); ++g4T){
    //vector<SimHitIdpr>::const_iterator isimid;
    vector<PSimHit>::const_iterator isimid;
    //    unsigned int simcount=0;
    //for (isimid = simhitids.begin(); isimid != simhitids.end(); isimid++){
    //in case of merged hits we have to make a for cicle 
    for(isimid = simhits.begin(); isimid != simhits.end(); isimid++){
      
      if (((*g4T).trackId() == (*isimid).trackId()) || (isimid->processType() == 2)){ 
	
       return 2;
       
      }
      
      else continue;
      
    }
    
  }
  
  return 1;
  
}



bool DAFValidator::check(const vector<PSimHit>& simhits, const TrackingParticle* tpref) const {
  
  if (simhits.empty()) return false;
  vector<PSimHit>::const_iterator itp;	
  for (itp = tpref->pSimHit_begin(); itp < tpref->pSimHit_end(); itp++){
    vector<PSimHit>::const_iterator isim;
    for (isim = simhits.begin(); isim != simhits.end(); isim++){
      if (itp->detUnitId() == isim->detUnitId() && 
	  (itp->localPosition()-isim->localPosition()).mag() < 1e-4) { return true; }
    }
  }
  return false;
}

bool DAFValidator::check(const vector<SimHitIdpr>& simhitids, const TrackingParticle* tpref) const {
  if (simhitids.empty()) return false;
  for (TrackingParticle::g4t_iterator g4T = tpref -> g4Track_begin(); g4T !=  tpref -> g4Track_end(); ++g4T){
    vector<SimHitIdpr>::const_iterator isimid;
    for (isimid = simhitids.begin(); isimid != simhitids.end(); isimid++){
      if ((*g4T).trackId() == (*isimid).first && tpref->eventId() == (*isimid).second){
	return true; 
      }	
    }
  }
  return false; 
}



void DAFValidator::fillPHistos(vector<PSimHit>& components){
  //check the hit validity
  if (!components.size()){
    edm::LogError("DAFValidator") << "empty rechit vector: this multirechit has no hits";
    return;
  }
  
  
  
  for(vector<PSimHit>::iterator icomp=components.begin(); icomp!=components.end(); icomp++ )
    {
      float pabs = icomp->pabs();
      Hit_Histo->Fill(pabs);
      
    }
  
  
  
}


void DAFValidator::fillDAFHistos(vector<PSimHit>& matched, 
				 float weight, 
				 const TrackingRecHit* rechit,
				 const TrackerGeometry* geom){
  //check the hit validity
  if (!matched.size()){
    edm::LogError("DAFValidator") << "empty simhit vector: this multirechit has no corresponding simhits";
    return;
  }
  
  unsigned short ptype;
  
  if (matched.size()==1)
    {
      
      float pull=calculatepull(rechit, matched.front(), geom);
      pull_vs_weight->Fill(pull,weight);
      
      ptype = matched.front().processType();
      processtype_withassociatedsimhit->Fill(ptype);
      weight_vs_processtype_notmerged->Fill(ptype, weight);
      
    }
  
  else if (matched.size()>1)
    {
 
      for(vector<PSimHit>::iterator imatched=matched.begin(); imatched!=matched.end(); imatched++ )
	{
	  float pull=calculatepull(rechit, (*imatched), geom);
	  pull_vs_weight->Fill(pull,weight);
	  
	  ptype = imatched->processType();
	  processtype_withassociatedsimhit_merged->Fill(ptype);
	  weight_vs_processtype_merged->Fill(ptype, weight); 
	}
    }
  
  histo_weight->Fill(weight); 
  
}

float DAFValidator::calculatepull(const TrackingRecHit* hit, 
				  PSimHit simhit,
				  const TrackerGeometry* geom){
  
  //perform the calculation of the pull
  AlgebraicVector reccoor(2);
  AlgebraicVector simcoor(2);
  AlgebraicVector diffcoor(2);
  
  //reccoor[0] = getGlobalPositionRec(hit, geom).x();
  //reccoor[1] = getGlobalPositionRec(hit, geom).y();
  
  //simcoor[0] = getGlobalPositionSim(simhit, geom).x();
  //simcoor[1] = getGlobalPositionSim(simhit, geom).y();
  
  reccoor[0] = hit->localPosition().x();
  reccoor[1] = hit->localPosition().y();
  
  simcoor[0] = simhit.localPosition().x();
  simcoor[1] = simhit.localPosition().y();
  
  diffcoor = reccoor-simcoor;
  float diff = sqrt(diffcoor[0]*diffcoor[0]+diffcoor[1]*diffcoor[1]);
  
  float sigma = sqrt((hit->localPositionError().xx() + hit->localPositionError().yy()+ hit->localPositionError().xy()));
  float pull = diff/sigma;
  
  return pull;
  //return diff;
}

void DAFValidator::fillMultiHitHistos(float weight, const TrackingRecHit* hit, const TrackerGeometry* geom){
  //check the hit validity
  if (!hit->isValid()){
    edm::LogError("DAFValidator") << "Invalid hit!";
    return;
  }
  
  //transform the rec hit in global position
  GlobalPoint global = getGlobalPosition(hit, geom);	
  //fill the histos in r and eta
  weight_withassociatedsimhit_vs_r->Fill(global.perp(), weight);
  weight_withassociatedsimhit_vs_eta->Fill(global.eta(), weight);
}








void DAFValidator::fillMultiHitHistosPartiallyUnmatched(const vector<pair<float,const TrackingRecHit*> >& map, const TrackerGeometry* geom){
  vector<pair<float,const TrackingRecHit*> >::const_iterator imap;
  for (imap = map.begin(); imap != map.end(); imap++){
    //weight_partially_unmatched->Fill(imap->first);
    //weight_vs_type_partially_unmatched->Fill(getType(imap->second), imap->first);
    GlobalPoint global = getGlobalPosition(imap->second, geom);			
    //weight_vs_r_partially_unmatched->Fill(global.perp(), imap->first);
    //weight_vs_eta_partially_unmatched->Fill(global.eta(), imap->first);
  }
}

void DAFValidator::fillMultiHitHistosTotallyUnmatched(const vector<pair<float,const TrackingRecHit*> >& map, const TrackerGeometry* geom){
  vector<pair<float,const TrackingRecHit*> >::const_iterator imap;
  for (imap = map.begin(); imap != map.end(); imap++){
    //weight_totally_unmatched->Fill(imap->first);
    //weight_vs_type_totally_unmatched->Fill(getType(imap->second), imap->first);
    GlobalPoint global = getGlobalPosition(imap->second, geom);
    //weight_vs_r_totally_unmatched->Fill(global.perp(), imap->first);
    //weight_vs_eta_totally_unmatched->Fill(global.eta(), imap->first);
  }
}

float DAFValidator::getType(const TrackingRecHit* hit)  const {
  if (!hit->isValid()){
    throw cms::Exception("DAFValidator") << "This hit is invalid, cannot be casted as any type of tracker hit (strip or pixel)! ";
  }
  const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit);
  const SiStripRecHit2D* stripmono = dynamic_cast<const SiStripRecHit2D*>(hit);
  const SiStripMatchedRecHit2D* stripmatched = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
  const ProjectedSiStripRecHit2D* stripprojected = dynamic_cast<const ProjectedSiStripRecHit2D*>(hit);
  if(pixhit) return 0.;
  else if (stripmono) return 1.;
  else if (stripmatched) return 2.;
  else if (stripprojected) return 3.;
  else throw cms::Exception("DAFValidator") << "Rec Hits of type " << typeid(*hit).name() << " should not be present at this stage ";
}

void DAFValidator::endJob(){
  output->Write();
  output->Close();
}


GlobalPoint DAFValidator::getGlobalPositionSim(const PSimHit hit, const TrackerGeometry* geom) const{
  
  cout << "detid" << hit << endl;
  DetId detid = DetId(hit.detUnitId());

  //const GeomDet* gdet = geom->idToDet(hit.detUnitId());
  const GeomDet* gdet = geom->idToDet(detid);
  GlobalPoint global = gdet->toGlobal(hit.localPosition());
  return global;
}

GlobalPoint DAFValidator::getGlobalPosition(const TrackingRecHit* hit, const TrackerGeometry* geom) const{
  const GeomDet* gdet = geom->idToDet(hit->geographicalId());
  GlobalPoint global = gdet->toGlobal(hit->localPosition());
  return global;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
DEFINE_ANOTHER_FWK_MODULE(DAFValidator);
