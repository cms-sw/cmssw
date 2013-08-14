#include "RecoTracker/SingleTrackPattern/test/CombTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>

CombTrack::CombTrack(const edm::ParameterSet& iConfig) : 
  conf_(iConfig)
{

}


CombTrack::~CombTrack()
{

}


void
CombTrack::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  edm::InputTag MuTag = conf_.getParameter<edm::InputTag>("cosmicMuTracks");
  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("cosmicTkTracks");
  Handle<reco::TrackCollection> Tk_trackCollection;
  iEvent.getByLabel(TkTag,Tk_trackCollection);
  
  Handle<reco::TrackCollection> Mu_trackCollection;
  iEvent.getByLabel(MuTag,Mu_trackCollection);
  
  Handle<TrackingRecHitCollection> trackrechitCollection;
  iEvent.getByLabel(TkTag,trackrechitCollection);
  edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
  if (trinevents){
    iEvent.getByLabel(TkTag,TrajectoryCollection);
  }

  const   reco::TrackCollection *tk_tracks=Tk_trackCollection.product();
  const   reco::TrackCollection *mu_tracks=Mu_trackCollection.product();
  vector<const reco::Track*> DTmuons;
  if ((tk_tracks->size()>0)&&(mu_tracks->size()>0)){

    AnalHits(*trackrechitCollection,iSetup);
    reco::TrackCollection::const_iterator itr=Tk_trackCollection.product()->begin();
    reco::TrackCollection::const_iterator imu;
    
    for (imu=Mu_trackCollection.product()->begin();
	 imu!=Mu_trackCollection.product()->end();imu++) DTmuons.push_back(&(*imu));
    
    stable_sort(DTmuons.begin(),DTmuons.end(),CompareChi2());
    const reco::Track *imubeg =*(DTmuons.begin());
    float mupt=sqrt(imubeg->innerMomentum().Perp2());
 
    
    hpt->Fill((*itr).pt(),mupt);
    hpz->Fill((*itr).pz(),imubeg->innerMomentum().Z());
    hq->Fill((*itr).charge(),imubeg->charge());
    hphi->Fill((*itr).outerPhi(),imubeg->innerMomentum().Phi());
    heta->Fill((*itr).outerEta(),imubeg->innerMomentum().Eta());

    if (trinevents){
      edm::ESHandle<GlobalTrackingGeometry> geo;
      iSetup.get<GlobalTrackingGeometryRecord>().get(geo);    
      iSetup.get<IdealMagneticFieldRecord>().get(magfield);
      const Trajectory tt=*(TrajectoryCollection.product()->begin());
      bool seedplus=(tt.seed().direction()==alongMomentum);
      const BoundPlane &bp=geo->idToDet((*imubeg->recHitsBegin())->geographicalId())->specificSurface();
      if((&bp)!=0){


	if (seedplus){
	  TrajectoryStateOnSurface tsos=tt.firstMeasurement().updatedState();
	  AnalyticalPropagator *thePropagator=  new AnalyticalPropagator(&(*magfield),alongMomentum );
	  TrajectoryStateOnSurface prSt=thePropagator->propagate(tsos,bp);

	  delete thePropagator;
	}else{
	  TrajectoryStateOnSurface tsos=tt.lastMeasurement().updatedState();
	  AnalyticalPropagator *thePropagator=  new AnalyticalPropagator(&(*magfield),alongMomentum );
	  TrajectoryStateOnSurface prSt=thePropagator->propagate(tsos,bp);
	  delete thePropagator;
	}
      }
    }
    if (nlay>2){
      hpt_3l->Fill((*itr).pt(),imubeg->pt());
      hpz_3l->Fill((*itr).pz(),imubeg->pz());
      hq_3l->Fill((*itr).charge(),imubeg->charge());
      hphi_3l->Fill((*itr).outerPhi(),imubeg->outerPhi());
      heta_3l->Fill((*itr).outerEta(),imubeg->outerEta());
      if(ltob2&&ltob1){
	hpt_tob->Fill((*itr).pt(),imubeg->pt());
	hpz_tob->Fill((*itr).pz(),imubeg->pz());
	hq_tob->Fill((*itr).charge(),imubeg->charge());
	hphi_tob->Fill((*itr).outerPhi(),imubeg->outerPhi());
	heta_tob->Fill((*itr).outerEta(),imubeg->outerEta());
      }
      if (nlay>3){
	hpt_4l->Fill((*itr).pt(),imubeg->pt());
	hpz_4l->Fill((*itr).pz(),imubeg->pz());
	hq_4l->Fill((*itr).charge(),imubeg->charge());
	hphi_4l->Fill((*itr).outerPhi(),imubeg->outerPhi());
	heta_4l->Fill((*itr).outerEta(),imubeg->outerEta());
      }
    }
    if(!(ltob1 || ltob2)){
      hpt_tib->Fill((*itr).pt(),imubeg->pt());
      hpz_tib->Fill((*itr).pz(),imubeg->pz());
      hq_tib->Fill((*itr).charge(),imubeg->charge());
      hphi_tib->Fill((*itr).outerPhi(),imubeg->outerPhi());
      heta_tib->Fill((*itr).outerEta(),imubeg->outerEta());
    }
  }
  
}
// ------------ method called once each job just before starting event loop  ------------
void 
CombTrack::beginRun(edm::Run & run, const edm::EventSetup&)
{
  hFile = new TFile ( "combtrack.root", "RECREATE" );
  hpt =new TH2F("hpt","pt correlation",25,0,50,25,0,50);
  hpz =new TH2F("hpz","pz correlation",15,-5,25,15,-5,25);
  hq = new TH2F("hq","charge correlation",2,-1.5,1.5,2,-1.5,1.5);
  heta = new TH2F("heta","eta correlation",50,-0.5,1.5,50,-0.5,1.5);
  hphi = new TH2F("hphi","phi correlation",50,-2.2,-1.0,50,-2.2,-1.0);

  hpt_tob =new TH2F("hpt_tob","pt correlation (2 TOB layers) ",10,0,50,10,0,50);
  hpz_tob =new TH2F("hpz_tob","pz correlation (2 TOB layers)",10,-5,25,10,-5,25);
  hq_tob = new TH2F("hq_tob","charge correlation (2 TOB layers)",2,-1.5,1.5,2,-1.5,1.5);
  heta_tob = new TH2F("heta_tob","eta correlation (2 TOB layers)",50,-0.5,1.5,50,-0.5,1.5);
  hphi_tob = new TH2F("hphi_tob","phi correlation (2 TOB layers)",50,-2.2,-1.0,50,-2.2,-1.0);
  hpt_tib =new TH2F("hpt_tib","pt correlation (hits only in TIB)",10,0,50,10,0,50);
  hpz_tib =new TH2F("hpz_tib","pz correlation (hits only in TIB)",10,-5,25,10,-5,25);
  hq_tib = new TH2F("hq_tib","charge correlation (hits only in TIB)",2,-1.5,1.5,2,-1.5,1.5);
  heta_tib = new TH2F("heta_tib","eta correlation (hits only in TIB)",50,-0.5,1.5,50,-0.5,1.5);
  hphi_tib = new TH2F("hphi_tib","phi correlation (hits only in TIB)",50,-2.2,-1.0,50,-2.2,-1.0);
  hpt_3l =new TH2F("hpt_3l","pt correlation (3 layers)",10,0,50,10,0,50);
  hpz_3l =new TH2F("hpz_3l","pz correlation (3 layers)",10,-5,25,10,-5,25);
  hq_3l = new TH2F("hq_3l","charge correlation (3 layers)",2,-1.5,1.5,2,-1.5,1.5);
  heta_3l = new TH2F("heta_3l","eta correlation (3 layers)",50,-0.5,1.5,50,-0.5,1.5);
  hphi_3l = new TH2F("hphi_3l","phi correlation (3 layers)",50,-2.2,-1.0,50,-2.2,-1.0);
  hpt_4l =new TH2F("hpt_4l","pt correlation (4 layers)",10,0,50,10,0,50);
  hpz_4l =new TH2F("hpz_4l","pz correlation (4 layers)",10,-5,25,10,-5,25);
  hq_4l = new TH2F("hq_4l","charge correlation (4 layers)",2,-1.5,1.5,2,-1.5,1.5);
  heta_4l = new TH2F("heta_4l","eta correlation (4 layers)",50,-0.5,1.5,50,-0.5,1.5);
  hphi_4l = new TH2F("hphi_4l","phi correlation (4 layers)",50,-2.2,-1.0,50,-2.2,-1.0);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CombTrack::endJob() {
  hFile->Write();
  hFile->Close();
}

void CombTrack::AnalHits(const TrackingRecHitCollection &hits,
			 const edm::EventSetup& es){
  using namespace std;
  ltob1=false; ltob2=false; ltib1=false; ltib2=false;

  TrackingRecHitCollection::const_iterator hit;
  for(hit=hits.begin();hit!=hits.end();hit++){
  
    unsigned int iid=(*hit).geographicalId().rawId();
    int sub=(iid>>25)&0x7 ;
    int lay=(iid>>16) & 0xF;
    if ((lay==1)&&(sub==3)) ltib1=true;
    if ((lay==2)&&(sub==3)) ltib2=true;
    if ((lay==1)&&(sub==5)) ltob1=true;
    if ((lay==2)&&(sub==5)) ltob2=true;
 
  }
  nlay=ltib1+ltib2+ltob1+ltob2;
}


