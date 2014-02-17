// File: ReadCosmicTracks.cc
// Description:  see ReadCosmicTracks.h
// Author:  M.Pioppi Univ. & INFN Perugia
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "RecoTracker/SingleTrackPattern/test/ReadCosmicTracks.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "RecoTracker/SingleTrackPattern/test/TrajectoryMeasurementResidual.h"
using namespace std;
using namespace edm;
ReadCosmicTracks::ReadCosmicTracks(edm::ParameterSet const& conf) : 
  conf_(conf)
{
  trinevents=conf_.getParameter<bool>("TrajInEvents");
}
void ReadCosmicTracks::beginRun(edm::Run & run, const edm::EventSetup& c){

  //HISTOGRAM BOOKING
  hFile = new TFile ( "ptRes.root", "RECREATE" );
  hptres = new TH1F("hptres","Pt resolution",100,-0.1,0.1);
  hptres1 = new TH1F("hptres1","Pt resolution  0-10 GeV",100,-0.005,0.005);
  hptres2 = new TH1F("hptres2","Pt resolution 10-15 GeV",100,-0.005,0.005);
  hptres3 = new TH1F("hptres3","Pt resolution 15-20 GeV",100,-0.005,0.005);
  hptres4 = new TH1F("hptres4","Pt resolution 20-25 GeV",100,-0.005,0.005);
  hptres5 = new TH1F("hptres5","Pt resolution 25-30 GeV",100,-0.005,0.005);
  hptres6 = new TH1F("hptres6","Pt resolution 30-35 GeV",100,-0.005,0.005);
  hptres7 = new TH1F("hptres7","Pt resolution 35-40 GeV",100,-0.005,0.005);
  hptres8 = new TH1F("hptres8","Pt resolution 40-45 GeV",100,-0.005,0.005);
  hptres9 = new TH1F("hptres9","Pt resolution 45-50 GeV",100,-0.005,0.005);
  hptres10 = new TH1F("hptres10","Pt resolution >50 GeV",100,-0.005,0.005);
  hchiSq = new TH1F("hchiSq","Chi2",50,0 ,300);
  hchiSq1 = new TH1F("hchiSq1","Chi2  0-10 GeV",50, 0 ,300);
  hchiSq2 = new TH1F("hchiSq2","Chi2 10-15 GeV",50, 0 ,300);
  hchiSq3 = new TH1F("hchiSq3","Chi2 15-20 GeV",50, 0 ,300);
  hchiSq4 = new TH1F("hchiSq4","Chi2 20-25 GeV",50, 0 ,300);
  hchiSq5 = new TH1F("hchiSq5","Chi2 25-30 GeV",50, 0 ,300);
  hchiSq6 = new TH1F("hchiSq6","Chi2 30-35 GeV",50, 0 ,300);
  hchiSq7 = new TH1F("hchiSq7","Chi2 35-40 GeV",50, 0 ,300);
  hchiSq8 = new TH1F("hchiSq8","Chi2 40-45 GeV",50, 0 ,300);
  hchiSq9 = new TH1F("hchiSq9","Chi2 45-50 GeV",50, 0 ,300);
  hchiSq10 = new TH1F("hchiSq10","Chi2 >50 GeV",50, 0 ,300);
  hchipt= new TH1F("hchipt","Chi2 as a function of Pt", 10, 5, 55);
  hchiR= new TH1F("hchiR","Chi2 as a function of R", 10, 5, 105);
  hrespt= new TH1F("hrespt","resolution as a function of Pt", 10, 5, 55);
  heffpt= new TH1F("heffpt","efficiency as a function of Pt", 10, 5, 55);
  heffhit= new TH1F("heffhit","efficiency as a function of number of hits", 9, 5, 50);
  hcharge=new TH1F("hcharge","1=SIM+REC+,2=SIM+REC-,3=SIM-REC-,4=SIM-REC+",4,0.5,4.5);
  hresTIB=new TH1F("hresTIB","residuals for TIB modules",100,-0.05,0.05);
  hresTOB=new TH1F("hresTOB","residuals for TOB modules",100,-0.05,0.05);
  hresTID=new TH1F("hresTID","residuals for TID modules",100,-0.05,0.05);
  hresTEC=new TH1F("hresTEC","residuals for TEC modules",100,-0.05,0.05);

  // COUNTERS INITIALIZATION
  for (unsigned int ik=0;ik<10;ik++){
    inum[ik]=0;
    iden[ik]=0;
    ichiR[ik]=0;
    iden3[ik]=0;
  }
  for (unsigned int ik=0;ik<9;ik++){
    inum2[ik]=0;
    iden2[ik]=0;
  }
  GlobDen=0;
  GlobNum=0;
}
// Virtual destructor needed.
ReadCosmicTracks::~ReadCosmicTracks() {  }  

// Functions that gets called by framework every event
void ReadCosmicTracks::analyze(const edm::Event& e, const edm::EventSetup& es)
{
 
  edm::InputTag seedTag = conf_.getParameter<edm::InputTag>("cosmicSeeds");
  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("cosmicTracks");
  std::string MCTag = conf_.getParameter<std::string>("SimHits");

  bool trackable_cosmic=false;
  //SIM HITS
 
  edm::Handle<edm::PSimHitContainer> TOBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TIBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TIBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> PixelBarrelLowTof;
  edm::Handle<edm::PSimHitContainer> PixelBarrelHighTof;
 
  e.getByLabel(MCTag,"TrackerHitsTOBLowTof", TOBHitsLowTof);
 
  e.getByLabel(MCTag,"TrackerHitsTOBHighTof", TOBHitsHighTof);
  e.getByLabel(MCTag,"TrackerHitsTIBLowTof", TIBHitsLowTof);
  e.getByLabel(MCTag,"TrackerHitsTIBHighTof", TIBHitsHighTof);
  theStripHits.clear();
//   e.getByLabel(MCTag,"TrackerHitsPixelBarrelLowTof", PixelBarrelLowTof);
//   e.getByLabel(MCTag,"TrackerHitsPixelBarrelHighTof", PixelBarrelHighTof);
  theStripHits.insert(theStripHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end());
 
  theStripHits.insert(theStripHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
  theStripHits.insert(theStripHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end());
  theStripHits.insert(theStripHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
  //  theStripHits.insert(theStripHits.end(), PixelBarrelLowTof->begin(), PixelBarrelLowTof->end());
  // theStripHits.insert(theStripHits.end(), PixelBarrelHighTof->begin(), PixelBarrelHighTof->end());
 
  //RECONSTRUCTED OBJECTS
  //SEEDS AND TRACKS
  edm::Handle<TrajectorySeedCollection> seedcoll;
  e.getByLabel(seedTag,seedcoll);


  edm::Handle<SiStripRecHit2DCollection> rphirecHits;
  edm::InputTag rphirecHitsTag = conf_.getParameter<edm::InputTag>("rphirecHits");
  e.getByLabel( rphirecHitsTag ,rphirecHits);
  //TRACKER GEOMETRY

  es.get<TrackerDigiGeometryRecord>().get(tracker);
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  es.get<TransientRecHitRecord>().get(builderName,theBuilder);  
  RHBuilder=   theBuilder.product();

  //SIMULATED INFORMATION:
  //NUMBER OF SIMHIT ,CHARGE  AND PT
  unsigned int nshit=theStripHits.size();
  stable_sort(theStripHits.begin(),theStripHits.end(),CompareTOF());
  PSimHit isimfirst=(*theStripHits.begin());
  // PSimHit isimlast=(*(theStripHits.end()-1));
  PSimHit isimlast=theStripHits.back();
 
  int chsim=-1*isimfirst.particleType()/abs(isimfirst.particleType());
  DetId tmp1=DetId(isimfirst.detUnitId());
  GlobalVector gvs= tracker->idToDet(tmp1)->surface().toGlobal(isimfirst.localDirection());
  float ptsims=gvs.perp()*isimfirst.pabs();
  if (nshit>50) nshit=50;
  unsigned int inshit=(nshit/5)-1;
  unsigned int iptsims= (unsigned int)(ptsims/5 -1);
  if (iptsims>10) iptsims=10; 
  trackable_cosmic=false;
  //CRITERION TO SAY IF A MUON IS TRACKABLE OR NOT
  if( (seedcoll.product()->size()>0) &&(rphirecHits.product()->size()>2) && (rphirecHits.product()->size()<10) ) {

    trackable_cosmic=true;
    
    if (trackable_cosmic){
     
      SiStripRecHit2DCollection::DataContainer::const_iterator istrip;
      SiStripRecHit2DCollection::DataContainer stripcoll=rphirecHits->data();
      for(istrip=stripcoll.begin();istrip!=stripcoll.end();istrip++)
      iden[iptsims]++;
      iden2[inshit]++;
      GlobDen++;
    }
  }
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByLabel(TkTag,trackCollection);
  edm::Handle<TrackingRecHitCollection> trackerhitCollection;
  e.getByLabel(TkTag,trackerhitCollection);
  edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
  if (trinevents){
    e.getByLabel(TkTag,TrajectoryCollection);
  }
  
  
  
  
  
  
  //RECONSTRUCTED INFORMATION
  const   reco::TrackCollection *tracks=trackCollection.product();
  
  
  
  //INCREASE THE NUMBER OF TRACKED MUONS
  if (tracks->size()>0){
    reco::TrackCollection::const_iterator ibeg=tracks->begin();
    if (trackable_cosmic) {
      inum[iptsims]++;
      inum2[inshit]++;
      GlobNum++;
    } 
 
    GlobalPoint gp((*ibeg).outerPosition().x(),
		   (*ibeg).outerPosition().y(),
		   (*ibeg).outerPosition().z());

    //PT,Chi2,CHARGE RECONSTRUCTED
    float ptrec = (*ibeg).outerPt();
    float chiSquared =(*ibeg).chi2();
    int chrec=(*ibeg).charge();
  
  

    //FIND THE SIMULATED HIT CLOSEST TO THE 
    //FIRST REC HIT OF THE TRACK  
    //AND THE RADIAL DISTANCE OF THE TRACK
    std::vector<PSimHit>::iterator ihit;
    float MAGR=10000;
    float magmag=10000;
    DetId idet;
  
  //    std::cout<<"DELTATOF "<<(*theStripHits.begin())->tof()-
    //   *(theStripHits.end()-1).tof()<<std::endl;
    bool tofcorr=(tracker->idToDet(DetId(isimlast.detUnitId()))->surface().
		  toGlobal(isimlast.localPosition()).y() >
		  tracker->idToDet(DetId(isimfirst.detUnitId()))->surface().
		  toGlobal(isimfirst.localPosition()).y())? false:true;
		  

    for (ihit=theStripHits.begin();ihit!=theStripHits.end();ihit++){
      DetId tmp=DetId((*ihit).detUnitId());
      GlobalPoint gp1 =tracker->idToDet(tmp)->surface().toGlobal((*ihit).localPosition());
      float RR=sqrt((gp1.x()*gp1.x())+(gp1.y()*gp1.y()));
      if (RR<MAGR) MAGR=RR;
      if (((gp1-gp).mag()<magmag)&&(abs((*ihit).particleType())==13)){
	magmag=(gp1-gp).mag();
	isim=&(*ihit);
	idet=DetId(tmp);
      }
    }
//     //CHI2 vs RADIAL DISTANCE
    unsigned int iRR=(unsigned int)(MAGR/10);
    iden3[iRR]++;
    ichiR[iRR]+=chiSquared;


    //REEVALUATION OF THE SIMULATED PT
    GlobalVector gv= tracker->idToDet(idet)->surface().toGlobal(isim->localDirection());
    if (!tofcorr) {
      gv=-1*gv;
      chsim=-1*chsim;
    }
    float PP=isim->pabs();
    float ptsim=gv.perp()*PP;
   


    //FILL CHARGE HISTOGRAM 
    if(chrec==1 && chsim==1) hcharge->Fill(1);
    if(chrec==-1 && chsim==1) hcharge->Fill(2);
    if(chrec==-1 && chsim==-1) hcharge->Fill(3);
    if(chrec==1 && chsim==-1) hcharge->Fill(4);

    //PT^-1 RESOLUTION
    float ptresrel=((1./ptrec)-(1./ptsim));
    //   if (seed_plus){  

    //FILL PT RESOLUTION AND CHI2 HISTO 
    //IN PT BIN
      hptres->Fill(ptresrel);
      hchiSq->Fill(chiSquared);
      unsigned int iptsim= (unsigned int)(ptsim/5 -1);
      if (iptsim>10) iptsim=10;
      
      if (iptsim==1) {
	hptres1->Fill(ptresrel);
	hchiSq1->Fill(chiSquared);
      }
      if (iptsim==2) {
	hchiSq2->Fill(chiSquared);
	hptres2->Fill(ptresrel);
      }
      if (iptsim==3) {
	hchiSq3->Fill(chiSquared);
	hptres3->Fill(ptresrel);
      }
      if (iptsim==4) {
	hchiSq4->Fill(chiSquared);
	hptres4->Fill(ptresrel);
      }
      if (iptsim==5) {
	hchiSq5->Fill(chiSquared);
	hptres5->Fill(ptresrel);
      }
      if (iptsim==6) {
	hchiSq6->Fill(chiSquared);
	hptres6->Fill(ptresrel);
      }
      if (iptsim==7) {
	hptres7->Fill(ptresrel);
	hchiSq7->Fill(chiSquared);
      }
      if (iptsim==8) {
	hchiSq8->Fill(chiSquared);
	hptres8->Fill(ptresrel);
      }
      if (iptsim==9) {
	hchiSq9->Fill(chiSquared);
	hptres9->Fill(ptresrel);
      }
      if (iptsim==10) {
	hchiSq10->Fill(chiSquared);
	hptres10->Fill(ptresrel);
      }
      // }
      if (trinevents)
	makeResiduals(*(TrajectoryCollection.product()->begin()));
      
  }
}
void ReadCosmicTracks::makeResiduals(const Trajectory traj){

  std::vector<TrajectoryMeasurement> TMeas=traj.measurements();
  vector<TrajectoryMeasurement>::iterator itm;
  for (itm=TMeas.begin();itm!=TMeas.end();itm++){
    TrajectoryMeasurementResidual*  TMR=new TrajectoryMeasurementResidual(*itm);
    StripSubdetector iid=StripSubdetector((*itm).recHit()->detUnit()->geographicalId().rawId());
    unsigned int subid=iid.subdetId();
    if    (subid==  StripSubdetector::TID) hresTID->Fill(TMR->measurementXResidual());
    if    (subid==  StripSubdetector::TEC) hresTEC->Fill(TMR->measurementXResidual());
    if    (subid==  StripSubdetector::TIB) hresTIB->Fill(TMR->measurementXResidual());
    if    (subid==  StripSubdetector::TOB) hresTOB->Fill(TMR->measurementXResidual());
    delete TMR;
  }
}

void ReadCosmicTracks::endJob(){
  //FILL EFFICIENCY vs PT
  //AND CHI2 vs R
  for (unsigned int ik=0;ik<10;ik++) {
    heffpt->Fill(7.5+(ik*5),float(inum[ik])/float(iden[ik]));
    hchiR->Fill(10+(ik*10),ichiR[ik]/float(iden3[ik]));
  }
  //FILL EFFICIENCY vs NHIT
  for  (unsigned int ik=0;ik<9;ik++) {
    heffhit->Fill(7.5+(ik*5),float(inum2[ik])/float(iden2[ik]));
  }
  //FILL CHI2 vs PT 
  hchipt->Fill(7.5,hchiSq1->GetMean());
  hchipt->Fill(12.5,hchiSq2->GetMean());
  hchipt->Fill(17.5,hchiSq3->GetMean());
  hchipt->Fill(22.5,hchiSq4->GetMean());
  hchipt->Fill(27.5,hchiSq5->GetMean());
  hchipt->Fill(32.5,hchiSq6->GetMean());
  hchipt->Fill(37.5,hchiSq7->GetMean());
  hchipt->Fill(42.5,hchiSq8->GetMean());
  hchipt->Fill(47.5,hchiSq9->GetMean());
  hchipt->Fill(52.5,hchiSq10->GetMean());

  //WRITE ROOT FILE
  hFile->Write();
  hFile->Close();
}



