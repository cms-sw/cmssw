////////////////////////////////////////////////////////////////////////////////
// Package:          RecoTracker/SingleTrackPattern
// Class:            AnalyzeHitEff
// Original Author:  Daniele Benedetti-INFN perugia
//
//
//*************** Important Notes ***********************************
//  The code works only for TIB-TOB modules
//  To run the code is necessary change the number of maxLostHit and maxConsecLostHit in GroupedCkfTrajectoryBuilderESProducerTIFTIBTOB.cff
//
//  replace GroupedCkfTrajectoryBuilderTIF.maxLostHit          = 6
//  replace GroupedCkfTrajectoryBuilderTIF.maxConsecLostHit    = 4
//  
//  The code has been tested using a seeding from pairs and only in TIF data. 
//
//  For Layer 2,3,4 ---> Select Seed Pairs from TOB (Lay = 8,9,10)
//  For Layer 5,6,7 ---> Select Seed Pairs From TIB (Lay = 1,2,3) and TOB (Lay = 8,9,10)
//  For Layer 8,9   ---> Select Seed Pairs From TIB (Lay = 1,2,3)
//
//  Change accordly the file in RecoTracker/SpecialSeedGenerators/data/CombinatorialSeedGeneratorForCosmicsTIFTOB.cff
//
///////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <iostream>

#include "RecoTracker/SingleTrackPattern/test/AnalyzeHitEff.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/SingleTrackPattern/test/TrajectoryInValidHit.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "TMath.h"

using namespace std;
AnalyzeHitEff::AnalyzeHitEff(edm::ParameterSet const& conf) : 
  conf_(conf)
{
  layers =conf_.getParameter<int>("Layer");
}
void AnalyzeHitEff::beginRun(edm::Run & run, const edm::EventSetup& c){
  hFile = new TFile ( "trackeff.root", "RECREATE" );

  //Tree
  n = new TTree("n","a Tree");
  n->Branch("RHNum",&RHNum,"RHNum/I");
  n->Branch("RHgpx",RHgpx,"RHgpx[RHNum]/F");
  n->Branch("RHgpy",RHgpy,"RHgpy[RHNum]/F");
  n->Branch("RHgpz",RHgpz,"RHgpz[RHNum]/F");
  n->Branch("RHMod",RHMod,"RHMod[RHNum]/I");
  n->Branch("RphiSte",RphiSte,"RphiSte[RHNum]/I");

  n->Branch("TRHNum",&TRHNum,"TRHNum/I");
  n->Branch("TRHgpx",TRHgpx,"TRHgpx[TRHNum]/F");
  n->Branch("TRHgpy",TRHgpy,"TRHgpy[TRHNum]/F");
  n->Branch("TRHgpz",TRHgpz,"TRHgpz[TRHNum]/F");
  n->Branch("TRHLay",TRHLay,"TRHLay[TRHNum]/I");
  n->Branch("TRHMod",TRHMod,"TRHMod[TRHNum]/I");



  // Histo
  AngleLayTIB =  new TH1F("AngleLayTIB","  ",100,-100.,100.);



  
  out1.open("LayerOut.txt");
 
  // take from eventSetup the SiStripDetCabling object

  edm::ESHandle<SiStripDetCabling> tkmechstruct;
  c.get<SiStripDetCablingRcd>().get(tkmechstruct);

  // get list of active detectors from SiStripDetCabling
  std::vector<uint32_t> activeDets;
  activeDets.clear(); // just in case
  tkmechstruct->addActiveDetectorsRawIds(activeDets);

  // use SiStripSubStructure for selecting certain regions
  SiStripSubStructure substructure;
  std::vector<uint32_t> DetIds = activeDets;

 
  // Active modules
  for (std::vector<uint32_t>::const_iterator DetItr=activeDets.begin(); DetItr!=activeDets.end(); DetItr++){

    StripSubdetector iid=StripSubdetector(*DetItr);
    unsigned int subid=iid.subdetId();

    int modulo;
    modulo = (*DetItr)%2;
    unsigned int TkLayers = 1000;

    if(subid ==  StripSubdetector::TIB) { 
      TIBDetId tibid(*DetItr);
      unsigned int laytib = 0;
      laytib = tibid.layer();
      TkLayers = laytib;
    }
   if(subid ==  StripSubdetector::TOB) { 
      TOBDetId tobid(*DetItr);
      unsigned int laytob = 0;
      laytob =tobid.layer();
      TkLayers = laytob + 4;
   }
   if (TkLayers  == layers ) {
     if ( modulo != 0){
       cout << " modulo stereo " << (*DetItr) << " TkLayers " << TkLayers << endl;
      ActiveLayStereo.push_back(*DetItr);
     }
     else{
       cout << " modulo mono " << (*DetItr) << " TkLayers " << TkLayers << endl;
       ActiveLayMono.push_back(*DetItr);
     }
   }
  }

    
 
  for (unsigned int i=0;i<ActiveLayStereo.size();i++){
    out1 << i << "   " << ActiveLayStereo[i] << endl;
  }
  for (unsigned int i=0;i<ActiveLayMono.size();i++){
    out2 << i << "   " << ActiveLayMono[i] << endl;
  }

  int rangeSte = ActiveLayStereo.size();
  int rangeRPhi = ActiveLayMono.size();
  CKFLayEff = new TH1F("CKFLayEff"," Layer Efficiency ",12,0.,12.);
  InvCKFLayEff = new TH1F("InvCKFLayEff"," Layer Efficiency ",12,0.,12.);

  TIFInvModEffStereo  = new TH1F("TIFInvModEffStereo"," TIFInvModEff Stereo ",rangeSte,0.,rangeSte);
  TIFInvModEffStereo_all  = new TH1F("TIFInvModEffStereo_all"," TIFInvModEff Stereo ",rangeSte,0.,rangeSte);
  TIFModEffStereo  = new TH1F("TIFModEffStereo"," TIFInvModEff Stereo ",rangeSte,0.,rangeSte);

  TIFInvModEffRPhi  = new TH1F("TIFInvModEffRPhi"," TIFInvModEff RPhi ",rangeRPhi,0.,rangeRPhi);
  TIFInvModEffRPhi_all  = new TH1F("TIFInvModEffRPhi_all"," TIFInvModEff RPhi ",rangeRPhi,0.,rangeRPhi);
  TIFModEffRPhi  = new TH1F("TIFModEffRPhi"," TIFInvModEff RPhi ",rangeRPhi,0.,rangeRPhi);

  ResidualXValidSte = new TH1F("ResidualXValidSte"," Residual TIB Matched Valid Ste",100,-0.5,0.5);
  ResidualXValidSte_2 = new TH1F("ResidualXValidSte_2"," Residual TIB Matched Valid Ste",100,-1.5,1.5);
  ResidualXValidRPhi = new TH1F("ResidualXValidRPhi"," Residual TIB Matched Valid RPhi",100,-0.5,0.5);
  ResidualXValidRPhi_2 = new TH1F("ResidualXValidRPhi_2"," Residual TIB Matched Valid RPhi",100,-1.5,1.5);

  hLocYSte = new TH1F("hLocYSte","   ",100,-12,12.);
  hLocXSte = new TH1F("hLocXSte","   ",100,-7,7.);
  hInvLocYSte = new TH1F("hInvLocYSte","   ",100,-12,12.);
  hInvLocXSte = new TH1F("hInvLocXSte","   ",100,-7,7.);
  hInvLocYSte_Mod = new TH1F("hInvLocYSte_Mod","   ",100,-12,12.);
  hInvLocXSte_Mod = new TH1F("hInvLocXSte_Mod","   ",100,-7,7.);

  hLocYRPhi = new TH1F("hLocYRPhi","   ",100,-12,12.);
  hLocXRPhi = new TH1F("hLocXRPhi","   ",100,-7,7.);
  hInvLocYRPhi = new TH1F("hInvLocYRPhi","   ",100,-12,12.);
  hInvLocXRPhi = new TH1F("hInvLocXRPhi","   ",100,-7,7.);
  hInvLocYRPhi_Mod = new TH1F("hInvLocYRPhi_Mod","   ",100,-12,12.);
  hInvLocXRPhi_Mod = new TH1F("hInvLocXRPhi_Mod","   ",100,-7,7.);

  hLocErrYSte = new TH1F("hLocErrYSte","   ",200,0,3.);
  hLocErrXSte = new TH1F("hLocErrXSte","   ",200,0,1.);
  hInvLocErrYSte = new TH1F("hInvLocErrYSte","   ",200,0,3.);
  hInvLocErrXSte = new TH1F("hInvLocErrXSte","   ",200,0,1.);

  hLocErrSte = new TH1F("hLocErrSte","   ",200,0,3.);
  hInvLocErrSte = new TH1F("hInvLocErrSte","   ",200,0,3.);

  hLocErrYRPhi = new TH1F("hLocErrYRPhi","   ",200,0,3.);
  hLocErrXRPhi = new TH1F("hLocErrXRPhi","   ",200,0,1.);
  hInvLocErrYRPhi = new TH1F("hInvLocErrYRPhi","   ",200,0,3.);
  hInvLocErrXRPhi = new TH1F("hInvLocErrXRPhi","   ",200,0,1.);

  hLocErrRPhi = new TH1F("hLocErrRPhi","   ",200,0,3.);
  hInvLocErrRPhi = new TH1F("hInvLocErrRPhi","   ",200,0,3.);



  // Basic Track Parameter Histo
  hTrkphiCKF = new     TH1F("hTrkphiCKF",    "Phi distribution",         100,-3.0, 0.);
  hTrketaCKF = new     TH1F("hTrketaCKF",    "Eta distribution",         100,  -1.5,  1.5); 
  hTrkchi2CKF = new     TH1F("hTrkchi2CKF",    "Chi squared of the track", 300,    0, 60    );
  hTrkchi2Good = new     TH1F("hTrkchi2Good",    "Chi squared of the track", 300,    0, 60    );
  hTrkchi2Bad = new     TH1F("hTrkchi2Bad",    "Chi squared of the track", 300,    0, 60    );
  hTrknhitCKF = new    TH1F("hTrknhitCKF",   "Number of Hits per Track ", 20,  2., 22.  );


  hdiscr1Ste = new TH1F("hdiscr1Ste"," ",200,0.,10.);
  hdiscr2Ste = new TH1F("hdiscr2Ste"," ",200,0.,10.);
  hInvdiscr1Ste = new TH1F("hInvdiscr1Ste"," ",200,0.,10.);
  hInvdiscr2Ste = new TH1F("hInvdiscr2Ste"," ",200,0.,10.);

  hdiscr1RPhi = new TH1F("hdiscr1RPhi"," ",200,0.,10.);
  hdiscr2RPhi = new TH1F("hdiscr2RPhi"," ",200,0.,10.);
  hInvdiscr1RPhi = new TH1F("hInvdiscr1RPhi"," ",200,0.,10.);
  hInvdiscr2RPhi = new TH1F("hInvdiscr2RPhi"," ",200,0.,10.);

  hdiscr1RPhi_log = new TH1F("hdiscr1RPhi_log"," ",200,0.,3.);
  hdiscr2RPhi_log = new TH1F("hdiscr2RPhi_log"," ",200,0.,3.);
  hInvdiscr1RPhi_log = new TH1F("hInvdiscr1RPhi_log"," ",200,0.,3.);
  hInvdiscr2RPhi_log = new TH1F("hInvdiscr2RPhi_log"," ",200,0.,3.);


  hMatchRPhi = new TH1F("hMatchRPhi"," ",6,0,6);
  hInvMatchRPhi = new TH1F("hInvMatchRPhi"," ",6,0,6);


  hHitRPhi = new TH1F("hHitRPhi"," ",14,0,14);
  hInvHitRPhi = new TH1F("hInvHitRPhi"," ",14,0,14);
  

  events = 0;
  EventSeedCKF = 0;
  EventTrackCKF = 0;
  EventTriggCKF = 0;


}
// Virtual destructor needed.
AnalyzeHitEff::~AnalyzeHitEff() {  }  

void AnalyzeHitEff::analyze(const edm::Event& e, const edm::EventSetup& es)
{

  using namespace edm;
  // Step A: Get Inputs 

  //CombinatorialSeed
  edm::Handle<TrajectorySeedCollection> seedcollCKF;
  edm::InputTag seedTagCKF = conf_.getParameter<edm::InputTag>("combinatorialSeeds");
  e.getByLabel(seedTagCKF,seedcollCKF);

  //CombinatoriaTrack
  edm::Handle<reco::TrackCollection> trackCollectionCKF;
  edm::InputTag TkTagCKF = conf_.getParameter<edm::InputTag>("combinatorialTracks");
  e.getByLabel(TkTagCKF,trackCollectionCKF);



  edm::Handle<std::vector<Trajectory> > TrajectoryCollectionCKF;
  e.getByLabel(TkTagCKF,TrajectoryCollectionCKF);



  // RecHit
  //rphi
  edm::Handle<SiStripRecHit2DCollection> rphirecHits;
  edm::InputTag rphirecHitsTag = conf_.getParameter<edm::InputTag>("rphirecHits");
  e.getByLabel( rphirecHitsTag ,rphirecHits);

  //stereo
  edm::Handle<SiStripRecHit2DCollection> stereorecHits;
  edm::InputTag stereorecHitsTag = conf_.getParameter<edm::InputTag>("stereorecHits");
  e.getByLabel( stereorecHitsTag ,stereorecHits);

  //*****************
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  const TrackerGeometry * tkgeom=&(* tracker);

  theAngleFinder = new TrackLocalAngle( tracker.product() );


  events++;
  

  for (int i=0;i<200;i++){
    RHgpx[i] = 0.;
    RHgpy[i] = 0.;
    RHgpz[i] = 0.;
    RHMod[i] = 0;
    RphiSte[i] = 0;
    TRHgpx[i] = 0.;
    TRHgpy[i] = 0.;
    TRHgpz[i] = 0.;
    TRHLay[i] = 0;
    TRHMod[i] = 0;
  }


  // ************ Stereo RecHit Collection
  int IsRphiSte = 2;
  int ContRH = 0;
  vRPhi_SiStripRecHit2D.clear();
  vSte_SiStripRecHit2D.clear();
  SiStripRecHit2DCollection::DataContainer::const_iterator istripSt;
  const SiStripRecHit2DCollection::DataContainer & stripcollSt = stereorecHits->data();

  
  if (stereorecHits.product()->size() > 0) {
    for(istripSt=stripcollSt.begin();istripSt!=stripcollSt.end();istripSt++) {
      const SiStripRecHit2D *hitSt = &*(istripSt);
      vSte_SiStripRecHit2D.push_back(hitSt);

      IsRphiSte = 1;
      DetId idet=(*istripSt).geographicalId();
      unsigned int idetMod = idet.rawId();
      unsigned int subid=idet.subdetId();
      GlobalPoint gpsRec = tracker->idToDet(idet)->surface().toGlobal(istripSt->localPosition());
      if    (subid==  StripSubdetector::TIB || subid==  StripSubdetector::TOB ) {
	if (ContRH < 200){
	  RHgpx[ContRH] = gpsRec.x();
	  RHgpy[ContRH] = gpsRec.y();
	  RHgpz[ContRH] = gpsRec.z();
	  RHMod[ContRH] = idetMod;
	  RphiSte[ContRH] = IsRphiSte;
	}
	ContRH++;
      }  
    }
  }	
  // ************ END Stereo RecHit Collection
  

  // ************ RPhi RecHit Collection
  SiStripRecHit2DCollection::DataContainer::const_iterator istrip;
  const SiStripRecHit2DCollection::DataContainer & stripcoll = rphirecHits->data();
  if (rphirecHits.product()->size() > 0) {
    for(istrip=stripcoll.begin();istrip!=stripcoll.end();istrip++) {
      DetId idet=(*istrip).geographicalId();
      unsigned int idetMod = idet.rawId();
      IsRphiSte = 0;
      
      unsigned int toblayer=0;
      unsigned int tiblayer=0;
      unsigned int Totlayer=0;
      unsigned int subid=idet.subdetId();
      GlobalPoint gpsRec = tracker->idToDet(idet)->surface().toGlobal(istrip->localPosition());
      if    (subid==  StripSubdetector::TIB) {
	tiblayer= TIBDetId(idet).layer();
	Totlayer = tiblayer;
      }
      if    (subid==  StripSubdetector::TOB) {
	toblayer= TOBDetId(idet).layer();
	Totlayer = toblayer + 4;
      }
      if    (subid==  StripSubdetector::TIB || subid==  StripSubdetector::TOB ) {
	if (ContRH < 200){
	  RHgpx[ContRH] = gpsRec.x();
	  RHgpy[ContRH] = gpsRec.y();
	  RHgpz[ContRH] = gpsRec.z();
	  RHMod[ContRH] = idetMod;
	  RphiSte[ContRH] = IsRphiSte;
	}
	ContRH++;
      }
      const SiStripRecHit2D *hit = &*(istrip);


      bool FakeArt = false;
      if (FakeArt == false) {
	vRPhi_SiStripRecHit2D.push_back(hit);
      }
      else {
	// Simulation of Artificial RPhi Inefficiency in layers, modules
	if(Totlayer  == layers ) {
	  double prob;
	  prob = RanGen2.Rndm();
	  if (prob > 0.1) {    
	    vRPhi_SiStripRecHit2D.push_back(hit);
	  }
	}
	else {
	  vRPhi_SiStripRecHit2D.push_back(hit);
	}
      }
    }
  }

  RHNum = ContRH;
  if (RHNum > 200 ) RHNum = 199;
 
  // Seed Trigg
  int SeedNumComb = 0;
  if (seedcollCKF.product()->size() > 0.) {
    SeedNumComb++;
  }
  if (SeedNumComb > 0) EventSeedCKF++;


  // Tracking 
  const   reco::TrackCollection *tracksCKF2=trackCollectionCKF.product();
  if (tracksCKF2->size() == 1 && SeedNumComb > 0 ){
    reco::TrackCollection::const_iterator iCKF=trackCollectionCKF.product()->begin();
    bool TIBTrigg = false;
    bool TOBTrigg = false;
    EventTrackCKF++;  

    // Basic Track Parameters 
 

    hTrkphiCKF->Fill((*iCKF).innerMomentum().phi());
    hTrketaCKF->Fill((*iCKF).innerMomentum().eta());
    hTrknhitCKF->Fill((*iCKF).recHitsSize());
    hTrkchi2CKF->Fill((*iCKF).chi2());


    // ************ Track Trigger
    int MatchedHit = 0;
    for (trackingRecHit_iterator itCkf = iCKF->recHitsBegin();  itCkf != iCKF->recHitsEnd(); itCkf++){

      if ((*itCkf)->isValid()){
	const TrackingRecHit* rechit = (*itCkf).get();
	const SiStripMatchedRecHit2D * matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(rechit); 
	if(matchedhit) MatchedHit++;
	  
	unsigned int iidd = (*itCkf)->geographicalId().rawId();
	StripSubdetector strip=StripSubdetector(iidd);
	unsigned int subid=strip.subdetId();
	if (subid ==  StripSubdetector::TOB) { 
	  TOBDetId tobid(iidd);
	  unsigned int laytob =tobid.layer();
	  // Trigger on TOB
	  if ( laytob == 6  ||  laytob == 5 ) TOBTrigg = true;
	}

	// Track Angle On the Modules 
	const Trajectory traj = *(TrajectoryCollectionCKF.product()->begin());
	std::vector<TrajectoryMeasurement> TMeas=traj.measurements();
	vector<TrajectoryMeasurement>::iterator itm;
	for (itm = TMeas.begin();itm != TMeas.end(); itm++){
	  ConstReferenceCountingPointer<TransientTrackingRecHit> theInHit;
	  theInHit = (*itm).recHit();
	  unsigned int iidd2 = theInHit->geographicalId().rawId();
	  StripSubdetector strip2=StripSubdetector(iidd2);
	  unsigned int subid2=strip2.subdetId();
	  if (subid2 ==  StripSubdetector::TIB) {
	    TIBDetId tibid(iidd2);
	    unsigned int laytib2 =tibid.layer();
	    std::pair<float,float> monoStereoAng = theAngleFinder->findtrackangle(*itm);
	    // Trigger on TIB
	    if (laytib2 == 1) {    // The trigger could be changed in general from TIB lay 3 is ok withoud any angular restriction. 
	      AngleLayTIB->Fill(monoStereoAng.first);
	      if (layers == 2){
		if (abs(monoStereoAng.first) < 45 )  TIBTrigg = true;
	      }
	      else TIBTrigg = true;
	    }
	  }
	}
      }
    }  //   ************ End Track Trigger

    int ContTRH = 0;      
    if (TIBTrigg && TOBTrigg && MatchedHit > 2) {
      
      EventTriggCKF++; 

      const Trajectory traj2 = *(TrajectoryCollectionCKF.product()->begin());
      std::vector<TrajectoryMeasurement> TMeas2=traj2.measurements();
      vector<TrajectoryMeasurement>::iterator itm2;
      double xloc = 0.;
      double yloc = 0.;
      double xlocSte = 0.;
      double ylocSte = 0.;
      double xErr = 0.;
      double yErr = 0.;
      double xglob,yglob,zglob;

      for (itm2=TMeas2.begin();itm2!=TMeas2.end();itm2++){
	ConstReferenceCountingPointer<TransientTrackingRecHit> theInHit2;
	theInHit2 = (*itm2).recHit();

	unsigned int iidd = theInHit2->geographicalId().rawId();
	unsigned int iiddStereo, iiddMono;

	StripSubdetector strip=StripSubdetector(iidd);
	unsigned int subid=strip.subdetId();
	unsigned int TKlayers = 0;
	if (subid ==  StripSubdetector::TIB) { 
	  TIBDetId tibid(iidd);
	  TKlayers = tibid.layer();
	}
	if (subid ==  StripSubdetector::TOB) { 
	  TOBDetId tobid(iidd);
	  TKlayers = tobid.layer() + 4 ; 
	}
	bool IsStereo = false;
	if (TKlayers == 1 || TKlayers == 2 || TKlayers == 5 || TKlayers == 6){    // Stereo layers
	  iiddStereo = iidd+1;
	  iiddMono = iidd+2;
	  IsStereo = true;
	}
	else {     // Mono layers
	  iiddStereo = 0;
	  iiddMono = iidd;
	}

	// Modules Constraints


	TrajectoryInValidHit*  TM = new TrajectoryInValidHit(*itm2,tkgeom);
	
	// --> Get trajectory from combinatedState 
	
	xloc = TM->localRPhiX();
	yloc = TM->localRPhiY();

	xlocSte = TM->localStereoX();
	ylocSte = TM->localStereoY();
	
	xErr =  TM->localErrorX();

	yErr =  TM->localErrorY();

	
	xglob = TM->globalX();
	yglob = TM->globalY();
	zglob = TM->globalZ();


	float YINTCons = 0.;
	if(TKlayers < 5) {
	  YINTCons = -1.;
	}
	if(TKlayers >= 5) {
	  YINTCons = 1.0;  // Bonding Region
	}



	if (layers != TKlayers) {    // Tracking position (just to draw) 
	  if (ContTRH < 200) {
	    TRHgpx[ContTRH] = xglob;
	    TRHgpy[ContTRH] = yglob;
	    TRHgpz[ContTRH] = zglob;
	    TRHLay[ContTRH] = TKlayers;
	    TRHMod[ContTRH] = iidd;
	  }
	  ContTRH++;
	}

 	if (layers == TKlayers) {   // Look at the layer not used to reconstruct the track
	  // *************************************  Stereo RecHit Efficiency  *****************************************************
	  
 	  if (IsStereo) {
	    if (vSte_SiStripRecHit2D.size() > 0) {
	      std::vector<const SiStripRecHit2D*>::const_iterator HitIterAll = vSte_SiStripRecHit2D.begin();
	      int ContSte = 0;
	      int IndexSte[10];
	      float ResRotateX[10];
	      float ModResRotateX[10];
	      for(; HitIterAll != vSte_SiStripRecHit2D.end(); HitIterAll++) {
		unsigned int SteRecHitID = ((*HitIterAll)->geographicalId()).rawId();
		if (SteRecHitID == iiddStereo){
		  ResRotateX[ContSte] = ((((*HitIterAll)->localPosition().x())) - xlocSte)*(-1.);
		  ModResRotateX[ContSte] = abs(ResRotateX[ContSte]);
		  ContSte++;
		}
	      }
	      // Just in case I have more than a rechit on the same module: I get the closer
	      float FinalResSte;
	      if (ContSte > 0) {
		if (ContSte > 1) {
		  TMath::Sort(ContSte,ModResRotateX,IndexSte,false);
		  FinalResSte = ResRotateX[IndexSte[0]];
		}
		else {
		  FinalResSte = ResRotateX[0];
		}
	      }
	      else {
		FinalResSte = 1000;
	      }

	      float discr1 = 0.;
	      float discr2 = 0.;
	      
	      if (layers < 5) {
		discr1 = 10.;
		discr2 = fabs(5.45 - fabs(ylocSte))/yErr;
	      }
	      if (layers >= 5) {
		discr1 = fabs(ylocSte)/yErr ;
		discr2 = fabs(9.4 - fabs(ylocSte))/yErr;
	      }  

	      //if ( abs(ylocSte) > YINTCons  && abs(xlocSte) < 1500 ) {  
	      if ( abs(ylocSte) > YINTCons  && abs(xlocSte) < 1500  && discr1 > 5 && discr2 > 5 ) {  
		
		for(unsigned int i=0;i<ActiveLayStereo.size();i++){    
		  if (iiddStereo == ActiveLayStereo[i] ) { 
		    ResidualXValidSte->Fill(FinalResSte); 
		    ResidualXValidSte_2->Fill(FinalResSte); 

		    
		    if (abs(FinalResSte) < 0.5 ) {    
		      TIFModEffStereo->Fill(i*1.); 
		      CKFLayEff->Fill((layers*1.));
		      hLocYSte->Fill(ylocSte);
		      hLocXSte->Fill(xlocSte);
		      
		      hLocErrYSte->Fill(yErr);
		      hLocErrXSte->Fill(xErr);
		      hLocErrSte->Fill(sqrt(xErr*xErr + yErr*yErr));

		      hdiscr1Ste->Fill(discr1);
		      hdiscr2Ste->Fill(discr2);

		      hTrkchi2Good->Fill((*iCKF).chi2());
		    }
		    else {
		      TIFInvModEffStereo->Fill(i*1.);     
		      InvCKFLayEff->Fill(layers*1.); 
		      cout << "######### Invalid Stereo FinalResX " << FinalResSte << " FinalRecHit " << iiddStereo << "   TKlayers  "  <<  TKlayers  << " xlocSte " <<  xlocSte << " ylocSte  " << ylocSte <<  endl;
		      cout << " Ste Error " << sqrt(xErr*xErr + yErr*yErr) << " ErrorX " << xErr << " yErr " <<  yErr << " Matched Hit " << MatchedHit <<  endl;


		      hInvLocYSte->Fill(ylocSte);
		      hInvLocXSte->Fill(xlocSte);
		      hInvLocErrSte->Fill(sqrt(xErr*xErr + yErr*yErr));

		      hTrkchi2Bad->Fill((*iCKF).chi2());

		      hInvdiscr1Ste->Fill(discr1);
		      hInvdiscr2Ste->Fill(discr2);
		      hInvLocErrYSte->Fill(yErr);
		      hInvLocErrXSte->Fill(xErr);

		      if (FinalResSte > 999. ) {
			TIFInvModEffStereo_all->Fill(i*1.);  
			cout << " ############  Mod Stereo Not Hit in The Module   ############ " << endl; 

			hInvLocYSte_Mod->Fill(ylocSte);
			hInvLocXSte_Mod->Fill(xlocSte);

		      }
		    }
		  }
		}
	      }
	    }
	  }

	  // *************************************  RPhi RecHit Efficiency  *****************************************************
	  if (vRPhi_SiStripRecHit2D.size() > 0) {
	    std::vector<const SiStripRecHit2D*>::const_iterator HitIterAll = vRPhi_SiStripRecHit2D.begin();
	    int ContRPhi = 0;
	    int IndexRPhi[10];
	    float ResX[10];
	    float ModResX[10];
	    for(; HitIterAll != vRPhi_SiStripRecHit2D.end(); HitIterAll++) {
	      unsigned int RPhiRecHitID = ((*HitIterAll)->geographicalId()).rawId();
	      if (RPhiRecHitID == iiddMono) {
		ResX[ContRPhi] = ( (*HitIterAll)->localPosition().x()) - xloc;
		ModResX[ContRPhi] = abs(ResX[ContRPhi]);
		ContRPhi++;
	      }
	    }
	    float FinalResRPhi;
	    if (ContRPhi > 0) {
	      if (ContRPhi > 1) {
		TMath::Sort(ContRPhi,ModResX,IndexRPhi,false);
		FinalResRPhi = ResX[IndexRPhi[0]];
	      }
	      else {
		FinalResRPhi = ResX[0];
	      }
	    }
	    else {
	      FinalResRPhi = 1000;
	    }
	    float discr1 = 0.;
	    float discr2 = 0.;
	    if (layers < 5) {
	      discr1 = 10.;
	      discr2 = fabs(5.45 - fabs(yloc))/yErr;
	    }
	    if (layers >= 5) {
	      discr1 = fabs(yloc)/yErr;
	      discr2 = fabs(9.4 - fabs(yloc))/yErr;
	    }  
	    
	    
	    if ( abs(yloc) > YINTCons && abs(xloc) < 1500 && discr1 > 5 && discr2 > 5 ) { 
	    //if ( abs(yloc) > YINTCons && abs(xloc) < 1500 ) { 
	      for(unsigned int i=0;i < ActiveLayMono.size();i++){   
		if (iiddMono == ActiveLayMono[i] ) { 

		  ResidualXValidRPhi->Fill(FinalResRPhi); 
		  ResidualXValidRPhi_2->Fill(FinalResRPhi);
		  
 
		  if (abs(FinalResRPhi) < 0.5) {    
		    TIFModEffRPhi->Fill(i*1.); 
		    CKFLayEff->Fill((layers*1.));
		    hLocYRPhi->Fill(yloc);
		    hLocXRPhi->Fill(xloc);
		    
		    hLocErrYRPhi->Fill(yErr);
		    hLocErrXRPhi->Fill(xErr);
		    hLocErrRPhi->Fill(sqrt(xErr*xErr + yErr*yErr));
		    hTrkchi2Good->Fill((*iCKF).chi2());

		    hdiscr1RPhi->Fill(discr1);
		    hdiscr2RPhi->Fill(discr2);

		    hdiscr1RPhi_log->Fill(log10(discr1));
		    hdiscr2RPhi_log->Fill(log10(discr2));

		    hMatchRPhi->Fill(MatchedHit);
		    hHitRPhi->Fill((*iCKF).recHitsSize());
		    

		  }
		  else {
		    TIFInvModEffRPhi->Fill(i*1.);     
		    InvCKFLayEff->Fill(layers*1.); 
		    cout << "######### Invalid RPhi FinalResX " << FinalResRPhi << " FinalRecHit " << iiddMono << "   TKlayers  "  <<  TKlayers  << " xloc " <<  xloc << " yloc  " << yloc << endl;
		    hInvLocYRPhi->Fill(yloc);
		    hInvLocXRPhi->Fill(xloc);

		    hInvLocErrYRPhi->Fill(yErr);
		    hInvLocErrXRPhi->Fill(xErr);
		    hInvLocErrRPhi->Fill(sqrt(xErr*xErr + yErr*yErr));
		    hTrkchi2Bad->Fill((*iCKF).chi2());

		    hInvdiscr1RPhi->Fill(discr1);
		    hInvdiscr2RPhi->Fill(discr2);
		    
		    hInvdiscr1RPhi_log->Fill(log10(discr1));
		    hInvdiscr2RPhi_log->Fill(log10(discr2));

		    cout << " RPhi Error " << sqrt(xErr*xErr + yErr*yErr) << " ErrorX " << xErr << " yErr " <<  yErr << " Matched Hit " << MatchedHit <<  endl;

		    hInvMatchRPhi->Fill(MatchedHit);
		    hInvHitRPhi->Fill((*iCKF).recHitsSize());
		    
		    if (FinalResRPhi > 999. ) {
		      TIFInvModEffRPhi_all->Fill(i*1.);  
		      cout << " ############  Mod RPhi Not Hit in The Module   ############ " << endl; 
		      hInvLocYRPhi_Mod->Fill(yloc);
		      hInvLocXRPhi_Mod->Fill(xloc);
		    }
		  }
		}
	      } 
	    }	    
	  }  // End RPhi
	}
      }
    }
    TRHNum = ContTRH;
    if (TRHNum > 200 ) TRHNum = 199;
    n->Fill();   // Fill the tree only if there are tracks. 
   }
}

void AnalyzeHitEff::endJob(){
  out1.close();
  out2.close();
  hFile->cd();
  n->Write();
  hFile->Write();
  hFile->Close();

  
  cout << " Events Analysed             " <<  events          << endl;
  cout << " Number Of Seeded events     " <<  EventSeedCKF    << endl;
  cout << " Number Of Tracked events    " <<  EventTrackCKF   << endl;
  cout << " Number Of Triggered events  " <<  EventTriggCKF   << endl;
}

