#include "Validation/TrackerHits/interface/TrackerHitAnalyzer.h"


#include "DataFormats/Common/interface/Handle.h"

// tracker info
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

// data in edm::event
#include "SimDataFormats/TrackerValidation/interface/PTrackerSimHit.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
//#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

// helper files
#include <CLHEP/Vector/LorentzVector.h>
#include <CLHEP/Units/SystemOfUnits.h>


#include <iostream>
using namespace edm;
using namespace std;

TrackerHitAnalyzer::TrackerHitAnalyzer(const edm::ParameterSet& ps) {

   fDBE = Service<DaqMonitorBEInterface>().operator->();
   fOutputFile = ps.getUntrackedParameter<string>("outputFile", "TrackerHitHisto.root");
//    src_ =  ps.getParameter<edm::InputTag>( "src" );
  
  //get Labels to use to extract information
  PxlBrlLowSrc_ = ps.getParameter<edm::InputTag>("PxlBrlLowSrc");
  PxlBrlHighSrc_ = ps.getParameter<edm::InputTag>("PxlBrlHighSrc");
  PxlFwdLowSrc_ = ps.getParameter<edm::InputTag>("PxlFwdLowSrc");
  PxlFwdHighSrc_ = ps.getParameter<edm::InputTag>("PxlFwdHighSrc");

  SiTIBLowSrc_ = ps.getParameter<edm::InputTag>("SiTIBLowSrc");
  SiTIBHighSrc_ = ps.getParameter<edm::InputTag>("SiTIBHighSrc");
  SiTOBLowSrc_ = ps.getParameter<edm::InputTag>("SiTOBLowSrc");
  SiTOBHighSrc_ = ps.getParameter<edm::InputTag>("SiTOBHighSrc");
  SiTIDLowSrc_ = ps.getParameter<edm::InputTag>("SiTIDLowSrc");
  SiTIDHighSrc_ = ps.getParameter<edm::InputTag>("SiTIDHighSrc");
  SiTECLowSrc_ = ps.getParameter<edm::InputTag>("SiTECLowSrc");
  SiTECHighSrc_ = ps.getParameter<edm::InputTag>("SiTECHighSrc");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output

////// booking histograms
   	
  Char_t  hname1[50], htitle1[80];
  Char_t  hname2[50], htitle2[80];
  Char_t  hname3[50], htitle3[80];
  Char_t  hname4[50], htitle4[80];
  Char_t  hname5[50], htitle5[80];
  Char_t  hname6[50], htitle6[80];
   
  if ( fDBE ) {
     if ( verbose_ ) {
       fDBE->setVerbose(1);
     } else {
       fDBE->setVerbose(0);
     }
  }
        																
  if ( fDBE) {
    if ( verbose_ ) fDBE->showDirStructure();
  } 

  if ( fDBE != NULL ) {
//   fDBE->setCurrentFolder("TrackerHitTask");
     
     // is there any way to record CPU Info ???
     // if so, it can be done once - via beginJob() 
 const float E2NEL = 1.; 
 
 const char *Region[] = {"005","051","115","152","225","253",
                       "-050","-105","-151","-215","-252","-325"};  
 
 int nbin = 10000;   
      
// Energy loss histograms
   for(int i=0; i<12; i++) {
  
    sprintf (htitle1,"Energy loss in TIB %s", Region[i]);
    sprintf (htitle2,"Energy loss in TOB %s", Region[i]);
    sprintf (htitle3,"Energy loss in TID %s", Region[i]);
    sprintf (htitle4,"Energy loss in TEC %s", Region[i]);
    sprintf (htitle5,"Energy loss in BPIX %s", Region[i]);
    sprintf (htitle6,"Energy loss in FPIX %s", Region[i]);
    
    sprintf (hname1,"Eloss_TIB_%i",i+1);
    sprintf (hname2,"Eloss_TOB_%i",i+1);
    sprintf (hname3,"Eloss_TID_%i",i+1);
    sprintf (hname4,"Eloss_TEC_%i",i+1);
    sprintf (hname5,"Eloss_BPIX_%i",i+1);
    sprintf (hname6,"Eloss_FPIX_%i",i+1);
   
    fDBE->setCurrentFolder("TIBHit");
    h1e[i]  = fDBE->book1D (hname1, htitle1, nbin , 0.0 , 0.001*E2NEL);
    fDBE->setCurrentFolder("TOBHit");
    h2e[i]  = fDBE->book1D (hname2, htitle2, nbin , 0.0 , 0.001*E2NEL);
    fDBE->setCurrentFolder("TIDHit");
    h3e[i]  = fDBE->book1D (hname3, htitle3, nbin , 0.0 , 0.001*E2NEL);
    fDBE->setCurrentFolder("TECHit");
    h4e[i]  = fDBE->book1D (hname4, htitle4, nbin , 0.0 , 0.001*E2NEL);
    fDBE->setCurrentFolder("BPIXHit");
    h5e[i]  = fDBE->book1D (hname5, htitle5, nbin , 0.0 , 0.001*E2NEL);
    fDBE->setCurrentFolder("FPIXHit");
    h6e[i]  = fDBE->book1D (hname6, htitle6, nbin , 0.0 , 0.001*E2NEL);
   
   }

// limits
const float high[] = {0.03, 0.03, 0.02, 0.03, 0.03, 0.03};
const float low[] = {-0.03, -0.03, -0.02, -0.03, -0.03, -0.03};
   
   for(int i=0; i<12; i++) {
  
    sprintf (htitle1,"Entryx-Exitx in TIB %s", Region[i]);
    sprintf (htitle2,"Entryx-Exitx in TOB %s", Region[i]);
    sprintf (htitle3,"Entryx-Exitx in TID %s", Region[i]);
    sprintf (htitle4,"Entryx-Exitx in TEC %s", Region[i]);
    sprintf (htitle5,"Entryx-Exitx in BPIX %s", Region[i]);
    sprintf (htitle6,"Entryx-Exitx in FPIX %s", Region[i]);
    
    sprintf (hname1,"Entryx-Exitx_TIB_%i",i+1);
    sprintf (hname2,"Entryx-Exitx_TOB_%i",i+1);
    sprintf (hname3,"Entryx-Exitx_TID_%i",i+1);
    sprintf (hname4,"Entryx-Exitx_TEC_%i",i+1);
    sprintf (hname5,"Entryx-Exitx_BPIX_%i",i+1);
    sprintf (hname6,"Entryx-Exitx_FPIX_%i",i+1);
   
    fDBE->setCurrentFolder("TIBHit");
    h1ex[i]  = fDBE->book1D (hname1, htitle1, nbin , low[0] , high[0]);
    fDBE->setCurrentFolder("TOBHit");
    h2ex[i]  = fDBE->book1D (hname2, htitle2, nbin , low[1] , high[1]);
    fDBE->setCurrentFolder("TIDHit");
    h3ex[i]  = fDBE->book1D (hname3, htitle3, nbin , low[2] , high[2]);
    fDBE->setCurrentFolder("TECHit");
    h4ex[i]  = fDBE->book1D (hname4, htitle4, nbin , low[3] , high[3]);
    fDBE->setCurrentFolder("BPIXHit");
    h5ex[i]  = fDBE->book1D (hname5, htitle5, nbin , low[4] , high[4]);
    fDBE->setCurrentFolder("FPIXHit");
    h6ex[i]  = fDBE->book1D (hname6, htitle6, nbin , low[5] , high[5]);
   
   }

   for(int i=0; i<12; i++) {
  
    sprintf (htitle1,"Entryy-Exity in TIB %s", Region[i]);
    sprintf (htitle2,"Entryy-Exity in TOB %s", Region[i]);
    sprintf (htitle3,"Entryy-Exity in TID %s", Region[i]);
    sprintf (htitle4,"Entryy-Exity in TEC %s", Region[i]);
    sprintf (htitle5,"Entryy-Exity in BPIX %s", Region[i]);
    sprintf (htitle6,"Entryy-Exity in FPIX %s", Region[i]);
    
    sprintf (hname1,"Entryy-Exity_TIB_%i",i+1);
    sprintf (hname2,"Entryy-Exity_TOB_%i",i+1);
    sprintf (hname3,"Entryy-Exity_TID_%i",i+1);
    sprintf (hname4,"Entryy-Exity_TEC_%i",i+1);
    sprintf (hname5,"Entryy-Exity_BPIX_%i",i+1);
    sprintf (hname6,"Entryy-Exity_FPIX_%i",i+1);
   
    fDBE->setCurrentFolder("TIBHit");
    h1ey[i]  = fDBE->book1D (hname1, htitle1, nbin , low[0] , high[0]);
    fDBE->setCurrentFolder("TOBHit");
    h2ey[i]  = fDBE->book1D (hname2, htitle2, nbin , low[1] , high[1]);
    fDBE->setCurrentFolder("TIDHit");
    h3ey[i]  = fDBE->book1D (hname3, htitle3, nbin , low[2] , high[2]);
    fDBE->setCurrentFolder("TECHit");
    h4ey[i]  = fDBE->book1D (hname4, htitle4, nbin , low[3] , high[3]);
    fDBE->setCurrentFolder("BPIXHit");
    h5ey[i]  = fDBE->book1D (hname5, htitle5, nbin , low[4] , high[4]);
    fDBE->setCurrentFolder("FPIXHit");
    h6ey[i]  = fDBE->book1D (hname6, htitle6, nbin , low[5] , high[5]);
   
   }

const float high1[] = {0.05, 0.06, 0.05, 0.06, 0.05, 0.05};
const float low1[]  = {0.,0.,0.,0.,0.,0.};

  for(int i=0; i<12; i++) {
  
    sprintf (htitle1,"abs(Entryz-Exitz) in TIB %s", Region[i]);
    sprintf (htitle2,"abs(Entryz-Exitz) in TOB %s", Region[i]);
    sprintf (htitle3,"abs(Entryz-Exitz) in TID %s", Region[i]);
    sprintf (htitle4,"abs(Entryz-Exitz) in TEC %s", Region[i]);
    sprintf (htitle5,"abs(Entryz-Exitz) in BPIX %s", Region[i]);
    sprintf (htitle6,"abs(Entryz-Exitz) in FPIX %s", Region[i]);
    
    sprintf (hname1,"Entryz-Exitz_TIB_%i",i+1);
    sprintf (hname2,"Entryz-Exitz_TOB_%i",i+1);
    sprintf (hname3,"Entryz-Exitz_TID_%i",i+1);
    sprintf (hname4,"Entryz-Exitz_TEC_%i",i+1);
    sprintf (hname5,"Entryz-Exitz_BPIX_%i",i+1);
    sprintf (hname6,"Entryz-Exitz_FPIX_%i",i+1);
   
    fDBE->setCurrentFolder("TIBHit");
    h1ez[i]  = fDBE->book1D (hname1, htitle1, nbin , low1[0] , high1[0]);
    fDBE->setCurrentFolder("TOBHit");
    h2ez[i]  = fDBE->book1D (hname2, htitle2, nbin , low1[1] , high1[1]);
    fDBE->setCurrentFolder("TIDHit");
    h3ez[i]  = fDBE->book1D (hname3, htitle3, nbin , low1[2] , high1[2]);
    fDBE->setCurrentFolder("TECHit");
    h4ez[i]  = fDBE->book1D (hname4, htitle4, nbin , low1[3] , high1[3]);
    fDBE->setCurrentFolder("BPIXHit");
    h5ez[i]  = fDBE->book1D (hname5, htitle5, nbin , low1[4] , high1[4]);
    fDBE->setCurrentFolder("FPIXHit");
    h6ez[i]  = fDBE->book1D (hname6, htitle6, nbin , low1[5] , high1[5]);
   
   }


const float high2[] = {3.2, 5.0, 5.5, 6.2, 0.85, 0.5};   
const float low2[] = {-3.2, -5.0, -5.5, -6.2, -0.85, -0.5};   

   for(int i=0; i<12; i++) {
  
    sprintf (htitle1,"Localx in TIB %s", Region[i]);
    sprintf (htitle2,"Localx in TOB %s", Region[i]);
    sprintf (htitle3,"Localx in TID %s", Region[i]);
    sprintf (htitle4,"Localx in TEC %s", Region[i]);
    sprintf (htitle5,"Localx in BPIX %s", Region[i]);
    sprintf (htitle6,"Localx in FPIX %s", Region[i]);
    
    sprintf (hname1,"Localx_TIB_%i",i+1);
    sprintf (hname2,"Localx_TOB_%i",i+1);
    sprintf (hname3,"Localx_TID_%i",i+1);
    sprintf (hname4,"Localx_TEC_%i",i+1);
    sprintf (hname5,"Localx_BPIX_%i",i+1);
    sprintf (hname6,"Localx_FPIX_%i",i+1);
   
    fDBE->setCurrentFolder("TIBHit");
    h1lx[i]  = fDBE->book1D (hname1, htitle1, nbin , low2[0] , high2[0]);
    fDBE->setCurrentFolder("TOBHit");
    h2lx[i]  = fDBE->book1D (hname2, htitle2, nbin , low2[1] , high2[1]);
    fDBE->setCurrentFolder("TIDHit");
    h3lx[i]  = fDBE->book1D (hname3, htitle3, nbin , low2[2] , high2[2]);
    fDBE->setCurrentFolder("TECHit");
    h4lx[i]  = fDBE->book1D (hname4, htitle4, nbin , low2[3] , high2[3]);
    fDBE->setCurrentFolder("BPIXHit");
    h5lx[i]  = fDBE->book1D (hname5, htitle5, nbin , low2[4] , high2[4]);
    fDBE->setCurrentFolder("FPIXHit");
    h6lx[i]  = fDBE->book1D (hname6, htitle6, nbin , low2[5] , high2[5]);
   
   }


const float high3[] = {6.0, 10., 5.6, 10.5, 3.4, 0.52};
const float low3[] = {-6.0, -10., -5.6, -10.5, -3.4, -0.52};

   for(int i=0; i<12; i++) {
  
    sprintf (htitle1,"Localy in TIB %s", Region[i]);
    sprintf (htitle2,"Localy in TOB %s", Region[i]);
    sprintf (htitle3,"Localy in TID %s", Region[i]);
    sprintf (htitle4,"Localy in TEC %s", Region[i]);
    sprintf (htitle5,"Localy in BPIX %s", Region[i]);
    sprintf (htitle6,"Localy in FPIX %s", Region[i]);
    
    sprintf (hname1,"Localy_TIB_%i",i+1);
    sprintf (hname2,"Localy_TOB_%i",i+1);
    sprintf (hname3,"Localy_TID_%i",i+1);
    sprintf (hname4,"Localy_TEC_%i",i+1);
    sprintf (hname5,"Localy_BPIX_%i",i+1);
    sprintf (hname6,"Localy_FPIX_%i",i+1);
   
    fDBE->setCurrentFolder("TIBHit");
    h1ly[i]  = fDBE->book1D (hname1, htitle1, nbin , low3[0] , high3[0]);
    fDBE->setCurrentFolder("TOBHit");
    h2ly[i]  = fDBE->book1D (hname2, htitle2, nbin , low3[1] , high3[1]);
    fDBE->setCurrentFolder("TIDHit");
    h3ly[i]  = fDBE->book1D (hname3, htitle3, nbin , low3[2] , high3[2]);
    fDBE->setCurrentFolder("TECHit");
    h4ly[i]  = fDBE->book1D (hname4, htitle4, nbin , low3[3] , high3[3]);
    fDBE->setCurrentFolder("BPIXHit");
    h5ly[i]  = fDBE->book1D (hname5, htitle5, nbin , low3[4] , high3[4]);
    fDBE->setCurrentFolder("FPIXHit");
    h6ly[i]  = fDBE->book1D (hname6, htitle6, nbin , low3[5] , high3[5]);
   
   }
   
  }
}

TrackerHitAnalyzer::~TrackerHitAnalyzer()
{
   // don't try to delete any pointers - they're handled by DQM machinery
}

void TrackerHitAnalyzer::beginJob(const edm::EventSetup& c){

}

void TrackerHitAnalyzer::endJob() 
{

  if ( fOutputFile.size() != 0 && fDBE ) fDBE->save(fOutputFile);

  return ;

}


void TrackerHitAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c)
{

   LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
   
  // iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;
  /////////////////////////////////
  // get Pixel Barrel information
  ////////////////////////////////
  // extract low container
  edm::Handle<edm::PSimHitContainer> PxlBrlLowContainer;
  e.getByLabel(PxlBrlLowSrc_,PxlBrlLowContainer);
  if (!PxlBrlLowContainer.isValid()) {
    edm::LogError("TrackerHitAnalyzer::analyze")
      << "Unable to find TrackerHitsPixelBarrelLowTof in event!";
    return;
  }  
  // extract high container
  edm::Handle<edm::PSimHitContainer> PxlBrlHighContainer;
  e.getByLabel(PxlBrlHighSrc_,PxlBrlHighContainer);
  if (!PxlBrlHighContainer.isValid()) {
    edm::LogError("TrackerHitAnalyzer::analyze")
      << "Unable to find TrackerHitsPixelBarrelHighTof in event!";
    return;
  }
  /////////////////////////////////
  // get Pixel Forward information
  ////////////////////////////////
  // extract low container
  edm::Handle<edm::PSimHitContainer> PxlFwdLowContainer;
  e.getByLabel(PxlFwdLowSrc_,PxlFwdLowContainer);
  if (!PxlFwdLowContainer.isValid()) {
    edm::LogError("TrackerHitAnalyzer::analyze")
      << "Unable to find TrackerHitsPixelEndcapLowTof in event!";
    return;
  }
  // extract high container
  edm::Handle<edm::PSimHitContainer> PxlFwdHighContainer;
  e.getByLabel(PxlFwdHighSrc_,PxlFwdHighContainer);
  if (!PxlFwdHighContainer.isValid()) {
    edm::LogError("TrackerHitAnalyzer::analyze")
      << "Unable to find TrackerHitsPixelEndcapHighTof in event!";
    return;
  }
  
  ///////////////////////////////////
  // get Silicon TIB information
  //////////////////////////////////
  // extract TIB low container
  edm::Handle<edm::PSimHitContainer> SiTIBLowContainer;
//  iEvent.getByLabel("g4SimHits","TrackerHitsTIBLowTof",SiTIBLowContainer);
  e.getByLabel(SiTIBLowSrc_,SiTIBLowContainer);
  if (!SiTIBLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::analyze")
      << "Unable to find TrackerHitsTIBLowTof in event!";
    return;
  }
  //////////////////////////////////
  // extract TIB low container
  edm::Handle<edm::PSimHitContainer> SiTIBHighContainer;
//  iEvent.getByLabel("g4SimHits","TrackerHitsTIBHighTof",SiTIBHighContainer);
  e.getByLabel(SiTIBHighSrc_,SiTIBHighContainer);
  if (!SiTIBHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::analyze")
      << "Unable to find TrackerHitsTIBHighTof in event!";
    return;
  }
  ///////////////////////////////////
  // get Silicon TOB information
  //////////////////////////////////
  // extract TOB low container
  edm::Handle<edm::PSimHitContainer> SiTOBLowContainer;
//  iEvent.getByLabel("g4SimHits","TrackerHitsTOBLowTof",SiTOBLowContainer);
  e.getByLabel(SiTOBLowSrc_,SiTOBLowContainer);
  if (!SiTOBLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::analyze")
      << "Unable to find TrackerHitsTOBLowTof in event!";
    return;
  }
  //////////////////////////////////
  // extract TOB low container
  edm::Handle<edm::PSimHitContainer> SiTOBHighContainer;
//  iEvent.getByLabel("g4SimHits","TrackerHitsTOBHighTof",SiTOBHighContainer);
  e.getByLabel(SiTOBHighSrc_,SiTOBHighContainer);
  if (!SiTOBHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::analyze")
      << "Unable to find TrackerHitsTOBHighTof in event!";
    return;
  }
  
  ///////////////////////////////////
  // get Silicon TID information
  //////////////////////////////////
  // extract TID low container
  edm::Handle<edm::PSimHitContainer> SiTIDLowContainer;
//  iEvent.getByLabel("g4SimHits","TrackerHitsTIDLowTof",SiTIDLowContainer);
  e.getByLabel(SiTIDLowSrc_,SiTIDLowContainer);
  if (!SiTIDLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::analyze")
      << "Unable to find TrackerHitsTIDLowTof in event!";
    return;
  }
  //////////////////////////////////
  // extract TID low container
  edm::Handle<edm::PSimHitContainer> SiTIDHighContainer;
//  iEvent.getByLabel("g4SimHits","TrackerHitsTIDHighTof",SiTIDHighContainer);
  e.getByLabel(SiTIDHighSrc_,SiTIDHighContainer);
  if (!SiTIDHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::analyze")
      << "Unable to find TrackerHitsTIDHighTof in event!";
    return;
  }
  ///////////////////////////////////
  // get Silicon TEC information
  //////////////////////////////////
  // extract TEC low container
  edm::Handle<edm::PSimHitContainer> SiTECLowContainer;
//  iEvent.getByLabel("g4SimHits","TrackerHitsTECLowTof",SiTECLowContainer);
  e.getByLabel(SiTECLowSrc_,SiTECLowContainer);
  if (!SiTECLowContainer.isValid()) {
    edm::LogError("TrackerHitProducer::analyze")
      << "Unable to find TrackerHitsTECLowTof in event!";
    return;
  }
  //////////////////////////////////
  // extract TEC low container
  edm::Handle<edm::PSimHitContainer> SiTECHighContainer;
//  iEvent.getByLabel("g4SimHits","TrackerHitsTECHighTof",SiTECHighContainer);
  e.getByLabel(SiTECHighSrc_,SiTECHighContainer);
  if (!SiTECHighContainer.isValid()) {
    edm::LogError("TrackerHitProducer::analyze")
      << "Unable to find TrackerHitsTECHighTof in event!";
    return;
  }
  
  ///////////////////////////
  // get G4Track information
  ///////////////////////////
  
  edm::Handle<edm::SimTrackContainer> G4TrkContainer;
  e.getByType(G4TrkContainer);
  if (!G4TrkContainer.isValid()) {
    edm::LogError("TrackerHitAnalyzer::analyze")
      << "Unable to find SimTrack in event!";
    return;
  }
  int ir = -100;
  edm::SimTrackContainer::const_iterator itTrk;
  for (itTrk = G4TrkContainer->begin(); itTrk != G4TrkContainer->end(); 
       ++itTrk) {

//    cout << "itTrk = "<< itTrk << endl;
    double eta =0, phi =0, p =0;
    const HepLorentzVector& G4Trk = itTrk->momentum();
    p =sqrt(G4Trk[0]*G4Trk[0]+G4Trk[1]*G4Trk[1]+G4Trk[2]*G4Trk[2]);
      if ( p == 0) 
          edm::LogError("TrackerHitAnalyzer::analyze") 
          << "TrackerTest::INFO: Primary has p = 0 ";
      else {
          double costheta  = G4Trk[2]/p;
          double theta = acos(TMath::Min(TMath::Max(costheta, -1.),1.));
          eta = -log(tan(theta/2));          
          if ( G4Trk[0] != 0 || G4Trk[1] != 0) phi = atan2(G4Trk[1],G4Trk[0]);
          
          if (eta>0.0 && eta<=0.5) ir = 0;
          if (eta>0.5 && eta<=1.0) ir = 1;
          if (eta>1.0 && eta<=1.5) ir = 2;
          if (eta>1.5 && eta<=2.0) ir = 3;
          if (eta>2.0 && eta<=2.5) ir = 4;
          if (eta>2.5) ir = 5;
	  
          if (eta>-0.5 && eta<= 0.0) ir = 6;
          if (eta>-1.0 && eta<=-0.5) ir = 7;
          if (eta>-1.5 && eta<=-1.0) ir = 8;
          if (eta>-2.0 && eta<=-1.5) ir = 9;
          if (eta>-2.5 && eta<=-2.0) ir = 10;
          if (eta<=-2.5) ir = 11;
//          LogInfo("EventInfo") << " eta = " << eta << " ir = " << ir;
//	  cout << " " <<endl;
//          cout << "eta " << eta << " ir = " << ir << endl;                  
//	  cout << " " <<endl;
      }
  }	  
  ///////////////////////////////
  // get Pixel information
  ///////////////////////////////
  for (itHit = PxlBrlLowContainer->begin(); itHit != PxlBrlLowContainer->end(); ++itHit) {
   h5e[ir]->Fill(itHit->energyLoss());
   h5ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h5ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h5ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h5lx[ir]->Fill(itHit->localPosition().x());
   h5ly[ir]->Fill(itHit->localPosition().y());
  }
  for (itHit = PxlBrlHighContainer->begin(); itHit != PxlBrlHighContainer->end(); ++itHit) {
   h5e[ir]->Fill(itHit->energyLoss());
   h5ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h5ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h5ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h5lx[ir]->Fill(itHit->localPosition().x());
   h5ly[ir]->Fill(itHit->localPosition().y());
  }
  for (itHit = PxlFwdLowContainer->begin(); itHit != PxlFwdLowContainer->end(); ++itHit) {
   h6e[ir]->Fill(itHit->energyLoss());
   h6ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h6ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h6ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h6lx[ir]->Fill(itHit->localPosition().x());
   h6ly[ir]->Fill(itHit->localPosition().y());
  }  
  for (itHit = PxlFwdHighContainer->begin(); itHit != PxlFwdHighContainer->end(); ++itHit) {
   h6e[ir]->Fill(itHit->energyLoss());
   h6ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h6ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h6ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h6lx[ir]->Fill(itHit->localPosition().x());
   h6ly[ir]->Fill(itHit->localPosition().y());
  }
  ///////////////////////////////
  // get TIB information
  ///////////////////////////////
  for (itHit = SiTIBLowContainer->begin(); itHit != SiTIBLowContainer->end(); ++itHit) {
   h1e[ir]->Fill(itHit->energyLoss());
   h1ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h1ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h1ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h1lx[ir]->Fill(itHit->localPosition().x());
   h1ly[ir]->Fill(itHit->localPosition().y());
  }
  for (itHit = SiTIBHighContainer->begin(); itHit != SiTIBHighContainer->end(); ++itHit) {
   h1e[ir]->Fill(itHit->energyLoss());
   h1ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h1ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h1ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h1lx[ir]->Fill(itHit->localPosition().x());
   h1ly[ir]->Fill(itHit->localPosition().y());
  }
  ///////////////////////////////
  // get TOB information
  ///////////////////////////////
  for (itHit = SiTOBLowContainer->begin(); itHit != SiTOBLowContainer->end(); ++itHit) {
   h2e[ir]->Fill(itHit->energyLoss());
   h2ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h2ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h2ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h2lx[ir]->Fill(itHit->localPosition().x());
   h2ly[ir]->Fill(itHit->localPosition().y());
  }  
  for (itHit = SiTOBHighContainer->begin(); itHit != SiTOBHighContainer->end(); ++itHit) {
   h2e[ir]->Fill(itHit->energyLoss());
   h2ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h2ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h2ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h2lx[ir]->Fill(itHit->localPosition().x());
   h2ly[ir]->Fill(itHit->localPosition().y());
  }
  ///////////////////////////////
  // get TID information
  ///////////////////////////////
  for (itHit = SiTIDLowContainer->begin(); itHit != SiTIDLowContainer->end(); ++itHit) {
   h3e[ir]->Fill(itHit->energyLoss());
   h3ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h3ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h3ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h3lx[ir]->Fill(itHit->localPosition().x());
   h3ly[ir]->Fill(itHit->localPosition().y());
  }  
  for (itHit = SiTIDHighContainer->begin(); itHit != SiTIDHighContainer->end(); ++itHit) {
   h3e[ir]->Fill(itHit->energyLoss());
   h3ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h3ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h3ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h3lx[ir]->Fill(itHit->localPosition().x());
   h3ly[ir]->Fill(itHit->localPosition().y());
  }
  ///////////////////////////////
  // get TEC information
  ///////////////////////////////
  for (itHit = SiTECLowContainer->begin(); itHit != SiTECLowContainer->end(); ++itHit) {
   h4e[ir]->Fill(itHit->energyLoss());
   h4ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h4ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h4ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h4lx[ir]->Fill(itHit->localPosition().x());
   h4ly[ir]->Fill(itHit->localPosition().y());
  }  
  for (itHit = SiTECHighContainer->begin(); itHit != SiTECHighContainer->end(); ++itHit) {
   h4e[ir]->Fill(itHit->energyLoss());
   h4ex[ir]->Fill(itHit->entryPoint().x()-itHit->exitPoint().x());
   h4ey[ir]->Fill(itHit->entryPoint().y()-itHit->exitPoint().y());
   h4ez[ir]->Fill(std::fabs(itHit->entryPoint().z()-itHit->exitPoint().z()));
   h4lx[ir]->Fill(itHit->localPosition().x());
   h4ly[ir]->Fill(itHit->localPosition().y());
  }
  
      
   return ;
   
}

DEFINE_FWK_MODULE(TrackerHitAnalyzer);

