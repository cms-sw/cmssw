// -*- C++ -*-
//
// Package:    PixelSimHitsTest
// Class:      PixelSimHitsTest
// 
/**\class PixelSimHitsTest PixelSimHitsTest.cc 

 Description: Test pixel simhits. Forward only. Uses root histos.
 Works with CMSSW_0_7_0_pre 
 
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  d.k.
//         Created:  Jan CET 2006
// $Id: PixelSimHitsTestForward.cc,v 1.5 2009/11/13 14:14:23 fambrogl Exp $
//
//
// system include files
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

// my includes
//#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" //


#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//#include "Geometry/Surface/interface/Surface.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>
#include <TProfile.h>

using namespace std;
using namespace edm;

//
// class decleration
//

class PixelSimHitsTestForward : public edm::EDAnalyzer {

public:
  explicit PixelSimHitsTestForward(const edm::ParameterSet&);
  ~PixelSimHitsTestForward();
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(); 

private:
  // ----------member data ---------------------------
  const static bool PRINT = true;

  TFile* hFile;
  TH1F  *heloss1,*heloss2, *heloss3,*hdetunit,*hpabs,*hpid,*htof,*htid;
  TH1F* hpixid,*hpixsubid,*hlayerid,*hladder1id,*hladder2id,*hladder3id,
    *hz1id,*hz2id,*hz3id;
  TH1F *hladder1idUp, *hladder2idUp, *hladder3idUp;
  TH1F* hthick1,*hthick2,*hthick3,*hlength1,*hlength2,*hlength3;
  TH1F *hwidth1,*hwidth2,*hwidth3;
  TH1F *hwidth1h,*hwidth2h,*hwidth3h;
  TH1F *hsimHitsPerDet1,*hsimHitsPerDet2,*hsimHitsPerDet3;
  TH1F *hsimHitsPerLay1,*hsimHitsPerLay2,*hsimHitsPerLay3;
  TH1F *hdetsPerLay1,*hdetsPerLay2,*hdetsPerLay3;
  TH1F *heloss1e, *heloss2e, *heloss3e;
  TH1F *heloss1mu, *heloss2mu, *heloss3mu;
  TH1F *htheta1, *htheta2, *htheta3;
  TH1F *hphi1, *hphi1h, *hphi2, *hphi3;
  TH1F *hdetr, *hdetz, *hdetphi1, *hdetphi2, *hdetphi3;
  TH1F *hglobr1,*hglobr2,*hglobr3,*hglobz1, *hglobz2, *hglobz3;
  TH1F *hglobr1h;
  TH1F *hcolsB,  *hrowsB,  *hcolsF,  *hrowsF;
  TH1F *hglox1;

  TH2F *htest, *htest2, *htest3, *htest4, *htest5;

  TProfile *hp1, *hp2, *hp3, *hp4, *hp5;

  //float modulePositionZ[3][44][8];
  //float modulePositionR[3][44][8];
  //float modulePositionPhi[3][44][8];

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PixelSimHitsTestForward::PixelSimHitsTestForward(const edm::ParameterSet& iConfig) {
  //We put this here for the moment since there is no better place 
  //edm::Service<MonitorDaemon> daemon;
  //daemon.operator->();

  cout<<" Construct PixelSimHitsTestForward "<<endl;
}


PixelSimHitsTestForward::~PixelSimHitsTestForward() {
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  cout<<" Destroy PixelSimHitsTestForward "<<endl;

}

//
// member functions
//
// ------------ method called at the begining   ------------
void PixelSimHitsTestForward::beginJob() {

   using namespace edm;
   cout << "Initialize PixelSimHitsTestForward " <<endl;

   // put here whatever you want to do at the beginning of the job
   hFile = new TFile ( "simhistos.root", "RECREATE" );

   const float max_charge = 200.; // in ke 
   heloss1 = new TH1F( "heloss1", "Eloss l1", 100, 0., max_charge);
   heloss2 = new TH1F( "heloss2", "Eloss l2", 100, 0., max_charge);
   heloss3 = new TH1F( "heloss3", "Eloss l3", 100, 0., max_charge);

   hdetunit = new TH1F( "hdetunit", "Det unit", 1000,
                              302000000.,302300000.);
    hpabs = new TH1F( "hpabs", "Pabs", 100, 0., 100.);
    htof = new TH1F( "htof", "TOF", 50, -25., 25.);
    hpid = new TH1F( "hpid", "PID", 1000, 0., 1000.);
    htid = new TH1F( "htid", "Track id", 100, 0., 100.);
 
    hpixid = new TH1F( "hpixid", "Pix det id", 10, 0., 10.);
    hpixsubid = new TH1F( "hpixsubid", "Pix Barrel id", 10, 0., 10.);
    hlayerid = new TH1F( "hlayerid", "Pix layer id", 10, 0., 10.);
    hladder1id = new TH1F( "hladder1id", "Ladder L1 id", 100, -0.5, 49.5);
    hladder2id = new TH1F( "hladder2id", "Ladder L2 id", 100, -0.5, 49.5);
    hladder3id = new TH1F( "hladder3id", "Ladder L3 id", 100, -0.5, 49.5);
    hz1id = new TH1F( "hz1id", "Z-index id L1", 10, 0., 10.);
    hz2id = new TH1F( "hz2id", "Z-index id L2", 10, 0., 10.);
    hz3id = new TH1F( "hz3id", "Z-index id L3", 10, 0., 10.);
    
    hthick1 = new TH1F( "hthick1", "Det 1 Thinckess", 400, 0.,0.04);
    hthick2 = new TH1F( "hthick2", "Det 2 Thinckess", 400, 0.,0.04);
    hthick3 = new TH1F( "hthick3", "Det 3 Thinckess", 400, 0.,0.04);
                                                                                
    hlength1 = new TH1F( "hlength1", "Det 1 Length", 700,-3.5,3.5);
    hlength2 = new TH1F( "hlength2", "Det 2 Length", 700,-3.5,3.5);
    hlength3 = new TH1F( "hlength3", "Det 3 Length", 700,-3.5,3.5);
 
    hwidth1 = new TH1F( "hwidth1", "Det 1 Width", 200,-1.,1.);
    hwidth2 = new TH1F( "hwidth2", "Det 2 Width", 200,-1.,1.);
    hwidth3 = new TH1F( "hwidth3", "Det 3 Width", 200,-1.,1.);

    hwidth1h = new TH1F( "hwidth1h", "Det 1 Width half-m", 200,-1.,1.);
    hwidth2h = new TH1F( "hwidth2h", "Det 2 Width half-m", 200,-1.,1.);
    hwidth3h = new TH1F( "hwidth3h", "Det 3 Width half-m", 200,-1.,1.);

    hsimHitsPerDet1 = new TH1F( "hsimHitsPerDet1", "SimHits per det l1", 
			      200, -0.5, 199.5);
    hsimHitsPerDet2 = new TH1F( "hsimHitsPerDet2", "SimHits per det l2", 
			      200, -0.5, 199.5);
    hsimHitsPerDet3 = new TH1F( "hsimHitsPerDet3", "SimHits per det l3", 
			      200, -0.5, 199.5);
    hsimHitsPerLay1 = new TH1F( "hsimHitsPerLay1", "SimHits per layer l1", 
			      2000, -0.5, 1999.5);
    hsimHitsPerLay2 = new TH1F( "hsimHitsPerLay2", "SimHits per layer l2", 
			      2000, -0.5, 1999.5);
    hsimHitsPerLay3 = new TH1F( "hsimHitsPerLay3", "SimHits per layer l3", 
			      2000, -0.5, 1999.5);
    hdetsPerLay1 = new TH1F( "hdetsPerLay1", "Full dets per layer l1", 
			      161, -0.5, 160.5);
    hdetsPerLay3 = new TH1F( "hdetsPerLay3", "Full dets per layer l3", 
			      353, -0.5, 352.5);
    hdetsPerLay2 = new TH1F( "hdetsPerLay2", "Full dets per layer l2", 
			      257, -0.5, 256.5);
    heloss1e = new TH1F( "heloss1e", "Eloss e l1", 100, 0., max_charge);
    heloss2e = new TH1F( "heloss2e", "Eloss e l2", 100, 0., max_charge);
    heloss3e = new TH1F( "heloss3e", "Eloss e l3", 100, 0., max_charge);

    heloss1mu = new TH1F( "heloss1mu", "Eloss mu l1", 100, 0., max_charge);
    heloss2mu = new TH1F( "heloss2mu", "Eloss mu l2", 100, 0., max_charge);
    heloss3mu = new TH1F( "heloss3mu", "Eloss mu l3", 100, 0., max_charge);

    htheta1 = new TH1F( "htheta1", "Theta det1",350,0.0,3.5);
    htheta2 = new TH1F( "htheta2", "Theta det2",350,0.0,3.5);
    htheta3 = new TH1F( "htheta3", "Theta det3",350,0.0,3.5);
    hphi1 = new TH1F("hphi1","phi l1",1400,-3.5,3.5);
    hphi2 = new TH1F("hphi2","phi l2",1400,-3.5,3.5);
    hphi3 = new TH1F("hphi3","phi l3",1400,-3.5,3.5);
    hphi1h = new TH1F("hphi1h","phi l1",1400,-3.5,3.5);
 
    hdetr = new TH1F("hdetr","det r",1500,0.,15.);
    hdetz = new TH1F("hdetz","det z",5200,-26.,26.);
    hdetphi1 = new TH1F("hdetphi1","det phi l1",700,-3.5,3.5);
    hdetphi2 = new TH1F("hdetphi2","det phi l2",700,-3.5,3.5);
    hdetphi3 = new TH1F("hdetphi3","det phi l3",700,-3.5,3.5);

    hcolsB = new TH1F("hcolsB","cols per bar det",450,0.,450.);
    hrowsB = new TH1F("hrowsB","rows per bar det",200,0.,200.);
    hcolsF = new TH1F("hcolsF","cols per for det",300,0.,300.);
    hrowsF = new TH1F("hrowsF","rows per for det",200,0.,200.);

    hladder1idUp = new TH1F( "hladder1idUp", "Ladder L1 id", 100, -0.5, 49.5);
    hladder2idUp = new TH1F( "hladder2idUp", "Ladder L2 id", 100, -0.5, 49.5);
    hladder3idUp = new TH1F( "hladder3idUp", "Ladder L3 id", 100, -0.5, 49.5);

    hglobr1 = new TH1F("hglobr1","global r1",150,0.,15.);
    hglobz1 = new TH1F("hglobz1","global z1",540,-27.,27.);
    hglobr2 = new TH1F("hglobr2","global r2",150,0.,15.);
    hglobz2 = new TH1F("hglobz2","global z2",540,-27.,27.);
    hglobr3 = new TH1F("hglobr3","global r3",150,0.,15.);
    hglobz3 = new TH1F("hglobz3","global z3",540,-27.,27.);

    hglox1 = new TH1F("hglox1","global x in l1",200,-1.,1.);
    hglobr1h = new TH1F("hglobr1h","global r1",700,4.1,4.8);

    htest = new TH2F("htest"," ",108,-27.,27.,35,-3.5,3.5);
    htest2 = new TH2F("htest2"," ",108,-27.,27.,60,0.,600.);
    htest3 = new TH2F("htest3"," ",240,-12.,12.,240,-12.,12.);
    htest4 = new TH2F("htest4"," ",80,-4.,4.,100,-5.,5.);

    hp1 = new TProfile("hp1"," ",50,0.,50.);    // default option
    hp2 = new TProfile("hp2"," ",50,0.,50.," "); // option set to " "
    hp3 = new TProfile("hp3"," ",50,0.,50.,-100.,100.); // with y limits
    hp4 = new TProfile("hp4"," ",50,0.,50.);
    hp5 = new TProfile("hp5"," ",50,0.,50.);

    // To get the module position
//     for(int i=0;i<3;i++) {
//       for(int n=0;n<44;n++) {
// 	for(int m=0;m<8;m++) {
//  	  modulePositionR[i][n][m]=-1;
//  	  modulePositionZ[i][n][m]=-1;
//  	  modulePositionPhi[i][n][m]=-1;
// 	}
//       }
//     }

}

// ------------ method called to produce the data  ------------
void PixelSimHitsTestForward::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<IdealGeometryRecord>().get(tTopo);


  const double PI = 3.142;

  using namespace edm;
  if(PRINT) cout<<" Analyze PixelSimHitsTestForward "<<endl;
  
  // Get event setup (to get global transformation)
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker(*geom);
 
  // Get input data
  int totalNumOfSimHits = 0;
  int totalNumOfSimHits1 = 0;
  int totalNumOfSimHits2 = 0;
  int totalNumOfSimHits3 = 0;

  // To count simhits per det module 
   //typedef std::map<unsigned int, std::vector<PSimHit>,
   //std::less<unsigned int>> simhit_map;
   //typedef simhit_map::iterator simhit_map_iterator;
   //simhit_map SimHitMap;
   map<unsigned int, vector<PSimHit>, less<unsigned int> > SimHitMap1;
   map<unsigned int, vector<PSimHit>, less<unsigned int> > SimHitMap2;
   map<unsigned int, vector<PSimHit>, less<unsigned int> > SimHitMap3;

   Handle<PSimHitContainer> PixelForwardHitsLowTof;
   Handle<PSimHitContainer> PixelForwardHitsHighTof;

   iEvent.getByLabel("g4SimHits","TrackerHitsPixelEndcapLowTof",
		     PixelForwardHitsLowTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsPixelEndcapHighTof",
		     PixelForwardHitsHighTof);

   //vector<PSimHit> pixelHits;
   //pixelHits.insert(pixelHits.end(),PixelBarrelHitsLowTof->begin(),
   //       PixelBarrelHitsLowTof->end());

   //cout<<" size = "<<PixelForwardHitsLowTof->size()<<endl;

   //for(vector<PSimHit>::const_iterator isim = PixelBarrelHitsHighTof->begin();
   //  isim != PixelBarrelHitsHighTof->end(); ++isim){
   for(vector<PSimHit>::const_iterator isim = PixelForwardHitsLowTof->begin();
       isim != PixelForwardHitsLowTof->end(); ++isim){

     totalNumOfSimHits++;
     // Det id
     DetId detId=DetId((*isim).detUnitId());
     unsigned int detid=detId.det(); // for pixel=1
     unsigned int subid=detId.subdetId();// barrel=1, forward=2
     
     if(detid!=1 && subid!=2) cout<<" error in det id "<<detid<<" "
				  <<subid<<endl;
     //if(PRINT) cout<<" Forward det id "<<(*isim).detUnitId()<<" "<<detId.rawId()<<" "
     //	   <<detId.null()<<" "<<detid<<" "<<subid<<endl;

     //const GeomDetUnit * theGeomDet = theTracker.idToDet(detId);
     //const PixelGeomDetUnit * theGeomDet = theTracker.idToDet(detId);
     const PixelGeomDetUnit * theGeomDet = 
       dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) );
     double detZ = theGeomDet->surface().position().z();
     double detR = theGeomDet->surface().position().perp();
     double detPhi = theGeomDet->surface().position().phi();
     hdetr->Fill(detR);
     hdetz->Fill(detZ);

     //const BoundPlane plane = theGeomDet->surface(); // does not work

     double detThick = theGeomDet->specificSurface().bounds().thickness();
     double detLength = theGeomDet->specificSurface().bounds().length();
     double detWidth = theGeomDet->specificSurface().bounds().width();

     int cols = theGeomDet->specificTopology().ncolumns();
     int rows = theGeomDet->specificTopology().nrows();

     hcolsB->Fill(float(cols));
     hrowsB->Fill(float(rows));
     if(PRINT) cout<<"Forward det z/r "<<detZ<<" "<<detR<<" "<<detThick<<" "
		   <<detLength<<" "<<detWidth<<" "<<cols<<" "<<rows
		   <<endl;

     
     unsigned int disk=tTopo->pxfDisk(detId); //1,2,3
     unsigned int blade=tTopo->pxfBlade(detId); //1-24
     unsigned int zindex=tTopo->pxfModule(detId); //
     unsigned int side=tTopo->pxfSide(detId); //size=1 for -z, 2 for +z
     unsigned int panel=tTopo->pxfPanel(detId); //panel=1

     if(PRINT) cout<<"det "<<subid<<", disk "<<disk<<", blade "
		   <<blade<<", module "<<zindex<<", side "<<side<<", panel "
		   <<panel<<" pos = "<<detZ<<" "<<detR<<" "<<detPhi<<endl;

     // To get the module position
//      modulePositionR[disk-1][blade-1][zindex-1] = detR;
//      modulePositionZ[disk-1][blade-1][zindex-1] = detZ;
//      modulePositionPhi[disk-1][blade-1][zindex-1] = detPhi;

     // SimHit information 
     float eloss = (*isim).energyLoss() * 1000000/3.7;//convert GeV to ke 
     float tof = (*isim).timeOfFlight();
     float p = (*isim).pabs();
     float theta = (*isim).thetaAtEntry();
     float phi = (*isim).phiAtEntry();
     int pid = abs((*isim).particleType()); // ignore sign
     int tid = (*isim).trackId();
     int procType = (*isim).processType();
     
     float x = (*isim).entryPoint().x(); // width (row index, in col direction)
     float y = (*isim).entryPoint().y(); // length (col index, in row direction) 
     float z = (*isim).entryPoint().z(); // thick
     
     float x2 = (*isim).exitPoint().x();
     float y2 = (*isim).exitPoint().y();
     float z2 = (*isim).exitPoint().z();

     float dz = abs(z-z2);
     bool moduleDirectionUp = ( z < z2 ); // for positive direction z2>z

     float xpos = (x+x2)/2.;
     float ypos = (y+y2)/2.;
     float zpos = (z+z2)/2.;

     if(PRINT) cout<<"simhit "<<pid<<" "<<tid<<" "<<procType<<" "<<tof<<" "
		   <<eloss<<" "<<p<<" "<<x<<" "<<y<<" "<<z<<" "<<dz<<endl;
     if(PRINT) cout<<" pos "<<xpos<<" "<<ypos<<" "<<zpos;
     
     LocalPoint loc(xpos,ypos,zpos);
     //GlobalPoint glo = theGeomDet->surface().toGlobal(loc); // does not work!
     double gloX = theGeomDet->surface().toGlobal(loc).x(); // 
     double gloY = theGeomDet->surface().toGlobal(loc).y(); // 
     double gloR = theGeomDet->surface().toGlobal(loc).perp(); // 
     double gloZ = theGeomDet->surface().toGlobal(loc).z(); // 
     if(PRINT) cout<<", global "<<gloX<<" "<<gloY<<" "<<gloR<<" "<<gloZ<<endl;

     htest3->Fill(gloX,gloY);
     hdetunit->Fill(float(detId.rawId()));
     hpabs->Fill(p);
     htof->Fill(tof);
     hpid->Fill(float(pid));
     htid->Fill(float(tid));
     hpixid->Fill(float(detid));
     hpixsubid->Fill(float(subid));
     hlayerid->Fill(float(disk));
 
     // Transform the theta from local module coordinates to global
     //if(theta<= PI/2.) theta = PI/2. - theta; // For +z global
     //else theta = (PI/2. + PI) - theta;

     if(disk==1) {
       //cout<<" disk "<<disk<<endl;
       totalNumOfSimHits1++;
       heloss1->Fill(eloss);
       if(pid==11) heloss1e->Fill(eloss);
       else heloss1mu->Fill(eloss);	 
       hladder1id->Fill(float(blade));
       hz1id->Fill(float(zindex));
       hthick1->Fill(dz);
       hlength1->Fill(y);
       if(blade==5 || blade==6 || blade==15 || blade==16 ) {
	 // half-modules
	 hwidth1h->Fill(x);
	 if(pid==13 && p>1.) {  // select primary muons
	   hphi1h->Fill(phi);
	   hglox1->Fill(gloX);
	   hglobr1h->Fill(gloR);

// 	   double gloX1 = 
// 	     theGeomDet->surface().toGlobal(LocalPoint(0,0,0)).x(); // 
// 	   double gloY1 = 
// 	     theGeomDet->surface().toGlobal(LocalPoint(0,0,0)).y(); // 
// 	   double gloR1 = 
// 	     theGeomDet->surface().toGlobal(LocalPoint(0,0,0)).perp();
// 	   cout<<" "<<blade<<" "<<gloX1<<" "<<gloY1<<" "<<gloR1<<" "
// 	       <<detR<<" "<<detPhi<<" "<<detZ<<" "<<gloX<<" "<<gloY<<" "
// 	       <<xpos<<" "<<ypos<<" "<<zpos<<endl;


	 }
       } else {
	 hwidth1->Fill(x);
	 if(pid==13 && p>1.) hphi1->Fill(phi);
       }
       SimHitMap1[detId.rawId()].push_back((*isim));
       htheta1->Fill(theta);
       hglobr1->Fill(gloR);
       hglobz1->Fill(gloZ);

       // Check the coordinate system and counting
       htest->Fill(gloZ,ypos);
       if(pid != 11) htest2->Fill(gloZ,eloss);

       if(pid!=11 && moduleDirectionUp) 
	 hladder1idUp->Fill(float(blade));

       if(blade==6) htest4->Fill(xpos,gloX);
       hp1->Fill(float(blade),detR,1);
       hp2->Fill(float(blade),detPhi);
       hdetphi1->Fill(detPhi);

     } else if(disk==2) {
       //cout<<" disk "<<disk<<endl;
       totalNumOfSimHits2++;
       heloss2->Fill(eloss);
       if(pid==11) heloss2e->Fill(eloss);
       else heloss2mu->Fill(eloss);	 
       hladder2id->Fill(float(blade));
       hz2id->Fill(float(zindex));
       hthick2->Fill(dz);
       hlength2->Fill(y);
       if(blade==8 || blade==9 || blade==24 || blade==25 ) {
	 hwidth2h->Fill(x);
       } else {
	 hwidth2->Fill(x);
	 if(pid==13 && p>1.) hphi2->Fill(phi);
       }
       SimHitMap2[detId.rawId()].push_back((*isim));
       hglobr2->Fill(gloR);
       hglobz2->Fill(gloZ);
       hdetphi2->Fill(detPhi);
       if(pid!=11 && moduleDirectionUp) hladder2idUp->Fill(float(blade));

     } else if(disk==3) {
       //cout<<" disk "<<disk<<endl;
       totalNumOfSimHits3++;
       heloss3->Fill(eloss);
       if(pid==11) heloss3e->Fill(eloss);
       else heloss3mu->Fill(eloss);	 

       hladder3id->Fill(float(blade));
       hz3id->Fill(float(zindex));
       hthick3->Fill(dz);
       hlength3->Fill(y);
       if(blade==11 || blade==12 || blade==33 || blade==34 ) {
	 hwidth3h->Fill(x);
       } else {
	 hwidth3->Fill(x); 
	 if(pid==13 && p>1.) hphi3->Fill(phi);
       }
       SimHitMap3[detId.rawId()].push_back((*isim));
       hglobr3->Fill(gloR);
       hglobz3->Fill(gloZ);
       hdetphi3->Fill(detPhi);
       if(pid!=11 && moduleDirectionUp) hladder3idUp->Fill(float(blade));
     }
   }


   hsimHitsPerLay1 ->Fill(float(totalNumOfSimHits1));
   hsimHitsPerLay2 ->Fill(float(totalNumOfSimHits2));
   hsimHitsPerLay3 ->Fill(float(totalNumOfSimHits3));

   int numberOfDetUnits1 = SimHitMap1.size();
   int numberOfDetUnits2 = SimHitMap2.size();
   int numberOfDetUnits3 = SimHitMap3.size();
   int numberOfDetUnits = numberOfDetUnits1+numberOfDetUnits2+
     numberOfDetUnits3;

   if(PRINT) 
     cout << " Number of full det-units = " <<numberOfDetUnits
	  <<" total simhits = "<<totalNumOfSimHits<<endl;

   hdetsPerLay1 ->Fill(float(numberOfDetUnits1));
   hdetsPerLay2 ->Fill(float(numberOfDetUnits2));
   hdetsPerLay3 ->Fill(float(numberOfDetUnits3));

   map<unsigned int, vector<PSimHit>, less<unsigned int> >::iterator 
     simhit_map_iterator;
   for(simhit_map_iterator = SimHitMap1.begin(); 
       simhit_map_iterator != SimHitMap1.end(); simhit_map_iterator++) {
     if(PRINT) cout << " Lay1 det = "<<simhit_map_iterator->first <<" simHits = "
	  << (simhit_map_iterator->second).size() 
	  << endl;
     hsimHitsPerDet1->Fill( float((simhit_map_iterator->second).size()) );
   }
   for(simhit_map_iterator = SimHitMap2.begin(); 
       simhit_map_iterator != SimHitMap2.end(); simhit_map_iterator++) {
     if(PRINT) cout << " Lay2 det = "<<simhit_map_iterator->first <<" simHits = "
	  << (simhit_map_iterator->second).size() 
	  << endl;
     hsimHitsPerDet2->Fill( float((simhit_map_iterator->second).size()) );
   }
   for(simhit_map_iterator = SimHitMap3.begin(); 
       simhit_map_iterator != SimHitMap3.end(); simhit_map_iterator++) {
     if(PRINT) cout << " Lay3 det = "<<simhit_map_iterator->first <<" simHits = "
	  << (simhit_map_iterator->second).size() 
	  << endl;
     hsimHitsPerDet3->Fill( float((simhit_map_iterator->second).size()) );
   }


}
// ------------ method called to at the end of the job  ------------
void PixelSimHitsTestForward::endJob(){
  cout << " End PixelSimHitsTestForward " << endl;
  hFile->Write();
  hFile->Close();

//   // To get module positions
//   cout<< " Module position"<<endl;
//   cout<<" Layer Ladder Zindex    R      Z      Phi "<<endl; 
//   for(int i=0;i<3;i++) {
//     int max_lad=0;
//     if(i==0) max_lad=20;
//     else if(i==1) max_lad=32;
//     else if(i==2) max_lad=44;
//     for(int n=0;n<max_lad;n++) {
//       for(int m=0;m<8;m++) {
// 	cout<<"   "<<i+1<<"      "<<n+1<<"      "<<m+1<<"    "
// 	    <<modulePositionR[i][n][m]<<" "
// 	    <<modulePositionZ[i][n][m]<<" "<<modulePositionPhi[i][n][m]<<endl;
//       }
//     }
//   }


}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelSimHitsTestForward);

