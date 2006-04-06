// -*- C++ -*-
//
// Package:    PixelDigisTest
// Class:      PixelDigisTest
// 
/**\class PixelDigisTest PixelDigisTest.cc 

 Description: Test pixel digis. Barrel only. Uses root histos.
 Works with CMSSW_0_5_0 
 Modifed for module() method in PXBDetId. 2/06

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  d.k.
//         Created:  Jan CET 2006
// $Id$
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

// my includes
//#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLinkCollection.h"

// For the big pixel recongnition
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

// for simulated Tracker hits
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

using namespace std;

//
// class decleration
//

class PixelDigisTest : public edm::EDAnalyzer {

public:

  explicit PixelDigisTest(const edm::ParameterSet&);
  ~PixelDigisTest();
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(); 

private:
  // ----------member data ---------------------------
  const static bool PRINT = false;

  TFile* hFile;
  TH1F *heloss1,*heloss2, *heloss3,*hdetunit;
  TH1F *hpixid,*hpixsubid,*hlayerid,*hladder1id,*hladder2id,*hladder3id,
    *hz1id,*hz2id,*hz3id;
  TH1F *hcols1,*hcols2,*hcols3,*hrows1,*hrows2,*hrows3;
  TH1F *hdigisPerDet1,*hdigisPerDet2,*hdigisPerDet3;
  TH1F *hdigisPerLay1,*hdigisPerLay2,*hdigisPerLay3;
  TH1F *hdetsPerLay1,*hdetsPerLay2,*hdetsPerLay3;
  TH1F *hdetr, *hdetz, *hdetrF, *hdetzF;
  TH1F *hcolsB,  *hrowsB,  *hcolsF,  *hrowsF;
  TH1F *hcols1big, *hrows1big, *heloss1bigx, *heloss1bigy;

  TH2F *htest, *htest2;

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
PixelDigisTest::PixelDigisTest(const edm::ParameterSet& iConfig) {
  //We put this here for the moment since there is no better place 
  //edm::Service<MonitorDaemon> daemon;
  //daemon.operator->();

  cout<<" Construct PixelDigisTest "<<endl;
}


PixelDigisTest::~PixelDigisTest() {
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  cout<<" Destroy PixelDigisTest "<<endl;

}

//
// member functions
//
// ------------ method called at the begining   ------------
void PixelDigisTest::beginJob(const edm::EventSetup& iSetup) {

   using namespace edm;
   cout << "Initialize PixelDigisTest " <<endl;


   // put here whatever you want to do at the beginning of the job
    hFile = new TFile ( "histo.root", "RECREATE" );
     
    hdetunit = new TH1F( "hdetunit", "Det unit", 1000,
                              302000000.,302300000.);
    hpixid = new TH1F( "hpixid", "Pix det id", 10, 0., 10.);
    hpixsubid = new TH1F( "hpixsubid", "Pix Barrel id", 10, 0., 10.);
    hlayerid = new TH1F( "hlayerid", "Pix layer id", 10, 0., 10.);
    hladder1id = new TH1F( "hladder1id", "Ladder L1 id", 50, 0., 50.);
    hladder2id = new TH1F( "hladder2id", "Ladder L2 id", 50, 0., 50.);
    hladder3id = new TH1F( "hladder3id", "Ladder L3 id", 50, 0., 50.);
    hz1id = new TH1F( "hz1id", "Z-index id L1", 10, 0., 10.);
    hz2id = new TH1F( "hz2id", "Z-index id L2", 10, 0., 10.);
    hz3id = new TH1F( "hz3id", "Z-index id L3", 10, 0., 10.);
 
    hdigisPerDet1 = new TH1F( "hdigisPerDet1", "Digis per det l1", 
			      200, -0.5, 199.5);
    hdigisPerDet2 = new TH1F( "hdigisPerDet2", "Digis per det l2", 
			      200, -0.5, 199.5);
    hdigisPerDet3 = new TH1F( "hdigisPerDet3", "Digis per det l3", 
			      200, -0.5, 199.5);
    hdigisPerLay1 = new TH1F( "hdigisPerLay1", "Digis per layer l1", 
			      2000, -0.5, 1999.5);
    hdigisPerLay2 = new TH1F( "hdigisPerLay2", "Digis per layer l2", 
			      2000, -0.5, 1999.5);
    hdigisPerLay3 = new TH1F( "hdigisPerLay3", "Digis per layer l3", 
			      2000, -0.5, 1999.5);
    hdetsPerLay1 = new TH1F( "hdetsPerLay1", "Full dets per layer l1", 
			      161, -0.5, 160.5);
    hdetsPerLay3 = new TH1F( "hdetsPerLay3", "Full dets per layer l3", 
			      353, -0.5, 352.5);
    hdetsPerLay2 = new TH1F( "hdetsPerLay2", "Full dets per layer l2", 
			      257, -0.5, 256.5);

    heloss1 = new TH1F( "heloss1", "Pix charge l1", 100, 0., 300.);
    heloss2 = new TH1F( "heloss2", "Pix charge l2", 100, 0., 300.);
    heloss3 = new TH1F( "heloss3", "Pix charge l3", 100, 0., 300.);
    heloss1bigx = new TH1F( "heloss1bigx", "L1 big-x pix", 100, 0., 300.);
    heloss1bigy = new TH1F( "heloss1bigy", "L1 big-y pix", 100, 0., 300.);

    hcols1 = new TH1F( "hcols1", "Layer 1 cols", 500,-1.5,498.5);
    hcols2 = new TH1F( "hcols2", "Layer 2 cols", 500,-1.5,498.5);
    hcols3 = new TH1F( "hcols3", "Layer 3 cols", 500,-1.5,498.5);
    hcols1big = new TH1F( "hcols1big", "Layer 1 big cols", 500,-1.5,498.5);
 
    hrows1 = new TH1F( "hrows1", "Layer 1 rows", 200,-1.5,198.5);
    hrows2 = new TH1F( "hrows2", "Layer 2 rows", 200,-1.5,198.5);
    hrows3 = new TH1F( "hrows3", "layer 3 rows", 200,-1.5,198.5);
    hrows1big = new TH1F( "hrows1big", "Layer 1 big rows", 200,-1.5,198.5);
 
    hdetr = new TH1F("hdetr","det r",150,0.,15.);
    hdetz = new TH1F("hdetz","det z",520,-26.,26.);
    hdetrF = new TH1F("hdetrF","det r",150,0.,15.);
    hdetzF = new TH1F("hdetzF","det z",700,-70.,70.);

    hcolsB = new TH1F("hcolsB","cols per bar det",450,0.,450.);
    hrowsB = new TH1F("hrowsB","rows per bar det",200,0.,200.);
    hcolsF = new TH1F("hcolsF","cols per for det",300,0.,300.);
    hrowsF = new TH1F("hrowsF","rows per for det",200,0.,200.);

    htest = new TH2F("htest"," ",10,0.,10.,20,0.,20.);
    htest2 = new TH2F("htest2"," ",10,0.,10.,300,0.,300.);

}

// ------------ method called to produce the data  ------------
void PixelDigisTest::analyze(const edm::Event& iEvent, 
			   const edm::EventSetup& iSetup) {
  using namespace edm;
  if(PRINT) cout<<" Analyze PixelDigisTest "<<endl;
  
  // Get digis
  edm::Handle<PixelDigiCollection> pixelDigis;
  iEvent.getByLabel("pixdigi", pixelDigis);

  // Get simlink data
  edm::Handle<PixelDigiSimLinkCollection> pixelSimLinks;
  iEvent.getByLabel("pixdigi", pixelSimLinks);

  // Get event setup (to get global transformation)
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker(*geom);

  int numberOfDetUnits = 0;
  int totalNumOfDigis = 0;
  int numberOfDetUnits1 = 0;
  int totalNumOfDigis1 = 0;
  int numberOfDetUnits2 = 0;
  int totalNumOfDigis2 = 0;
  int numberOfDetUnits3 = 0;
  int totalNumOfDigis3 = 0;
  int numOfDigisPerDet1 = 0;
  int numOfDigisPerDet2 = 0;
  int numOfDigisPerDet3 = 0;
  
   // Get vector of detIDs in map
   const std::vector<unsigned int> detIDs = pixelDigis->detIDs();

   //--- Loop over detunits.
   std::vector<unsigned int>::const_iterator detunit_it;
   for (detunit_it  = detIDs.begin(); detunit_it != detIDs.end(); 
	detunit_it++ ) {
     unsigned int detid = *detunit_it; // return uint, = rawid

     // Det id
     DetId detId = DetId(detid);       // Get the Detid
     unsigned int detType=detId.det(); // det type, tracker=1
     unsigned int subid=detId.subdetId(); //subdetector type, barrel=1

     if(PRINT) 
       cout<<detId.rawId()<<" "<<detId.null()<<" "<<detType<<" "<<subid<<endl;

     hdetunit->Fill(float(detid));
     hpixid->Fill(float(detType));
     hpixsubid->Fill(float(subid));

     if(detType!=1) continue; // look only at tracker
 
     // Get the geom-detector 
     const PixelGeomDetUnit * theGeomDet = 
       dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) );
     double detZ = theGeomDet->surface().position().z();
     double detR = theGeomDet->surface().position().perp();
      //const BoundPlane plane = theGeomDet->surface(); // does not work

     double detThick = theGeomDet->specificSurface().bounds().thickness();
     int cols = theGeomDet->specificTopology().ncolumns();
     int rows = theGeomDet->specificTopology().nrows();
 
    // Subdet it, pix barrel=1
     if(subid != 1) {
       if(subid==2) {
	 // test cols & rows
	 //cout<<" det z/r "<<detZ<<" "<<detR<<" "<<detThick<<" "
	 //  <<cols<<" "<<rows<<endl;
	 hdetrF->Fill(detR);
	 hdetzF->Fill(detZ);
	 hcolsF->Fill(float(cols));
	 hrowsF->Fill(float(rows));

	 if(PRINT) cout<<" forward hit "<<endl;
       }
       continue; // look only at barrel
     }

     ++numberOfDetUnits;

     if(PRINT) cout<<" det z/r "<<detZ<<" "<<detR<<" "<<detThick<<" "
		   <<cols<<" "<<rows<<endl;

     hdetr->Fill(detR);
     hdetz->Fill(detZ);
     hcolsB->Fill(float(cols));
     hrowsB->Fill(float(rows));
     
     PXBDetId pdetId = PXBDetId(detid);
     unsigned int detTypeP=pdetId.det();
     unsigned int subidP=pdetId.subdetId();
     // Barell layer = 1,2,3
     unsigned int layer=pdetId.layer();
     // Barrel ladder id 1-20,32,44.
     unsigned int ladder=pdetId.ladder();
     // Barrel Z-index=1,8
     unsigned int zindex=pdetId.module();
     if(PRINT) 
       cout<<detTypeP<<" "<<subidP<<" "<<layer<<" "<<ladder<<" "<<zindex<<" "
	   <<pdetId.rawId()<<" "<<pdetId.null()<<endl;
     
     // Some histos
     hlayerid->Fill(float(layer));
     if(layer==1) {
       hladder1id->Fill(float(ladder));
       hz1id->Fill(float(zindex));
       ++numberOfDetUnits1;
       numOfDigisPerDet1=0;
       
     } else if(layer==2) {
        hladder2id->Fill(float(ladder));
        hz2id->Fill(float(zindex));
	++numberOfDetUnits2;
	numOfDigisPerDet2=0;
     } else if(layer==3) {
        hladder3id->Fill(float(ladder));
        hz3id->Fill(float(zindex));
	++numberOfDetUnits3;
	numOfDigisPerDet3=0;
     }
 
     const PixelDigiSimLinkCollection::Range simLinkRange = 
       pixelSimLinks->get(detid);
     int numberOfSimLinks = 0;
     // Loop over DigisSimLink in this det unit
     for(PixelDigiSimLinkCollection::ContainerIterator 
	   it = simLinkRange.first; it != simLinkRange.second; ++it) { 
       numberOfSimLinks++;
       // these methods should be declared const, fixed by M.P.
       // wait for next releasse and then uncomment
       //unsigned int chan = it->channel();
       //unsigned int simTrack = it->SimTrackId();
       //float frac = it->fraction();
       //cout<<" Sim link "<<numberOfSimLinks<<" "<<chan<<" "
       //  <<simTrack<<" "<<frac<<endl;
       // I should probably load it in a map so I can use it later
       // (below) with the digis.
     }

     unsigned int numberOfDigis = 0;
     // Look at digis now
     const PixelDigiCollection::Range digiRange = pixelDigis->get(detid);
     PixelDigiCollection::ContainerIterator di;
     // Loop over Digis in this det unit
     for(di = digiRange.first; di != digiRange.second; ++di) { 
       numberOfDigis++;
       totalNumOfDigis++;
       int adc = di->adc();    // charge
       int col = di->column(); // column 
       int row = di->row();    // row
       //int tof = di->time();    // tof always 0

       // channel index needed to look for the simlink to simtracks
       int channel = PixelChannelIdentifier::pixelToChannel(row,col);
       //cout <<numberOfDigis<< " Col: " << col << " Row: " << row 
       //   << " ADC: " << adc <<" chanel = "<<channel<<endl;

       if(col>415) cout<<" Error: column index too large "<<col<<endl;
       if(row>159) cout<<" Error: row index too large "<<row<<endl;

       if(layer==1) {
	 heloss1->Fill(float(adc));
	 hcols1->Fill(float(col));
	 hrows1->Fill(float(row));
	 totalNumOfDigis1++;
	 htest2->Fill(float(zindex),float(adc));
	 numOfDigisPerDet1++;
	 if(RectangularPixelTopology::isItBigPixelInX(row)) {
	   //cout<<" big in X "<<row<<endl;
	   heloss1bigx->Fill(float(adc));
	   hrows1big->Fill(float(row));
	 }
	 if(RectangularPixelTopology::isItBigPixelInY(col)) {
	   //cout<<" big in Y "<<col<<endl;
	   heloss1bigy->Fill(float(adc));
	   hcols1big->Fill(float(col));
	 }

       } else if(layer==2) {
	 heloss2->Fill(float(adc));
	 hcols2->Fill(float(col));
	 hrows2->Fill(float(row));
	 totalNumOfDigis2++;
	 numOfDigisPerDet1++;
       } else if(layer==3) {
	 heloss3->Fill(float(adc));
	 hcols3->Fill(float(col));
	 hrows3->Fill(float(row));
	 totalNumOfDigis3++;
	 numOfDigisPerDet1++;
       } // end if layer
        
     } // end for digis
     if(PRINT) 
       cout<<" for det "<<detid<<" digis = "<<numberOfDigis<<endl;

     if(layer==1) {
       hdigisPerDet1->Fill(float(numberOfDigis));
       htest->Fill(float(zindex),float(numberOfDigis));
     } else if(layer==2) hdigisPerDet2->Fill(float(numberOfDigis));
     else if(layer==3) hdigisPerDet3->Fill(float(numberOfDigis));

   } // end for det-units

   if(PRINT) 
     cout << " Number of full det-units = " <<numberOfDetUnits
	  <<" total digis = "<<totalNumOfDigis<<endl;

   hdigisPerLay1 ->Fill(float(totalNumOfDigis1));
   hdigisPerLay2 ->Fill(float(totalNumOfDigis2));
   hdigisPerLay3 ->Fill(float(totalNumOfDigis3));
   hdetsPerLay1 ->Fill(float(numberOfDetUnits1));
   hdetsPerLay2 ->Fill(float(numberOfDetUnits2));
   hdetsPerLay3 ->Fill(float(numberOfDetUnits3));

}
// ------------ method called to at the end of the job  ------------
void PixelDigisTest::endJob(){
  cout << " End PixelDigisTest " << endl;
  hFile->Write();
  hFile->Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelDigisTest)
