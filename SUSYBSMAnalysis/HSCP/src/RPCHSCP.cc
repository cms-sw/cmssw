// -*- C++ -*-
//
// Package:    RPCHSCP
// Class:      RPCHSCP
// 
/**\class RPCHSCP RPCHSCP.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Camilo Carrillo camilo.carrillo AT cern.ch
//         Created:  Wed Aug  6 17:45:45 CEST 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "FWCore/Framework/interface/ESHandle.h"
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
//Root
#include "TFile.h"
#include "TF1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TMath.h"
#include "TCanvas.h"
#include <fstream>


//
// class decleration
//

class RPCHSCP : public edm::EDProducer {
   public:
      explicit RPCHSCP(const edm::ParameterSet&);
      ~RPCHSCP();
  


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::ofstream fileMatrix;
      std::string rootFileName;
      std::string fileMatrixName;
      //int matrixbitTOTAL[4][7];
      TFile* theFile;
      TH1F * matrixHisto;
      int Maxbx[7];
      int totalHSCP;
      std::string fileMatrixname; 
      // ----------member data ---------------------------
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
RPCHSCP::RPCHSCP(const edm::ParameterSet& iConfig)
{
  
  fileMatrixName= iConfig.getUntrackedParameter<std::string>("fileMatrixname");
  rootFileName = iConfig.getUntrackedParameter<std::string>("rootFileName");
  produces<int>("JustATest");
  
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/
   //now do what ever other initialization is needed

}


RPCHSCP::~RPCHSCP()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RPCHSCP::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel("rpcRecHits",rpcHits);
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  RPCRecHitCollection::const_iterator recIt;
  
  //Identifing in the RecHits-HSPC
   
  int const layitLimit=7;
  int const bxsitLimit=4;
  int matrixbit[bxsitLimit][layitLimit];
  for(int lay=0;lay<layitLimit;lay++){//looping layers
    for(int bxs=0;bxs<bxsitLimit;bxs++){//looping bx
      matrixbit[bxs][lay]=0;
    }
  }

  
  for(recIt=rpcHits->begin();recIt!=rpcHits->end();++recIt){
    RPCDetId RPCId((*recIt).rpcId());
    std::cout<<"RPCId"<<RPCId<<std::endl;
    
    int BX = (*recIt).BunchX();
    int layer=0;
    int mySt = RPCId.station();
    
    if(RPCId.region()==0){
      int myLa = RPCId.layer();
      if(mySt==1&&myLa==1){layer=1;}
      if(mySt==1&&myLa==2){layer=2;}
      if(mySt==2&&myLa==1){layer=3;}
      if(mySt==2&&myLa==2){layer=4;}
      if(mySt==3){layer=5;}
      if(mySt==4){layer=6;}
      matrixbit[BX][layer]++;
    }else{
      if(mySt==1){layer=1;}
      if(mySt==2){layer=2;}
      if(mySt==3){layer=3;}
    }
  }
  
  fileMatrix<<"\n";

  std::cout<<"Printing matrix Event "<<iEvent.id().event()<<std::endl;
  fileMatrix<<"Printing matrix Event "<<iEvent.id().event()<<std::endl;
  
  for(int bxs=bxsitLimit-1;bxs>=0;bxs--){//looping bx
    for(int lay=1;lay<layitLimit;lay++){//looping layers
      fileMatrix<<matrixbit[bxs][lay];
    }
    fileMatrix<<"\n";
  }

  //When we count endcaps this should work
  //assert(counter==rpcHits->size());
 
  bool hscp = false;
  int layersHited = 0;
  std::cout<<"Cleaning array Maxbx"<<std::endl;
  for(int i=0;i<7;i++)Maxbx[i]=0;
  std::cout<<"Cleaned"<<std::endl;

  for(int lay=1;lay<7;lay++){//looping layers
    bool anyhit = false;
    int maxbx = 0;
    for(int bxs=bxsitLimit-1;bxs>=0;bxs--){//looping bx
      if(matrixbit[bxs][lay]!=0.){
	//fileMatrix<<"Increasing "<<bxs<<" "<<lay<<" "<<matrixbit[bxs][lay]<<std::endl;
	matrixHisto->Fill(bxs+(lay-1)*5,matrixbit[bxs][lay]);
	if(matrixbit[bxs][lay]!=0){
	  anyhit=true;
	  if(maxbx<bxs)maxbx=bxs;
	}
      }
    }	
    if(anyhit)layersHited++;
    std::cout<<"Writing the Max array"<<" lay = "<<lay<<" maxbx = "<<maxbx<<std::endl;
    Maxbx[lay]=maxbx;
  }

  fileMatrix<<"Number Of layers "<<layersHited<<std::endl;
  std::cout<<"Number Of layers "<<layersHited<<std::endl;
  
  for(int i=1;i<7;i++)fileMatrix<<" L"<<i<<" "<<Maxbx[i];
  fileMatrix<<"\n";

  fileMatrix<<"pendpos ";
  
  int pendpos=0;
  for(int i=1;i<6;i++){
    if(Maxbx[i]<Maxbx[i+1]){
      pendpos++;
    }
    if(Maxbx[i]>Maxbx[i+1]){
      pendpos--;
    }
    fileMatrix<<" "<<pendpos;
  }

  float average=0;
  for(int i=1;i<7;i++)average=average+(float)(Maxbx[i]);
  average=average/6;
  

    
  if(layersHited>=3&&(pendpos>0||average>=1))hscp=true;
  
  std::cout<<" pendpos= "<<pendpos<<" average= "<<average<<" boolean hscp= "<<hscp<<std::endl;
  fileMatrix<<" pendpos= "<<pendpos<<" average= "<<average<<" boolean hscp= "<<hscp<<std::endl;
  
  if(hscp) totalHSCP++;



/* This is an event example
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which 
   // is put into the Event
   std::auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
   iEvent.put(pOut);
*/

/* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
RPCHSCP::beginJob(const edm::EventSetup&){

  fileMatrix.open(fileMatrixname.c_str());
  theFile = new TFile(rootFileName.c_str(),"RECREATE");
  //matrixHisto = new TH1F("LayersandBX","Histogram 2D Layers and BX",6,0.5,6.5,4,-0.5,3.5);
  matrixHisto = new TH1F("LayersandBX","Histogram 2D Layers and BX",30,-0.5,29.5);
  totalHSCP=0;
  
  //for(int lay=0;lay<4;lay++){//looping layers
  //  for(int bxs=0;bxs<7;bxs++){//looping bx
  //    matrixbitTOTAL[bxs][lay]=0;
  //  }
  //}

}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCHSCP::endJob() {
  std::cout<<"\n TOTAL HSPCs = "<<totalHSCP<<std::endl;
  fileMatrix<<"\n TOTAL HSPCs = "<<totalHSCP<<std::endl;

  theFile->cd();
  matrixHisto->Write();
  theFile->Close();
  fileMatrix.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCHSCP);
