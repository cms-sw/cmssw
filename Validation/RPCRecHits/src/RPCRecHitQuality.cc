  /*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/01/12 12:34:34 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RPCRecHitQuality.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "Histograms.h"



#include "TFile.h"

//#include <iostream>
//#include <map>



using namespace std;
using namespace edm;



// Constructor

  RPCRecHitQuality::RPCRecHitQuality(const ParameterSet& pset){
   
// Get the debug parameter for verbose output
  rootFileName = pset.getUntrackedParameter<string>("rootFileName");

// the name of the simhit collection
  simHitLabel = pset.getUntrackedParameter<string>("simHitLabel", "SimG4Object");

// the name of the rechit collection
  recHitLabel = pset.getUntrackedParameter<string>("recHitLabel", "RPCRecHitProducer");

        
  cout << "--- [RPCRecHitQuality] Constructor called" << endl;

  
  // Create the root file
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
    
}


// Destructor
  RPCRecHitQuality::~RPCRecHitQuality(){
  cout << "--- [RPCRecHitQuality] Destructor called" << endl;
}



void RPCRecHitQuality::endJob() {
  theFile->Close();
}



// The real analysis
void RPCRecHitQuality::analyze(const Event & event, const EventSetup& eventSetup){
  cout << "--- [RPCRecHitQuality] Analysing Event: #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;

  theFile->cd();

  // Get the RPC Geometry
  cout << endl << "--- [RPCRecHitQuality] Reading RPC Geometry" << endl;
  ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);

  // Get the SimHit collection from the event
  cout << "--- [RPCRecHitQuality] Reading SimHits from event" << endl;
  Handle<PSimHitContainer> simHit;
  event.getByLabel("g4SimHits", "MuonRPCHits", simHit);

 
  // Map simhits per roll
  cout << "--- [RPCRecHitQuality] Arranging SimHits by roll" << endl;
  std::map<int, edm::PSimHitContainer> hitmap(const edm::PSimHitContainer& simHits);
  

  // Get the digis from the event
  //Handle<RPCDigiCollection> digis; 
  //event.getByLabel(digiLabel,digis);


  //================================================================================================

  
  // RechHit Analisys 
  //cout << "  -- RPCRecHit: begin analysis:" << endl;
   
  // Get the rechit collection from the event
  cout << "--- [RPCRecHitQuality] Reading RecHits from event" << endl;
  Handle<RPCRecHitCollection> recHit;
  event.getByLabel("rpcRecHits", recHit);

  // Map rechits per roll
  //cout << "--- [RPCRecHitQuality] Arranging recHits by roll" << endl;
  //std::map<int, edm::RPCRecHitCollection> rechitmap(const edm::RPCRecHitCollection& rechits);
 

}


