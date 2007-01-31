/* 
 *  See header file for a description of this class.
 *
 *  $Date: 2007/01/30 14:06:06 $
 *  $Revision: 1.5 $
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



using namespace std;
using namespace edm;


 
  // Constructor
  RPCRecHitQuality::RPCRecHitQuality(const ParameterSet& pset){
  cout << "--- [RPCRecHitQuality] Constructor called" << endl;
  
  // Get the debug parameter for verbose output
  rootFileName = pset.getUntrackedParameter<string>("rootFileName");

  // Set the name of the simhit collection
  simHitLabel = pset.getUntrackedParameter<string>("simHitLabel", "SimG4Object");

  // Set the name of the rechit collection
  recHitLabel = pset.getUntrackedParameter<string>("recHitLabel", "RPCRecHitProducer");

  // Create the root file
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
   
}


// Destructor
  RPCRecHitQuality::~RPCRecHitQuality(){
  cout << "--- [RPCRecHitQuality] Destructor called" << endl;
}



void RPCRecHitQuality::endJob() {
 
  

// Write the histos to file
  theFile->cd();
  
  Residuals->GetXaxis()->SetTitle("Distance (cm)");
  //Pulls->GetXaxis()->SetTitle("residual/error");
 
  // Write histos to file
  Rechisto->Write();
  Residuals->Write();
  Simhisto->Write(); 
  Pulls->Write();
  theFile->Close();
  
}




  
  // Access to simhits and rechits
  void RPCRecHitQuality::analyze(const Event & event, const EventSetup& eventSetup){
  cout << endl <<"--- [RPCRecHitQuality] Analysing Event: #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;

    // Get the RPC Geometry
  ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);

  // Get the SimHit collection from the event
  Handle<PSimHitContainer> simHit;
  event.getByLabel("g4SimHits", "MuonRPCHits", simHit);

  // Map simhits
  std::map<double, int> mapsim;
  std::map<int, double> nmapsim;
  
 
  //================================================================================================
  
  
  // Get the rechit collection from the event
  Handle<RPCRecHitCollection> recHit;
  event.getByLabel("rpcRecHits", recHit);

  // Map rechits, errors and residuals
  std::map<double, int> maprec;
  std::map<int, double> nmaprec;
  std::map<double, double> nmaperr;
  std::map<int, double> nmapres;




  // Loop on rechits
  RPCRecHitCollection::const_iterator recIt;
  int nrec = 0;
  for (recIt = recHit->begin(); recIt != recHit->end(); recIt++) {
    // Find chamber with rechits in RPC 
  RPCDetId Rid = (RPCDetId)(*recIt).rpcId();
  const RPCRoll* roll = dynamic_cast<const RPCRoll* >( rpcGeom->roll(Rid));
  if((roll->isForward())) return;
  nrec = nrec + 1;
  LocalPoint rhitlocal = (*recIt).localPosition();
  LocalError locerr = (*recIt).localPositionError(); 
  double rhitlocalx = rhitlocal.x();
  double rhiterrx =locerr.xx();
  //cout << "[RecHit] Pos: " << rhitlocal << " Err: " << rhiterrx << endl;
  Rechisto->Fill(rhitlocalx);
  maprec[rhitlocalx] = nrec;
  nmaperr[rhitlocalx] = rhiterrx;
  }
  cout << "  --> Found " << nrec << " rechit in event " << event.id().event() << endl;
  int i = 0;
  for (map<double, int>::iterator iter = maprec.begin(); iter != maprec.end(); iter++) {
       i = i + 1;
       nmaprec[i] = (*iter).first;
  }
 



  // Loop on simhits
  PSimHitContainer::const_iterator simIt;
  int nsim = 0;
  for (simIt = simHit->begin(); simIt != simHit->end(); simIt++) {
  nsim = nsim + 1;
  LocalPoint shitlocal = (*simIt).localPosition();
  double shitlocalx = shitlocal.x();
  //cout << " [SimHit] Pos: " << shitlocal << endl;
  Simhisto->Fill(shitlocalx);
  mapsim[shitlocalx] = nsim;
  }
  cout << "  --> Found " << nsim <<" simhits in event " << event.id().event() << endl;
  i = 0;
  for (map<double, int>::iterator iter = mapsim.begin(); iter != mapsim.end(); iter++) {
    i = i + 1;
    nmapsim[i] = (*iter).first;
  }


  // compute residuals 
  if (nrec != 0) cout << "-----------Residuals---------" << endl;
  double res;
  if (nsim == nrec) {
    for (int r=0; r<nsim; r++) {
      res = nmapsim[r+1] - nmaprec[r+1];
      cout << r+1 << " " << res << endl;
      nmapres[r+1] = res;
      Residuals->Fill(res);
      }
  }

  
  // compute Pulls 
  if (nrec != 0) cout << "-----------Pulls-------------" << endl;
  double pull;
  if (nsim == nrec) {
    for (int r=0; r<nsim; r++) {
      pull = nmapres[r+1] / nmaperr[nmaprec[r+1]];
      cout << r+1 << " " << pull << endl;
      Pulls->Fill(pull);
      }
  }




}

 
