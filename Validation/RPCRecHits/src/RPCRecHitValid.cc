 /* 
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/21 10:48:14 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "Validation/RPCRecHits/interface/RPCRecHitValid.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;
 
  
RPCRecHitValid::RPCRecHitValid(const ParameterSet& pset){
  rootFileName = pset.getUntrackedParameter<string>("rootFileName", "rpcRecHitValidPlots.root");
  //simHitLabel = pset.getUntrackedParameter<string>("g4SimHits", "MuonRPCHits");
  //recHitLabel = pset.getUntrackedParameter<string>("recHitLabel", "RPCRecHitProducer");
  dbe_ = Service<DQMStore>().operator->();
  
  if ( dbe_ ) {

    Rechisto   = dbe_->book1D("RecHits", "RPC RecHits", 300, -150, 150);
    Simhisto   = dbe_->book1D("SimHits", "Simulated Hits", 300, -150, 150);
    Pulls      = dbe_->book1D("Global Pulls", "RPC Global Pulls", 100, -4,4);
    ClSize     = dbe_->book1D("Global ClSize", "Global Cluster Size", 10, 0, 10);
    res1cl     = dbe_->book1D("Residuals CS = 1", "Residuals for ClSize = 1", 300, -8, 8);

//     dbe_->setCurrentFolder("Residuals");
//     Res  = dbe_->book1D("Global Residuals", "Global Residuals", 300, -8, 8);
//     ResWmin2 = dbe_->book1D("W-2 Residuals", "Residuals for Wheel -2", 300, -8, 8);
//     ResWmin1 = dbe_->book1D("W-1 Residuals", "Residuals for Wheel -1", 300, -8, 8);
//     ResWzer0 = dbe_->book1D("W 0 Residuals", "Residuals for Wheel 0", 300, -8, 8);
//     ResWplu1 = dbe_->book1D("W+1 Residuals", "Residuals for Wheel +1", 300, -8, 8);
//     ResWplu2 = dbe_->book1D("W+2 Residuals", "Residuals for Wheel +2", 300, -8, 8);
//     ResS1    = dbe_->book1D("Sector 1 Residuals", "Sector 1 Residuals", 300, -8, 8);
//     ResS3    = dbe_->book1D("Sector 3 Residuals", "Sector 3 Residuals", 300, -8, 8);

    dbe_->setCurrentFolder("Residuals and Occupancy");
    occRB1IN   = dbe_->book1D("RB1 IN Occupancy", "RB1 IN Occupancy", 100, 0, 100);
    occRB1OUT   = dbe_->book1D("RB1 OUT Occupancy", "RB1 OUT Occupancy", 100, 0, 100);

    //    dbe_->setCurrentFolder("Residuals");
    Res  = dbe_->book1D("Global Residuals", "Global Residuals", 300, -8, 8);
    ResWmin2 = dbe_->book1D("W-2 Residuals", "Residuals for Wheel -2", 300, -8, 8);
    ResWmin1 = dbe_->book1D("W-1 Residuals", "Residuals for Wheel -1", 300, -8, 8);
    ResWzer0 = dbe_->book1D("W 0 Residuals", "Residuals for Wheel 0", 300, -8, 8);
    ResWplu1 = dbe_->book1D("W+1 Residuals", "Residuals for Wheel +1", 300, -8, 8);
    ResWplu2 = dbe_->book1D("W+2 Residuals", "Residuals for Wheel +2", 300, -8, 8);
    ResS1    = dbe_->book1D("Sector 1 Residuals", "Sector 1 Residuals", 300, -8, 8);
    ResS3    = dbe_->book1D("Sector 3 Residuals", "Sector 3 Residuals", 300, -8, 8);




  }  
}

void RPCRecHitValid::beginJob() {}

// Destructor
RPCRecHitValid::~RPCRecHitValid(){
}

void RPCRecHitValid::endJob() {
 if ( rootFileName.size() != 0 && dbe_ ) dbe_->save(rootFileName);
}



void RPCRecHitValid::analyze(const Event & event, const EventSetup& eventSetup){
  
  // Get the RPC Geometry
  ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);
  
  Handle<PSimHitContainer> simHit;
  event.getByLabel("g4SimHits", "MuonRPCHits", simHit);
  std::map<double, int> mapsim;
  std::map<int, double> nmapsim;
  std::map<double, int> simWmin2;
  std::map<double, int> simWmin1;
  std::map<double, int> simWzer0;
  std::map<double, int> simWplu1;
  std::map<double, int> simWplu2;
  std::map<double, int> simS1;
  std::map<double, int> simS3;
  std::map<int, double> nsimWmin2;
  std::map<int, double> nsimWmin1;
  std::map<int, double> nsimWzer0;
  std::map<int, double> nsimWplu1;
  std::map<int, double> nsimWplu2;
  std::map<int, double> nsimS1;
  std::map<int, double> nsimS3;

  Handle<RPCRecHitCollection> recHit;
  event.getByLabel("rpcRecHits", recHit);
  std::map<double, int> maprec;
  std::map<int, double> nmaprec;
  std::map<double, double> nmaperr;
  std::map<int, double> nmapres;
    
  std::map<double, int> maprecCL1;
  std::map<int, double> nmaprecCL1;

  std::map<double, int> recWmin2;
  std::map<double, int> recWmin1;
  std::map<double, int> recWzer0;
  std::map<double, int> recWplu1;
  std::map<double, int> recWplu2;
  std::map<double, int> recS1;
  std::map<double, int> recS3;
  std::map<int, double> nrecWmin2;
  std::map<int, double> nrecWmin1;
  std::map<int, double> nrecWzer0;
  std::map<int, double> nrecWplu1;
  std::map<int, double> nrecWplu2;
  std::map<int, double> nrecS1;
  std::map<int, double> nrecS3;
  std::map<double, double> errWmin2;
  std::map<double, double> errWmin1;
  std::map<double, double> errWzer0;
  std::map<double, double> errWplu1;
  std::map<double, double> errWplu2;
  

  // Loop on rechits
  RPCRecHitCollection::const_iterator recIt;
  int nrec = 0; 
  int nrecCL1 = 0;
  int nrecmin2 = 0;
  int nrecmin1 = 0;
  int nreczer0 = 0;
  int nrecplu1 = 0;
  int nrecplu2 = 0;
  int nrecS1c = 0;
  int nrecS3c = 0;
  
  for (recIt = recHit->begin(); recIt != recHit->end(); recIt++) {
    RPCDetId Rid = (RPCDetId)(*recIt).rpcId();
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >( rpcGeom->roll(Rid));
    if((roll->isForward())) return;
    int clsize = (*recIt).clusterSize();
    int fstrip = (*recIt).firstClusterStrip();
    
    nrec = nrec + 1;
    LocalPoint rhitlocal = (*recIt).localPosition();
    LocalError locerr = (*recIt).localPositionError(); 
    double rhitlocalx = rhitlocal.x();
    double rhiterrx = locerr.xx();
    Rechisto->Fill(rhitlocalx);
    int wheel = roll->id().ring();
    int sector = roll->id().sector(); 
    int station = roll->id().station();
    int k = roll->id().layer();
    //    int s = roll->id().subsector();

    //-----CLSIZE = 1------------
    if (clsize == 1) {
      maprecCL1[rhitlocalx] = nrec;
      nrecCL1 = nrecCL1 + 1;
    }
    //-----------------------------

    ClSize->Fill(clsize); //Global Cluster Size
    
    
    // occupancy
    for (int occ = 0; occ < clsize; occ++) {
      int occup = fstrip + occ;
      if (station == 1 && k == 1) {
	occRB1IN->Fill(occup);
      }
      if (station == 1 && k == 2) {
	occRB1OUT->Fill(occup);
      }
    }      
    

    maprec[rhitlocalx] = nrec;
    nmaperr[rhitlocalx] = rhiterrx;
    
    //-------PHI-------------------
    if(sector == 1) {
      recS1[rhitlocalx] = nrec;
      nrecS1c = nrecS1c + 1;
    }
    if(sector == 3) {
      recS3[rhitlocalx] = nrec;
      nrecS3c = nrecS3c + 1;
    }
    //----------------------------
       
    if(wheel == -2) {
      recWmin2[rhitlocalx] = nrec;
      errWmin2[rhitlocalx] = rhiterrx;
      nrecmin2 = nrecmin2 + 1;
    }
    if(wheel == -1) {
      recWmin1[rhitlocalx] = nrec;
      errWmin1[rhitlocalx] = rhiterrx;
      nrecmin1 = nrecmin1 + 1;
    }
    if(wheel == 0) {
      recWzer0[rhitlocalx] = nrec;
      errWzer0[rhitlocalx] = rhiterrx;
      nreczer0 = nreczer0 + 1;
    }
    if(wheel == 1) {
      recWplu1[rhitlocalx] = nrec;
      errWplu1[rhitlocalx] = rhiterrx;
      nrecplu1 = nrecplu1 + 1;
    }
    if(wheel == 2) {
      recWplu2[rhitlocalx] = nrec;
      errWplu2[rhitlocalx] = rhiterrx;
      nrecplu2 = nrecplu2 + 1;
    }
  }
  //cout << " --> Found " << nrec << " rechit in event " << event.id().event() << endl;
   
  // Global rechit mapping
  int i = 0;
  for (map<double, int>::iterator iter = maprec.begin(); iter != maprec.end(); iter++) {
    i = i + 1;
    nmaprec[i] = (*iter).first;
  }
  // CL = 1 rechit mapping
  i = 0;
  for (map<double, int>::iterator iter = maprecCL1.begin(); iter != maprecCL1.end(); iter++) {
    i = i + 1;
    nmaprecCL1[i] = (*iter).first;
  }
  // Wheel -2 rechit mapping
  i = 0;
  for (map<double, int>::iterator iter = recWmin2.begin(); iter != recWmin2.end(); iter++) {
    i = i + 1;
    nrecWmin2[i] = (*iter).first;
  }  
  // Wheel -1 rechit mapping
  i = 0;
  for (map<double, int>::iterator iter = recWmin1.begin(); iter != recWmin1.end(); iter++) {
    i = i + 1;
    nrecWmin1[i] = (*iter).first;
  }
  // Wheel 0 rechit mapping
  i = 0;
  for (map<double, int>::iterator iter = recWzer0.begin(); iter != recWzer0.end(); iter++) {
    i = i + 1;
    nrecWzer0[i] = (*iter).first;
  }
  // Wheel +1 rechit mapping
  i = 0;
  for (map<double, int>::iterator iter = recWplu1.begin(); iter != recWplu1.end(); iter++) {
    i = i + 1;
    nrecWplu1[i] = (*iter).first;
  }
  // Wheel +2 rechit mapping
  i = 0;
  for (map<double, int>::iterator iter = recWplu2.begin(); iter != recWplu2.end(); iter++) {
    i = i + 1;
    nrecWplu2[i] = (*iter).first;
  }
  // Sector 1 rechit mapping
  i = 0;
  for (map<double, int>::iterator iter = recS1.begin(); iter != recS1.end(); iter++) {
    i = i + 1;
    nrecS1[i] = (*iter).first;
  }
  // Sector 3 rechit mapping
  i = 0;
  for (map<double, int>::iterator iter = recS3.begin(); iter != recS3.end(); iter++) {
    i = i + 1;
    nrecS3[i] = (*iter).first;
  }

  
  // Loop on simhits
  PSimHitContainer::const_iterator simIt;
  int nsim = 0;
  int nsimmin2 = 0;
  int nsimmin1 = 0;
  int nsimzer0 = 0;
  int nsimplu1 = 0;
  int nsimplu2 = 0;
  int nsimS1c = 0;
  int nsimS3c = 0;

  for (simIt = simHit->begin(); simIt != simHit->end(); simIt++) {
    int ptype = (*simIt).particleType();
    RPCDetId Rsid = (RPCDetId)(*simIt).detUnitId();
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >( rpcGeom->roll(Rsid));
    int Swheel = roll->id().ring();
    int Ssector = roll->id().sector();
        
    // selection of muon hits 
    if (ptype == 13 || ptype == -13) {
      nsim = nsim + 1;
      LocalPoint shitlocal = (*simIt).localPosition();
      double shitlocalx = shitlocal.x();
      Simhisto->Fill(shitlocalx);      

      mapsim[shitlocalx] = nsim;

      //----PHI------------------------
      if(Ssector == 1) {
	simS1[shitlocalx] = nsim;
	nsimS1c = nsimS1c + 1;
      }
      if(Ssector == 3) {
	simS3[shitlocalx] = nsim;
	nsimS3c = nsimS3c + 1;
      }
      //--------------------------------


      if(Swheel == -2) {
	simWmin2[shitlocalx] = nsim;
	nsimmin2 = nsimmin2 + 1;
      }
      if(Swheel == -1) {
	simWmin1[shitlocalx] = nsim;
	nsimmin1 = nsimmin1 + 1;
      }
      if(Swheel == 0) {
	simWzer0[shitlocalx] = nsim;
	nsimzer0 = nsimzer0 + 1;
      }
      if(Swheel == 1) {
	simWplu1[shitlocalx] = nsim;
	nsimplu1 = nsimplu1 + 1;
      }
      if(Swheel == 2) {
	simWplu2[shitlocalx] = nsim;
	nsimplu2 = nsimplu2 + 1;
      }
     
    }
  }
  //cout << " --> Found " << nsim <<" simhits in event " << event.id().event() << endl;

  // Global simhit mapping
  i = 0;
  for (map<double, int>::iterator iter = mapsim.begin(); iter != mapsim.end(); iter++) {
    i = i + 1;
    nmapsim[i] = (*iter).first;
  }
  // Wheel -2 simhit mapping
  i = 0;
  for (map<double, int>::iterator iter = simWmin2.begin(); iter != simWmin2.end(); iter++) {
    i = i + 1;
    nsimWmin2[i] = (*iter).first;
  }
  // Wheel -1 simhit mapping
  i = 0;
  for (map<double, int>::iterator iter = simWmin1.begin(); iter != simWmin1.end(); iter++) {
    i = i + 1;
    nsimWmin1[i] = (*iter).first;
  }
  // Wheel 0 simhit mapping
  i = 0;
  for (map<double, int>::iterator iter = simWzer0.begin(); iter != simWzer0.end(); iter++) {
    i = i + 1;
    nsimWzer0[i] = (*iter).first;
  }
  // Wheel +1 simhit mapping
  i = 0;
  for (map<double, int>::iterator iter = simWplu1.begin(); iter != simWplu1.end(); iter++) {
    i = i + 1;
    nsimWplu1[i] = (*iter).first;
  }
  // Wheel +2 simhit mapping
  i = 0;
  for (map<double, int>::iterator iter = simWplu2.begin(); iter != simWplu2.end(); iter++) {
    i = i + 1;
    nsimWplu2[i] = (*iter).first;
  }
  // Sector 1 simhit mapping
  i = 0;
  for (map<double, int>::iterator iter = simS1.begin(); iter != simS1.end(); iter++) {
    i = i + 1;
    nsimS1[i] = (*iter).first;
  }
  // Sector 3 simhit mapping
  i = 0;
  for (map<double, int>::iterator iter = simS3.begin(); iter != simS3.end(); iter++) {
    i = i + 1;
    nsimS3[i] = (*iter).first;
  }

  // Compute residuals 
  double res,resmin2,resmin1,reszer0,resplu1,resplu2,resS1,resS3;
  if (nsim == nrec) {
    for (int r=0; r<nsim; r++) {
      res = nmapsim[r+1] - nmaprec[r+1];
      nmapres[r+1] = res;
      Res->Fill(res);
    }
  }
  if (nsim == nrecCL1) {
   for (int r=0; r<nsim; r++) {
     res = nmapsim[r+1] - nmaprecCL1[r+1];
     //cout << nmapsim[r+1] << " " << nmaprecCL1[r+1] << endl;
     if (abs(res) < 3) {
       res1cl->Fill(res);
     }
   }
 }
  if (nsimmin2 == nrecmin2) {
    for (int r=0; r<nsimmin2; r++) {
      resmin2 = nsimWmin2[r+1] - nrecWmin2[r+1];
      ResWmin2->Fill(resmin2);
    }
  }
  if (nsimmin1 == nrecmin1) {
    for (int r=0; r<nsimmin1; r++) {
      resmin1 = nsimWmin1[r+1] - nrecWmin1[r+1];
      ResWmin1->Fill(resmin1);
    }
  }
  if (nsimzer0 == nreczer0) {
    for (int r=0; r<nsimzer0; r++) {
      reszer0 = nsimWzer0[r+1] - nrecWzer0[r+1];
      ResWzer0->Fill(reszer0);
    }
  }
  if (nsimplu1 == nrecplu1) {
    for (int r=0; r<nsimplu1; r++) {
      resplu1 = nsimWplu1[r+1] - nrecWplu1[r+1];
      ResWplu1->Fill(resplu1);
    }
  }
  if (nsimplu2 == nrecplu2) {
    for (int r=0; r<nsimplu2; r++) {
      resplu2 = nsimWplu2[r+1] - nrecWplu2[r+1];
      ResWplu2->Fill(resplu2);
    }
  }
  if (nsimS1c == nrecS1c) {
    for (int r=0; r<nsimS1c; r++) {
      resS1 = nsimS1[r+1] - nrecS1[r+1];
      ResS1->Fill(resS1);
    }
  }
  if (nsimS3c == nrecS3c) {
    for (int r=0; r<nsimS3c; r++) {
      resS3 = nsimS3[r+1] - nrecS3[r+1];
      ResS3->Fill(resS3);
    }
  }


  // compute Pulls 
  double pull;
  if (nsim == nrec) {
    for (int r=0; r<nsim; r++) {
      pull = nmapres[r+1] / nmaperr[nmaprec[r+1]];
      Pulls->Fill(pull);
    }
  }
}


