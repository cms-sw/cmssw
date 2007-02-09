 /* 
 *  See header file for a description of this class.
 *
 *  $Date: 2007/01/31 10:23:42 $
 *  $Revision: 1.6 $
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
#include "TFolder.h"


using namespace std;
using namespace edm;


 
  
RPCRecHitQuality::RPCRecHitQuality(const ParameterSet& pset){
  cout << "--- [RPCRecHitQuality] Constructor called" << endl;
  rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  simHitLabel = pset.getUntrackedParameter<string>("simHitLabel", "SimG4Object");
  recHitLabel = pset.getUntrackedParameter<string>("recHitLabel", "RPCRecHitProducer");
  
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  // Tree Structure Definition
  char *sectname = new char[15];
  char *wheelname = new char[15];
  char *staname = new char[15];
  char *occ = new char[15];
  char *clu = new char[15];
  int wn, sn, st, prog;
  prog = 0;
  for (int i = 0; i < 5; i++) { 
    wn = i - 2;
    sprintf(wheelname,"Wheel %d",wn);	
    whe[i] = new TFolder (wheelname, wheelname);
    
    for (int j = 0; j < 12; j++) {
      sn = j + 1;
      sprintf(sectname,"Sector %d",sn);
      sec[i][j] = whe[i]->AddFolder(sectname, sectname);
      
      for (int l = 0; l < 4; l++) {
	st = l + 1;
	sprintf(staname,"Station %d",st);
	sta[i][j][l] = sec[i][j]->AddFolder(staname, staname);
	prog = prog + 1;
	if (st == 1 || st == 2) {
	  lay[i][j][l][0] = sta[i][j][l]->AddFolder("IN", "IN");
	  lay[i][j][l][1] = sta[i][j][l]->AddFolder("OUT", "OUT");
	  sprintf(occ,"Occupancy %d",prog);
	  sprintf(clu,"Cluster Size %d",prog);
	  ochamb[i][j][l][1] = new TH1F (occ, "Occupancy", 90, 0 ,90);
	  clchamb[i][j][l][1] = new TH1F (clu, "Cluster Size", 10, 0 ,10);

	  prog = prog + 1;
	  sprintf(occ,"Occupancy %d",prog);
	  sprintf(clu,"Cluster Size %d",prog);
	  ochamb[i][j][l][2] = new TH1F (occ, "Occupancy", 90, 0 ,90);
	  clchamb[i][j][l][2] = new TH1F (clu, "Cluster Size", 10, 0 ,10);
	}
	else {
	  sprintf(occ, "Occupancy %d", prog); 
	  sprintf(clu,"Cluster Size %d",prog);
	  ochamb[i][j][l][0] = new TH1F (occ, "Occupancy", 90, 0 ,90);
	  clchamb[i][j][l][0] = new TH1F (clu, "Cluster Size", 10, 0 ,10);
	}
      }
    }
  }
}

  
// Destructor
RPCRecHitQuality::~RPCRecHitQuality(){
  cout << "--- [RPCRecHitQuality] Destructor called" << endl;
}



void RPCRecHitQuality::endJob() {
  
  theFile->cd();
  
   

  Res->GetXaxis()->SetTitle("Distance (cm)");
  // Pulls->GetXaxis()->SetTitle("residual/error");



  for (int i = 0; i < 5; i++) { 
    for (int j = 0; j < 12; j++) {
      for (int l = 0; l < 4; l++) {
	if (l == 0 || l == 1) {
	  lay[i][j][l][0]->Add(ochamb[i][j][l][1]);
	  lay[i][j][l][1]->Add(ochamb[i][j][l][2]);
	  lay[i][j][l][0]->Add(clchamb[i][j][l][1]);
	  lay[i][j][l][1]->Add(clchamb[i][j][l][2]);
	}
	else {
	  sta[i][j][l]->Add(ochamb[i][j][l][0]);
	  sta[i][j][l]->Add(clchamb[i][j][l][0]);
	}
      }
    }  
    whe[i]->Write();
  } 
  
  
  fres->Add(ResWmin2);
  fres->Add(ResWmin1);
  fres->Add(ResWzer0);
  fres->Add(ResWplu1);
  fres->Add(ResWplu2);  
  
  fres->Write();
  focc->Write();

  Rechisto->Write();
  Res->Write();
  Simhisto->Write(); 
  Pulls->Write();
  ClSize->Write();
  Occupancy->Write(); 
  
  theFile->Close();
  
}



void RPCRecHitQuality::analyze(const Event & event, const EventSetup& eventSetup){
  cout << endl <<"--- [RPCRecHitQuality] Analysing Event: #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;
  
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
  std::map<int, double> nsimWmin2;
  std::map<int, double> nsimWmin1;
  std::map<int, double> nsimWzer0;
  std::map<int, double> nsimWplu1;
  std::map<int, double> nsimWplu2;

  Handle<RPCRecHitCollection> recHit;
  event.getByLabel("rpcRecHits", recHit);
  std::map<double, int> maprec;
  std::map<int, double> nmaprec;
  std::map<double, double> nmaperr;
  std::map<int, double> nmapres;
  
  std::map<double, int> recWmin2;
  std::map<double, int> recWmin1;
  std::map<double, int> recWzer0;
  std::map<double, int> recWplu1;
  std::map<double, int> recWplu2;
  std::map<int, double> nrecWmin2;
  std::map<int, double> nrecWmin1;
  std::map<int, double> nrecWzer0;
  std::map<int, double> nrecWplu1;
  std::map<int, double> nrecWplu2;
  std::map<double, double> errWmin2;
  std::map<double, double> errWmin1;
  std::map<double, double> errWzer0;
  std::map<double, double> errWplu1;
  std::map<double, double> errWplu2;
  
  


  
  // Loop on rechits
  RPCRecHitCollection::const_iterator recIt;
  int nrec = 0; 
  int nrecmin2 = 0;
  int nrecmin1 = 0;
  int nreczer0 = 0;
  int nrecplu1 = 0;
  int nrecplu2 = 0;
  
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
    double rhiterrx =locerr.xx();
    int wheel = roll->id().ring();
    int sector = roll->id().sector(); 
    int station = roll->id().station();
    int k = roll->id().layer();
    //    int s = roll->id().subsector();

    int i = wheel + 2;
    int j = sector - 1;
    int l = station - 1;
    

    // cluster size
    ClSize->Fill(clsize); //Global Cluster Size
    if (l == 0 || l == 1) {
      clchamb[i][j][l][k]->Fill(clsize);
    } else {
      clchamb[i][j][l][0]->Fill(clsize);
    }
    
    
    // occupancy
    for (int occ = 0; occ < clsize; occ++) {
      int occup = fstrip + occ;
      Occupancy->Fill(occup);
      if (l == 0 || l == 1) {
	ochamb[i][j][l][k]->Fill(occup);
      } else {
	ochamb[i][j][l][0]->Fill(occup);
      }
    }      
    

    Rechisto->Fill(rhitlocalx);
    maprec[rhitlocalx] = nrec;
    nmaperr[rhitlocalx] = rhiterrx;
    
       
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
  cout << " --> Found " << nrec << " rechit in event " << event.id().event() << endl;
   
  // Global rechit mapping
  int i = 0;
  for (map<double, int>::iterator iter = maprec.begin(); iter != maprec.end(); iter++) {
    i = i + 1;
    nmaprec[i] = (*iter).first;
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



  
  // Loop on simhits
  PSimHitContainer::const_iterator simIt;
  int nsim = 0;
  int nsimmin2 = 0;
  int nsimmin1 = 0;
  int nsimzer0 = 0;
  int nsimplu1 = 0;
  int nsimplu2 = 0;

  for (simIt = simHit->begin(); simIt != simHit->end(); simIt++) {
    int ptype = (*simIt).particleType();
    RPCDetId Rsid = (RPCDetId)(*simIt).detUnitId();
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >( rpcGeom->roll(Rsid));
    int Swheel = roll->id().ring();
    
        
    // selection of muon hits 
    if (ptype == 13) {
      nsim = nsim + 1;
      LocalPoint shitlocal = (*simIt).localPosition();
      double shitlocalx = shitlocal.x();
      Simhisto->Fill(shitlocalx);
      mapsim[shitlocalx] = nsim;
      
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
  cout << " --> Found " << nsim <<" simhits in event " << event.id().event() << endl;

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
  

  // Compute residuals 
  double res;
  double resmin2;
  double resmin1;
  double reszer0;
  double resplu1;
  double resplu2;
  if (nsim == nrec) {
    for (int r=0; r<nsim; r++) {
      res = nmapsim[r+1] - nmaprec[r+1];
      nmapres[r+1] = res;
      Res->Fill(res);
    }
  }
  if (nsimmin2 == nrecmin2) {
    for (int r=0; r<nsimmin2; r++) {
      resmin2 = nsimWmin2[r+1] - nrecWmin2[r+1];
      // nresWmin2[r+1] = resmin2;
      ResWmin2->Fill(resmin2);
    }
  }
  if (nsimmin1 == nrecmin1) {
    for (int r=0; r<nsimmin1; r++) {
      resmin1 = nsimWmin1[r+1] - nrecWmin1[r+1];
      // nresWmin2[r+1] = resmin1;
      ResWmin1->Fill(resmin1);
    }
  }
  if (nsimzer0 == nreczer0) {
    for (int r=0; r<nsimzer0; r++) {
      reszer0 = nsimWzer0[r+1] - nrecWzer0[r+1];
      // nresWmin2[r+1] = resmin2;
      ResWzer0->Fill(reszer0);
    }
  }
  if (nsimplu1 == nrecplu1) {
    for (int r=0; r<nsimplu1; r++) {
      resplu1 = nsimWplu1[r+1] - nrecWplu1[r+1];
      // nresWplu1[r+1] = resplu1;
      ResWplu1->Fill(resplu1);
    }
  }
  if (nsimplu2 == nrecplu2) {
    for (int r=0; r<nsimplu2; r++) {
      resplu2 = nsimWplu2[r+1] - nrecWplu2[r+1];
      // nresWplu2[r+1] = resplu2;
      ResWplu2->Fill(resplu2);
    }
  }
  
  
  // compute Pulls 
  double pull;
  if (nsim == nrec) {
    for (int r=0; r<nsim; r++) {
      pull = nmapres[r+1] / nmaperr[nmaprec[r+1]];
      //ut << r+1 << " " << pull << endl;
      Pulls->Fill(pull);
    }
  }
  
  
  
  
}


