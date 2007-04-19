/* 
 *  See header file for a description of this class.
 *
 *  $Date: 2007/02/28 15:35:49 $
 *  $Revision: 1.8 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RPCDigiQuality.h"
#include "Histograms.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TFolder.h"

using namespace std;
using namespace edm;
   
RPCDigiQuality::RPCDigiQuality(const ParameterSet& pset){
  cout << "--- [RPCDigiQuality] Constructor called" << endl;
  rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  digiLabel = pset.getUntrackedParameter<string>("RPCdigiLabel", "muonRPCDigis");
  
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

}

  
// Destructor
RPCDigiQuality::~RPCDigiQuality(){
  cout << "--- [RPCDigiQuality] Destructor called" << endl;
}



void RPCDigiQuality::endJob() {

    
  theFile->cd();
  
  Res->GetXaxis()->SetTitle("x distance (cm)");
  xyview->GetXaxis()->SetTitle("X coordinate (cm)");
  xyview->GetYaxis()->SetTitle("Y coordinate (cm)");
  rzview->GetXaxis()->SetTitle("Z coordinate (cm)");
  rzview->GetYaxis()->SetTitle("R  (cm)");

  fres->Add(ResWmin2);
  fres->Add(ResWmin1);
  fres->Add(ResWzer0);
  fres->Add(ResWplu1);
  fres->Add(ResWplu2);  

  fres->Write();

  Res->Write();
  xyview->Write();
  rzview->Write();

  theFile->Close();
  
}



void RPCDigiQuality::analyze(const Event & event, const EventSetup& eventSetup){
  cout << endl <<"--- [RPCDigiQuality] Analysing Event: #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;
  
  // Get the RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);
  
  edm::Handle<PSimHitContainer> simHit;
  event.getByLabel("g4SimHits", "MuonRPCHits", simHit);
  

  edm::Handle<RPCDigiCollection> rpcDigis;
  event.getByLabel(digiLabel, rpcDigis);

  // Loop on simhits
  PSimHitContainer::const_iterator simIt;

  //loop over Simhit
  std::map<RPCDetId, std::vector<double> > allsims;

  for (simIt = simHit->begin(); simIt != simHit->end(); simIt++) {
    RPCDetId Rsid = (RPCDetId)(*simIt).detUnitId();
    const RPCRoll* soll = dynamic_cast<const RPCRoll* >( rpcGeom->roll(Rsid));
    int ptype = simIt->particleType();

    std::cout <<"This is a Simhit with Parent "<<ptype<<std::endl;
    if (ptype == 13 || ptype == -13) {
      std::vector<double> buff;
      if (allsims.find(Rsid) != allsims.end() ){
	buff= allsims[Rsid];
      }
      buff.push_back(simIt->localPosition().x());
      allsims[Rsid]=buff;
    }
    GlobalPoint p=soll->toGlobal(simIt->localPosition());
    std::cout <<"Muon Position phi="<<p.phi()
	    <<" R="<<p.perp()
	      <<" z="<<p.z()<<std::endl;
    xyview->Fill(p.x(),p.y());
    rzview->Fill(p.z(),p.perp());

  }
  //loop over Digis
  RPCDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=rpcDigis->begin(); detUnitIt!=rpcDigis->end();++detUnitIt){
    const RPCDetId Rsid = (*detUnitIt).first;
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >( rpcGeom->roll(Rsid));
    const RPCDigiCollection::Range& range = (*detUnitIt).second;
    std::vector<double> sims;
    if (allsims.find(Rsid) != allsims.end() ){
      sims = allsims[Rsid];
    }
    int ndigi=0;
    for (RPCDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; ++digiIt){
      ndigi++;
    }
    

    std::cout<<" Number of Digi "<<ndigi<<" for "<<Rsid<<std::endl;

    if (sims.size() == 1 &&  ndigi == 1){
      double dis = roll->centreOfStrip(range.first->strip()).x()-sims[0];
      Res->Fill(dis);   
      
      if (Rsid.region() == 0 ){
	if (Rsid.ring() == -2)
	  ResWmin2->Fill(dis);
	else if (Rsid.ring() == -1)
	  ResWmin1->Fill(dis);
	else if (Rsid.ring() == 0)
	  ResWzer0->Fill(dis);
	else if (Rsid.ring() == 1)
	  ResWplu1->Fill(dis);
	else if (Rsid.ring() == 2)
	  ResWplu2->Fill(dis);
      }
    }
  }
}
