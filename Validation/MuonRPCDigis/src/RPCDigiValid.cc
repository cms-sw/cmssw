#include "Validation/MuonRPCDigis/interface/RPCDigiValid.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

RPCDigiValid::RPCDigiValid(const ParameterSet& ps):dbe_(0){
    
  digiLabel = ps.getUntrackedParameter<std::string>("digiLabel");
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "rpcDigiValidPlots.root");
  dbe_ = Service<DQMStore>().operator->();
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("RPCDigisV/RPCDigis");
    
    xyview = dbe_->book2D("X_Vs_Y_View","X_Vs_Y_View",1000, -760., 760., 1000, -760., 760.);
    xyvWmin2 = dbe_->book2D("Wmin2_XvsY","Wmin2_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvWmin1 = dbe_->book2D("Wmin1_XvsY","Wmin1_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvWzer0 = dbe_->book2D("Wzer0_XvsY","Wzer0_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvWplu1 = dbe_->book2D("Wplu1_XvsY","Wplu1_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvWplu2 = dbe_->book2D("Wplu2_XvsY","Wplu2_XvsY",1000, -760., 760., 1000, -760., 760.);

    xyvDplu1 = dbe_->book2D("Dplu1_XvsY","Dplu1_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvDplu2 = dbe_->book2D("Dplu2_XvsY","Dplu2_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvDplu3 = dbe_->book2D("Dplu3_XvsY","Dplu3_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvDmin1 = dbe_->book2D("Dmin1_XvsY","Dmin1_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvDmin2 = dbe_->book2D("Dmin2_XvsY","Dmin2_XvsY",1000, -760., 760., 1000, -760., 760.);
    xyvDmin3 = dbe_->book2D("Dmin3_XvsY","Dmin3_XvsY",1000, -760., 760., 1000, -760., 760.);

    rzview = dbe_->book2D("R_Vs_Z_View","R_Vs_Z_View",1000, -1100., 1100.,1000,0., 800.);
    Res  = dbe_->book1D("Digi_SimHit_difference", "Digi_SimHit_difference", 300, -8, 8);
    ResWmin2 = dbe_->book1D("W_Min2_Residuals", "W_Min2_Residuals", 300, -8, 8);
    ResWmin1 = dbe_->book1D("W_Min1_Residuals", "W_Min1_Residuals", 300, -8, 8);
    ResWzer0 = dbe_->book1D("W_Zer0_Residuals", "W_Zer0_Residuals", 300, -8, 8);
    ResWplu1 = dbe_->book1D("W_Plu1_Residuals", "W_Plu1_Residuals", 300, -8, 8);
    ResWplu2 = dbe_->book1D("W_Plu2_Residuals", "W_Plu2_Residuals", 300, -8, 8);

    BxDist = dbe_->book1D("Bunch_Crossing", "Bunch_Crossing", 20, -10., 10.);
    StripProf = dbe_->book1D("Strip_Profile", "Strip_Profile", 100, 0, 100);

    ResDmin1 = dbe_->book1D("Disk_Min1_Residuals", "Disk_Min1_Residuals", 300, -8, 8);
    ResDmin2 = dbe_->book1D("Disk_Min2_Residuals", "Disk_Min2_Residuals", 300, -8, 8);
    ResDmin3 = dbe_->book1D("Disk_Min3_Residuals", "Disk_Min3_Residuals", 300, -8, 8);
    ResDplu1 = dbe_->book1D("Disk_Plu1_Residuals", "Disk_Plu1_Residuals", 300, -8, 8);
    ResDplu2 = dbe_->book1D("Disk_Plu2_Residuals", "Disk_Plu2_Residuals", 300, -8, 8);
    ResDplu3 = dbe_->book1D("Disk_Plu3_Residuals", "Disk_Plu3_Residuals", 300, -8, 8);   

  }
}

RPCDigiValid::~RPCDigiValid(){}

void RPCDigiValid::beginJob(){}

void RPCDigiValid::endJob() {
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void RPCDigiValid::analyze(const Event& event, const EventSetup& eventSetup){

  //  cout << endl <<"--- [RPCDigiQuality] Analysing Event: #Run: " << event.id().run()
  //       << " #Event: " << event.id().event() << endl;
  
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

    //    std::cout <<"This is a Simhit with Parent "<<ptype<<std::endl;
    if (ptype == 13 || ptype == -13) {
      std::vector<double> buff;
      if (allsims.find(Rsid) != allsims.end() ){
	buff= allsims[Rsid];
      }
      buff.push_back(simIt->localPosition().x());
      allsims[Rsid]=buff;
    }
    GlobalPoint p=soll->toGlobal(simIt->localPosition());
    /*
    std::cout <<"Muon Position phi="<<p.phi()
	    <<" R="<<p.perp()
	      <<" z="<<p.z()<<std::endl;
    */

    double sim_x = p.x();
    double sim_y = p.y();

    xyview->Fill(sim_x, sim_y);

      if (Rsid.region() == 0 ){
	if (Rsid.ring() == -2)
	  xyvWmin2->Fill(sim_x, sim_y);
	else if (Rsid.ring() == -1)
	  xyvWmin1->Fill(sim_x, sim_y);
	else if (Rsid.ring() == 0)
	  xyvWzer0->Fill(sim_x, sim_y);
	else if (Rsid.ring() == 1)
	  xyvWplu1->Fill(sim_x, sim_y);
	else if (Rsid.ring() == 2)
	  xyvWplu2->Fill(sim_x, sim_y);
      }
     else if (Rsid.region() == (+1)){
        if (Rsid.station() == 1)
          xyvDplu1->Fill(sim_x, sim_y);
        else if (Rsid.station() == 2)
          xyvDplu2->Fill(sim_x, sim_y);
        else if (Rsid.station() == 3)
          xyvDplu3->Fill(sim_x, sim_y);
      }
     else if (Rsid.region() == (-1)){
        if (Rsid.station() == 1)
          xyvDmin1->Fill(sim_x, sim_y);
        else if (Rsid.station() == 2)
          xyvDmin2->Fill(sim_x, sim_y);
        else if (Rsid.station() == 3)
          xyvDmin3->Fill(sim_x, sim_y);
      }


//    xyview->Fill(p.x(),p.y());
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
      StripProf->Fill(digiIt->strip());
      BxDist->Fill(digiIt->bx());
      ndigi++;
    }
    

    //    std::cout<<" Number of Digi "<<ndigi<<" for "<<Rsid<<std::endl;

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
     else if (Rsid.region() == (+1)){
        if (Rsid.station() == 1)
          ResDplu1->Fill(dis);
        else if (Rsid.station() == 2)
          ResDplu2->Fill(dis);
        else if (Rsid.station() == 3)
          ResDplu3->Fill(dis);
      }
     else if (Rsid.region() == (-1)){
        if (Rsid.station() == 1)
          ResDmin1->Fill(dis);
        else if (Rsid.station() == 2)
          ResDmin2->Fill(dis);
        else if (Rsid.station() == 3)
          ResDmin3->Fill(dis);
      }
    }
  }
}
