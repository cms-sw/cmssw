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
    
    xyview = dbe_->book2D("X Vs Y View","X Vs Y View",1000, -700., 700., 1000, -700., 700.);
    rzview = dbe_->book2D("R Vs Z View","X Vs Y View",1000, -1100., 1100.,1000,0., 700.);
    Res  = dbe_->book1D("Digi SimHit difference", "Digi SimHit Difference", 300, -8, 8);
    ResWmin2 = dbe_->book1D("W-2 Residuals", "Residuals for Wheel -2", 300, -8, 8);
    ResWmin1 = dbe_->book1D("W-1 Residuals", "Residuals for Wheel -1", 300, -8, 8);
    ResWzer0 = dbe_->book1D("W 0 Residuals", "Residuals for Wheel 0", 300, -8, 8);
    ResWplu1 = dbe_->book1D("W+1 Residuals", "Residuals for Wheel +1", 300, -8, 8);
    ResWplu2 = dbe_->book1D("W+2 Residuals", "Residuals for Wheel +2", 300, -8, 8);

    BxDist = dbe_->book1D("Bunch Crossing", "Bunch Crossing", 20, -9.5, 9.5);
    StripProf = dbe_->book1D("Strip Profile", "Strip Profile", 100, 0, 100);
  }
}

RPCDigiValid::~RPCDigiValid(){}

void RPCDigiValid::beginJob(const EventSetup& c){}

void RPCDigiValid::endJob() {
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void RPCDigiValid::analyze(const Event& event, const EventSetup& eventSetup){

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
      StripProf->Fill(digiIt->strip());
      BxDist->Fill(digiIt->bx());
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

