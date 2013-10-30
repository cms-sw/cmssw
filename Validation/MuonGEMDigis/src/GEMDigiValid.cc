#include "Validation/MuonGEMDigis/interface/GEMDigiValid.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <cmath>

//for GEMs
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

using namespace std;
using namespace edm;

/*
GEMDigiValid::GEMDigiValid(const ParameterSet& ps) :
  dbe_(0)
{

  digiLabel = ps.getUntrackedParameter<std::string> ("digiLabel");
  outputFile_ = ps.getUntrackedParameter<string> ("outputFile", "GEMDigiValidPlots.root");
  dbe_ = Service<DQMStore> ().operator->();
}
*/

GEMDigiValid::GEMDigiValid(const ParameterSet& ps) :
  dbe_(0)
{

//  Init the tokens for run data retrieval - stanislav
//  ps.getUntackedParameter<InputTag> retrieves a InputTag from the configuration. The second param is default value
//  module, instance and process labels may be passed in a single string if separated by colon ':'
//  (@see the edm::InputTag constructor documentation)
  simHitToken = consumes<PSimHitContainer>(ps.getUntrackedParameter<edm::InputTag >("simHitTag", edm::InputTag("g4SimHits:MuonGEMHits")));
  gemDigiToken    = consumes<GEMDigiCollection>(ps.getUntrackedParameter<edm::InputTag>("gemDigiTag", edm::InputTag("simMuonGEMDigis")));

  outputFile_ = ps.getUntrackedParameter<string> ("outputFile", "gemDigiValidPlots.root");
  dbe_ = Service<DQMStore> ().operator->();
}




GEMDigiValid::~GEMDigiValid()
{
}

void GEMDigiValid::beginJob()
{
}

void GEMDigiValid::endJob()
{
  if (outputFile_.size() != 0 && dbe_)
    dbe_->save(outputFile_);

}

void GEMDigiValid::analyze(const Event& event, const EventSetup& eventSetup)
{

//  countEvent++;
  edm::ESHandle<GEMGeometry> gemsGeom;
  eventSetup.get<MuonGeometryRecord> ().get(gemsGeom);

/*
  edm::Handle<PSimHitContainer> gemsSimHit;
  event.getByLabel("g4SimHits", "MuonGEMHits", gemsSimHit);
  edm::Handle<GEMDigiCollection> gemsDigis;
  event.getByLabel("simMuonGEMDigis", gemsDigis);
*/

  edm::Handle<PSimHitContainer> gemsSimHit;
  event.getByToken(simHitToken, gemsSimHit);

  edm::Handle<GEMDigiCollection> gemsDigis;
  event.getByToken(gemDigiToken, gemsDigis);


  //get the list of Ids and the number of strips:
  GEMDigiCollection::DigiRangeIterator gemUnitIt;
  for (gemUnitIt = gemsDigis->begin(); gemUnitIt != gemsDigis->end(); ++gemUnitIt)
  {
    const GEMDetId& id = (*gemUnitIt).first;
    const GEMEtaPartition* roll = gemsGeom->etaPartition(id);
    if (!roll)
      continue;

    const GEMDigiCollection::Range& range = (*gemUnitIt).second;
    for (GEMDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt)
    {
//      cout << " digi " << *digiIt << endl;
      if (digiIt->strip() < 1 || digiIt->strip() > roll->nstrips())
      {
        cout << " XXXXXXXXXXXXX Problemt with " << id << " a digi has strip# = " << digiIt->strip() << endl;
      }
for    (const auto& simGemHit: *gemsSimHit)
    {
      GEMDetId gemId(simGemHit.detUnitId());
      //        std::cout << "particle Id = " << simGemHit.particleType() << std::endl;
      /*
       if (gemId == id && abs(simGemHit.particleType()) == 13)
       {
       cout<<"entry: "<< simGemHit.entryPoint()<<endl
       <<"exit: "<< simGemHit.exitPoint()<<endl
       <<"TOF: "<< simGemHit.timeOfFlight()<<endl;
       }
       */
    }
  }// for digis in layer
}// for layers
//end


// Loop on simhits
PSimHitContainer::const_iterator simIt;
std::map<GEMDetId, std::vector<double> > allGEMsims;

//gem loop over gemsHit
for (simIt = gemsSimHit->begin(); simIt != gemsSimHit->end(); simIt++)
{
  GEMDetId Gsid = (GEMDetId)(*simIt).detUnitId();
  const GEMEtaPartition* gemoll = dynamic_cast<const GEMEtaPartition*> (gemsGeom->etaPartition(Gsid));
  int ptype = simIt->particleType();

  // what particles are in there?
  particleIDs->Fill(ptype);
  pidsSet.insert(ptype);

  GlobalPoint pGem = gemoll->toGlobal(simIt->localPosition());

  double sim_xgem = pGem.x();
  double sim_ygem = pGem.y();

  xyview_simHits->Fill(fabs(sim_xgem), fabs(sim_ygem));
//  rzview_gem->Fill(pGem.z(), pGem.perp());
  rzview_simHits->Fill(fabs(pGem.z()), fabs(pGem.perp()));

  if (fabs(ptype) == 13)
//  if (ptype == 13 || ptype == -13)
  {
    std::vector<double> buffGEM;
    if (allGEMsims.find(Gsid) != allGEMsims.end())
    {
      buffGEM = allGEMsims[Gsid];
    }
    buffGEM.push_back(simIt->localPosition().x());
    allGEMsims[Gsid] = buffGEM;
  }//end gemsSimHit
}

//loop over gem digis
GEMDigiCollection::DigiRangeIterator detGemUnitIt;
for (detGemUnitIt = gemsDigis->begin(); detGemUnitIt != gemsDigis->end(); ++detGemUnitIt)
{
  const GEMDetId Gsid = (*detGemUnitIt).first;
  const GEMEtaPartition* gemoll = dynamic_cast<const GEMEtaPartition*> (gemsGeom->etaPartition(Gsid));

  //    std:: cout << "GEM Id = " << (gemoll->id().rawId()) << std::endl;

  const GEMDigiCollection::Range& range = (*detGemUnitIt).second;
  std::vector<double> gems_sims;
  if (allGEMsims.find(Gsid) != allGEMsims.end())
  {
    gems_sims = allGEMsims[Gsid];
  }
  double nGemdigi = 0;
  for (GEMDigiCollection::const_iterator gemdigiIt = range.first; gemdigiIt != range.second; ++gemdigiIt)
  {
    gemStripProf->Fill(gemdigiIt->strip());
    gemBxDist->Fill(gemdigiIt->bx());
    nGemdigi++;
  }
  /*
   std::cout << "Gsid.region() = " << Gsid.region() << std::endl;
   std::cout << "Gsid.ring() = " << Gsid.ring() << std::endl;
   std::cout << "Gsid.station() = " << Gsid.station() << std::endl;
   std::cout << "Gsid.layer() = " << Gsid.layer() << std::endl;
   std::cout << "Gsid.chamber() = " << Gsid.chamber() << std::endl;
   std::cout << "Gsid.roll() = " << Gsid.roll() << std::endl;
   */

  if (gems_sims.size() == 0)
  {
    if (Gsid.region() == 0)
    {
      std::cout << "no gems and gem's cls in the barrel" << std::endl;
    }
    else
    noiseGemCLS->Fill(nGemdigi);
  }
  if (Gsid.region() != 0)
  {
    clsGEMs->Fill(nGemdigi);
    if (gems_sims.size() == 1 && nGemdigi == 1)
    {
      LocalPoint lcP = gemoll->centreOfStrip(range.first->strip());
      double gis = gemoll->centreOfStrip(range.first->strip()).x() - gems_sims[0];
      Res_gem->Fill(gis);
      GlobalPoint glP = gemoll->toGlobal(lcP);

      xyview_gemDigis->Fill(fabs(glP.x()), fabs(glP.y()));
      rzview_gemDigis->Fill(fabs(glP.z()), fabs(glP.perp()));
    }
  }
}
}

void GEMDigiValid::beginRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
//  countEvent = 0;

  if (dbe_)
  {
    dbe_->setCurrentFolder("GEMDigisV/GEMDigis");

    gemBxDist = dbe_->book1D("Bunch_Crossing_GEM", "Bunch_Crossing_GEM", 10, -5., 5.);
    gemStripProf = dbe_->book1D("Strip_Profile_GEM", "Strip_Profile_GEM", 390, 0, 390);
    Res_gem = dbe_->book1D("residuals_gem", "residuals_gem", 200, -1., 1.);
    xyview_simHits = dbe_->book2D("XY_simHits", "XY_simHits", 2400, 0., 240., 2400, 0., 240.);
    xyview_gemDigis = dbe_->book2D("XY_Digis", "XY_Digis", 2400, 0., 240., 2400, 0., 240.);
//    rzview_gem = dbe_->book2D("RZ_GEMView", "RZ_GEMView", 117, -585., 585., 13, 120., 250.); //(560-580)
    rzview_simHits = dbe_->book2D("RZ_simHits", "RZ_simHits", 100, 564., 574., 1100, 130., 240.);
    rzview_gemDigis = dbe_->book2D("RZ_Digis", "RZ_Digis", 100, 564., 574., 1100, 130., 240.);
    //cls histos
    noiseGemCLS = dbe_->book1D("noiseGemCLS", "noiseGemCLS", 10, 0.5, 10.5);
    clsGEMs = dbe_->book1D("clsGEMs", "clsGEMs", 10, 0.5, 10.5);
    //particle types
    particleIDs = dbe_->book1D("Particle_IDs", "Particle_IDs", 30, -15, 15);

  }//end dbe
}

void GEMDigiValid::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
 
//particle types
  cout << "Different particle ids (unordered):" << endl;
  for (std::set<int>::iterator itr = pidsSet.begin(); itr != pidsSet.end(); ++itr)
  {
    cout << (*itr) << endl;
  }

}
