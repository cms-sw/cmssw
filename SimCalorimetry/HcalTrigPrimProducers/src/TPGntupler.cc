// -*- C++ -*-
//
// Package:    TPGntupler
// Class:      TPGntupler
// 
/**\class TPGntupler TPGntupler.cc src/TPGanalyzer/src/TPGntupler.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Adam Aurisano
//         Created:  Thur Jan 18 2007
// $Id$
//
//

#include "TPGntupler.h"

using namespace std;
using namespace edm;


TPGntupler::TPGntupler(const edm::ParameterSet& iConfig) : 
  //now do what ever initialization is needed
  file("tpg_ntuple.root", "RECREATE" ),
  tree("TPGntuple","Trigger Primitive Ntuple")
{
  tree.Branch("run",&run_num,"run/I");
  tree.Branch("event",&event_num,"event/I");
  tree.Branch("ieta",&ieta,"ieta/I");
  tree.Branch("iphi",&iphi,"iphi/I");
  tree.Branch("tpg_energy",&tpg_energy,"tpg_energy/F");
  tree.Branch("hit_energy",&hit_energy,"hit_energy/F");
}



TPGntupler::~TPGntupler()
{
  file.cd();
  tree.Write();
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TPGntupler::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  iSetup.get<HcalDbRecord>().get(conditions);
  // get the correct geometry
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<IdealGeometryRecord>().get(geometry);
  vector<DetId> hbCells =  geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  vector<DetId> heCells =  geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  vector<DetId> hfCells =  geometry->getValidDetIds(DetId::Hcal, HcalForward);


  Handle<HcalTrigPrimDigiCollection> tpg;
  iEvent.getByLabel("hcalTriggerPrimitiveDigis", tpg);  

  Handle<CrossingFrame> cf;
  iEvent.getByType(cf);

  run_num = iEvent.id().run();
  event_num = iEvent.id().event();


  auto_ptr<MixCollection<PCaloHit> > pcalo(new MixCollection<PCaloHit>(cf.product(),"HcalHits"));


  double hit_e;
  double tpg_e;
  double temp_e;
  //double hit_energy, tpg_energy;
  std::vector<HcalTrigTowerDetId> towerids;
  std::vector<HcalDetId> cellids;
  hit_map.clear();
  Hit_cells.clear();
  Hit_towers.clear();
  TP_towers.clear();

  assert(hit_map.size() == 0);
  //Construct map of hit cells vs. trigger tower id 
  for(MixCollection<PCaloHit>::MixItr j = pcalo->begin(); j != pcalo->end(); ++j)
    {
      PCaloHit cell(j->id(),j->energy(),j->time(),j->geantTrackId());
      HcalDetId hcalid = HcalDetId(j->id());
      //fill energies vs. hcal ids
      Hit_cells.insert(Cell_Map::value_type(hcalid,j->energy()));
      towerids = theTrigTowerGeometry.towerIds(hcalid);
      assert(towerids.size() == 2 || towerids.size() == 1);
      //fill map with cells vs tower ids
      for(unsigned int n = 0; n < towerids.size(); n++)
	{
	  hit_map.insert(IdtoHit::value_type(towerids[n],cell));
	}
    }

  for(HcalTrigPrimDigiCollection::const_iterator j = tpg->begin(); j != tpg->end(); ++j )
    {
      temp_e = 0.0;
      hit_e = 0.0;
      tpg_e = 0.0;
      
      for(IdtoHit::iterator i = hit_map.lower_bound(j->id()); i != hit_map.upper_bound(j->id()); ++i)
	{
	  towerids = theTrigTowerGeometry.towerIds((i->second).id());
	  assert(towerids.size() == 2 || towerids.size() == 1);
	  HcalDetId id = HcalDetId((i->second).id());
	  if (id.subdet() == 1)
	    {
	      temp_e += 117*((i->second).energy()/double(towerids.size()));
	    }
	  else if (id.subdet() == 2)
	    {
	      temp_e += 178*((i->second).energy()/double(towerids.size()));
	    }
	  else 
	    {
	      temp_e += 2.84*((i->second).energy()/double(towerids.size()));
	    }
	}
      hit_e = temp_e;  //hit_e is sum of all hit energy for cells in tower
      //Fill Hit_towers
      Hit_towers.insert(IdtoEnergy::value_type(j->id(),hit_e));

      tpg_e = double(j->SOI_compressedEt());
      TP_towers.insert(IdtoEnergy::value_type(j->id(),tpg_e));
    }

  //Now Hit_towers, Digi_towers, TP_towers, Hit_cells, and Digi_cells are filled
  //Next, fill histgrams using them
  //Now go through all cells and find digi and hit energy associated with them
  //Fill digi_v_hit_HE,HB,HF
  //for(vector<DetId>::const_iterator j = hbCells.begin(); j != hbCells.end(); ++j)
  // {
  //   HcalDetId id = HcalDetId(*j);
  //   towerids = theTrigTowerGeometry.towerIds(id);
  //   Cell_Map::const_iterator hit = Hit_cells.find(id);
  //  double hite;
  //  if (hit != Hit_cells.end()) 
  //{
  //  hite = (hit->second)*117;
  //}
  //  else 
  //{
  //  hite = 0;
  //}
  //}
  //for(vector<DetId>::const_iterator j = heCells.begin(); j != heCells.end(); ++j)
  //{
  //  HcalDetId id = HcalDetId(*j);
  //  towerids = theTrigTowerGeometry.towerIds(id);
  //  Cell_Map::const_iterator hit = Hit_cells.find(id);
  //  double hite;
  //  if (hit != Hit_cells.end()) 
  //{
  //  hite = (hit->second)*178;
  //}
  //  else 
  //{
  //  hite = 0;
  //}
  //}
  //for(vector<DetId>::const_iterator j = hfCells.begin(); j != hfCells.end(); ++j)
  //{
  //  HcalDetId id = HcalDetId(*j);
  //  towerids = theTrigTowerGeometry.towerIds(id);
  //  Cell_Map::const_iterator hit = Hit_cells.find(id);
  //  double hite;
  //  if (hit != Hit_cells.end()) 
  //{
  //  hite = hit->second;
  //}
  //  else 
  //{
  //  hite = 0;
  //}
  //}

  for(IdtoEnergy::const_iterator j = TP_towers.begin(); j != TP_towers.end(); ++j)
    {
      ieta = j->first.ieta();
      iphi = j->first.iphi();
      tpg_energy = j->second;
      hit_energy = (Hit_towers.find(j->first))->second;
      //cout << "ieta=" << ieta << " iphi=" << iphi << " tpg_energy=" << tpg_energy << " hit_energy=" << hit_energy << "\n";
      tree.Fill();
    }
}




// ------------ method called once each job just before starting event loop  ------------
void 
TPGntupler::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TPGntupler::endJob() 
{
}
