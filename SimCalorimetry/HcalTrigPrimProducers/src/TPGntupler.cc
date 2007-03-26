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
#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "TMath.h"

using namespace std;
using namespace edm;


TPGntupler::TPGntupler(const edm::ParameterSet& iConfig) : 
  //now do what ever initialization is needed
  file("tpg_ntuple.root", "RECREATE" ),
  tree("TPGntuple","Trigger Primitive Ntuple")
{
  //  gROOT->ProcessLine(".L TPinfo.cc+");
  //infoarray = new TClonesArray("TPinfo",4176);
  //tree.Branch("tpinfoarray","TClonesArray",&infoarray,32000,99);
  tree.Branch("run",&run_num,"run/I");
  tree.Branch("event",&event_num,"event/I");
  tree.Branch("ieta",ieta,"ieta[4176]/I");
  tree.Branch("iphi",iphi,"iphi[4176]/I");
  tree.Branch("tpg_energy",tpg_energy,"tpg_energy[4176]/F");
  tree.Branch("hit_energy",hit_energy,"hit_energy[4176]/F");
  tree.Branch("tpg_uncompressed",tpg_uncompressed,"tpg_uncompressed[4176]/F");
  tree.Branch("tpg_index",index,"index[4176]/I");
  //transcoder_ = transcoder;
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

  edm::ESHandle<CaloTPGTranscoder> _transcoder;
  iSetup.get<CaloTPGRecord>().get(_transcoder);
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
  double eta1, eta2;
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
      PCaloHit cell(j->id(),j->energyEM(),j->energyHad(),j->time(),j->geantTrackId());
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

  int ntower = 0;
  double eta;
  for(IdtoEnergy::const_iterator j = TP_towers.begin(); j != TP_towers.end(); ++j)
    {
      ieta[ntower] = j->first.ieta();
      iphi[ntower] = j->first.iphi();
      tpg_energy[ntower] = j->second;
      hit_energy[ntower] = (Hit_towers.find(j->first))->second;
      theTrigTowerGeometry.towerEtaBounds(ieta[ntower],eta1,eta2);
      eta = (eta1+eta2)/2;
      //      cout << "Eta value " << eta << " ieta value " << ieta[ntower] << "\n";
      tpg_uncompressed[ntower] = float(_transcoder->hcaletValue(j->first.ietaAbs(),int(j->second)));
      index[ntower] = 100*ieta[ntower]+iphi[ntower];
      //cout << "Hit energy = " << hit_energy[ntower] << " Uncommpressed = " << tpg_uncompressed[ntower] << "\n";
      ++ntower;
    }
  tree.Fill();
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
