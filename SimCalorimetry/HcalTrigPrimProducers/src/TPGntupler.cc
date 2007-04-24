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
// $Id: TPGntupler.cc,v 1.3 2007/03/26 09:25:29 aurisano Exp $
//
//

#include "TPGntupler.h"
#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "TMath.h"
#include <fstream>

using namespace std;
using namespace edm;

TPGntupler::TPGntupler(const edm::ParameterSet& iConfig) : 
  //now do what ever initialization is needed
  file("tpg_ntuple.root", "RECREATE" ),
  tree("TPGntuple","Trigger Primitive Ntuple")
{
  tree.Branch("run",&run_num,"run/I");
  tree.Branch("event",&event_num,"event/I");
  tree.Branch("ieta",ieta,"ieta[4176]/I");
  tree.Branch("iphi",iphi,"iphi[4176]/I");
  tree.Branch("tpg_energy",tpg_energy,"tpg_energy[4176]/F");
  //tree.Branch("hit_energy",hit_energy,"hit_energy[4176]/F");
  tree.Branch("tpg_uncompressed",tpg_uncompressed,"tpg_uncompressed[4176]/F");
  tree.Branch("tpg_index",index,"index[4176]/I");
  tree.Branch("rec_energy",rec_energy,"rec_energy[4176]/F");
}

TPGntupler::~TPGntupler()
{
  file.cd();
  tree.Write();
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

  ofstream out;
  out.open("outfile.txt");
  char semicolon = 59;

  edm::ESHandle<CaloTPGTranscoder> _transcoder;
  iSetup.get<CaloTPGRecord>().get(_transcoder);
  // get the appropriate gains, noises, & widths for this event
  //edm::ESHandle<HcalDbService> conditions;
  //iSetup.get<HcalDbRecord>().get(conditions);
  // get the correct geometry
  //edm::ESHandle<CaloGeometry> geometry;
  //iSetup.get<IdealGeometryRecord>().get(geometry);
  //vector<DetId> hbCells =  geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  //vector<DetId> heCells =  geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  //vector<DetId> hfCells =  geometry->getValidDetIds(DetId::Hcal, HcalForward);

  Handle<HcalTrigPrimDigiCollection> tpg;
  iEvent.getByType(tpg);  

  //  Handle<CrossingFrame> cf;
  //iEvent.getByType(cf);

  Handle<HBHERecHitCollection> hbhe_rec;
  iEvent.getByType(hbhe_rec);
  
  Handle<HFRecHitCollection> hf_rec;
  iEvent.getByType(hf_rec);

  run_num = iEvent.id().run();
  event_num = iEvent.id().event();

  //auto_ptr<MixCollection<PCaloHit> > pcalo(new MixCollection<PCaloHit>(cf.product(),"HcalHits"));

  //float hit_e;
  float tpg_e;
  float rec_e;
  double eta1, eta2;
  float calib;
  std::vector<HcalTrigTowerDetId> towerids;
  double eta;
  //  Hit_towers.clear();
  Rec_towers.clear();

  //Construct map of hit cells vs. trigger tower id 
  //for(MixCollection<PCaloHit>::MixItr sim_iter = pcalo->begin(); sim_iter != pcalo->end(); ++sim_iter)
  //{
  //HcalDetId hcalid = HcalDetId(sim_iter->id());
  //if (hcalid.subdet() == 1) 
  //{
  //calib = 117.0;
  //}
  //else if (hcalid.subdet() == 2) 
  //{
  //calib = 178.0;
  //}
  //else if (hcalid.subdet() == 3) 
  //{
  // calib = 2.84;
  //}
  //else
  //{
  //calib = 0.0;
  //}
  //towerids = theTrigTowerGeometry.towerIds(hcalid);
  //assert(towerids.size() == 2 || towerids.size() == 1);
  //for(unsigned int n = 0; n < towerids.size(); n++)
  //{
  //Hit_towers.insert(IdtoEnergy::value_type(towerids[n],(calib*sim_iter->energy())/towerids.size()));
  //}
  //}

  for(HBHERecHitCollection::const_iterator hbhe_iter = hbhe_rec->begin(); hbhe_iter != hbhe_rec->end(); ++hbhe_iter)
    {
      towerids = theTrigTowerGeometry.towerIds(hbhe_iter->id());
      assert(towerids.size() == 2 || towerids.size() == 1);
      for(unsigned int n = 0; n < towerids.size(); n++)
	{
	  Rec_towers.insert(IdtoEnergy::value_type(towerids[n],hbhe_iter->energy()/towerids.size()));
	}
    }

  for(HFRecHitCollection::const_iterator hf_iter = hf_rec->begin(); hf_iter != hf_rec->end(); ++hf_iter)
    {
      towerids = theTrigTowerGeometry.towerIds(hf_iter->id());
      assert(towerids.size() == 2 || towerids.size() == 1);
      for(unsigned int n = 0; n < towerids.size(); n++)
        {
          Rec_towers.insert(IdtoEnergy::value_type(towerids[n],hf_iter->energy()/towerids.size()));
        }
    }

  int ntower = 0;
  for(HcalTrigPrimDigiCollection::const_iterator tpg_iter = tpg->begin(); tpg_iter != tpg->end(); ++tpg_iter )
    {
      //hit_e = 0.0;
      tpg_e = 0.0;
      rec_e = 0.0;
      //for(IdtoEnergy::iterator i = Hit_towers.lower_bound(tpg_iter->id()); i != Hit_towers.upper_bound(tpg_iter->id()); ++i)
      //{
      //hit_e += i->second;
      ///}
      for(IdtoEnergy::iterator i = Rec_towers.lower_bound(tpg_iter->id()); i != Rec_towers.upper_bound(tpg_iter->id()); ++i)
	{
	  rec_e += i->second;
	}
      tpg_e = double(tpg_iter->SOI_compressedEt());
      ieta[ntower] = tpg_iter->id().ieta();
      iphi[ntower] = tpg_iter->id().iphi();
      tpg_energy[ntower] = tpg_e;
      //hit_energy[ntower] = hit_e;
      rec_energy[ntower] = rec_e;
      theTrigTowerGeometry.towerEtaBounds(ieta[ntower],eta1,eta2);
      eta = (eta1+eta2)/2;
      tpg_uncompressed[ntower] = float(_transcoder->hcaletValue(tpg_iter->id().ietaAbs(),int(tpg_e)));
      index[ntower] = tpg_iter->id().zside()*(100*tpg_iter->id().ietaAbs()+tpg_iter->id().iphi());
      out << "output[" <<index[ntower] << "] = " << ntower << semicolon <<"\n";
      ntower++;
    }
  tree.Fill();
  out.close();
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
