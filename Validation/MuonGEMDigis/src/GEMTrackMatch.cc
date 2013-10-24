#include "Validation/MuonGEMDigis/interface/GEMTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>

const float PI=TMath::Pi();


struct MySimTrack
{
  Float_t pt, eta, phi;
  Char_t charge;
  Char_t endcap;
  Char_t gem_sh_layer1, gem_sh_layer2;
  Char_t gem_dg_layer1, gem_dg_layer2;
  Char_t gem_pad_layer1, gem_pad_layer2;
  Float_t gem_sh_eta, gem_sh_phi;
  Float_t gem_sh_x, gem_sh_y;
  Float_t gem_dg_eta, gem_dg_phi;
  Float_t gem_pad_eta, gem_pad_phi;
  Float_t gem_lx_even, gem_ly_even;
  Float_t gem_lx_odd, gem_ly_odd;
  Char_t has_gem_sh_l1, has_gem_sh_l2;
  Char_t has_gem_dg_l1, has_gem_dg_l2;
  Char_t has_gem_pad_l1, has_gem_pad_l2;
  Float_t gem_trk_eta, gem_trk_phi, gem_trk_rho;
};



GEMTrackMatch::GEMTrackMatch(DQMStore* dbe, std::string simInputLabel , edm::ParameterSet cfg )
{
   //theEff_eta_dg_l1  =  dbe_->book1D("eff_eta_track_dg_gem_l1", "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack |#eta|;Eff.", 140,1.5,2.2);


   cfg_= cfg; 
   simInputLabel_= simInputLabel;
   dbe_= dbe;



   buildLUT();


  track_eta =  new TH1F("track_eta", "track_eta;SimTrack |#eta|;Eff.", 140,1.5,2.2);
  track_phi =  new TH1F("track_phi", "track_phi;SimTrack |#eta|;Eff.", 100,-PI,PI);

  dg_eta[0] = new TH1F("dg_eta_l1","dg_eta_l1",140,1.5,2.2);
  dg_eta[1] = new TH1F("dg_eta_l2","dg_eta_l2",140,1.5,2.2);
  dg_eta[2] = new TH1F("dg_eta_l1or2","dg_eta_l1or2",140,1.5,2.2);
  dg_eta[3] = new TH1F("dg_eta_l1and2","dg_eta_l1and2",140,1.5,2.2);


  dg_sh_eta_l1 = new TH1F("dg_sh_eta_l1","dg_sh_eta_l1",140,1.5,2.2);
  dg_sh_eta_l2 = new TH1F("dg_sh_eta_l2","dg_sh_eta_l2",140,1.5,2.2);
  dg_sh_eta_l1or2 = new TH1F("dg_sh_eta_l1or2","dg_sh_eta_l1or2",140,1.5,2.2);
  dg_sh_eta_l1and2 = new TH1F("dg_sh_eta_l1and2","dg_sh_eta_l1and2",140,1.5,2.2);


  dg_phi_l1 = new TH1F("dg_phi_l1","dg_phi_l1",100,-PI,PI);
  dg_phi_l2 = new TH1F("dg_phi_l2","dg_phi_l2",100,-PI,PI);
  dg_phi_l1or2 = new TH1F("dg_phi_l1or2","dg_phi_l1or2",100,-PI,PI);
  dg_phi_l1and2 = new TH1F("dg_phi_l1and2","dg_phi_l1and2",100,-PI,PI);
  
  dg_sh_phi_l1 = new TH1F("dg_sh_phi_l1","dg_sh_phi_l1",100,-PI,PI);
  dg_sh_phi_l2 = new TH1F("dg_sh_phi_l2","dg_sh_phi_l2",100,-PI,PI);
  dg_sh_phi_l1or2 = new TH1F("dg_sh_phi_l1or2","dg_sh_phi_l1or2",100,-PI,PI);
  dg_sh_phi_l1and2 = new TH1F("dg_sh_phi_l1and2","dg_sh_phi_l1and2",100,-PI,PI);

  pad_eta_l1 = new TH1F("pad_eta_l1","pad_eta_l1",140,1.5,2.2);
  pad_eta_l2 = new TH1F("pad_eta_l2","pad_eta_l2",140,1.5,2.2);
  pad_eta_l1or2 = new TH1F("pad_eta_l1or2","pad_eta_l1or2",140,1.5,2.2);

  pad_phi_l1 = new TH1F("pad_phi_l1","pad_phi_l1",100,-PI,PI);
  pad_phi_l2 = new TH1F("pad_phi_l2","pad_phi_l2",100,-PI,PI);
  pad_phi_l1or2 = new TH1F("pad_phi_l1or2","pad_phi_l1or2",100,-PI,PI);

  pad_sh_eta_l1 = new TH1F("pad_sh_eta_l1","pad_eta_l1",140,1.5,2.2);
  pad_sh_eta_l2 = new TH1F("pad_sh_eta_l2","pad_eta_l2",140,1.5,2.2);
  pad_sh_eta_l1or2 = new TH1F("pad_sh_eta_l1or2","pad_eta_l1or2",140,1.5,2.2);

  pad_sh_phi_l1 = new TH1F("pad_sh_phi_l1","pad_phi_l1",100,-PI,PI);
  pad_sh_phi_l2 = new TH1F("pad_sh_phi_l2","pad_phi_l2",100,-PI,PI);
  pad_sh_phi_l1or2 = new TH1F("pad_sh_phi_l1or2","pad_phi_l1or2",100,-PI,PI);


  copad_eta_l1and2 = new TH1F("copad_eta","copad_eta",140,1.5,2.2);
  copad_phi_l1and2 = new TH1F("copad_phi","copad_phi",100,-PI,PI);
  

  copad_sh_eta_l1and2 = new TH1F("copad_sh_eta","copad_sh_eta",140,1.5,2.2);
  copad_sh_phi_l1and2 = new TH1F("copad_sh_phi","copad_sh_phi",100,-PI,PI);


}





GEMTrackMatch::~GEMTrackMatch() {
 

}

bool GEMTrackMatch::isSimTrackGood(const SimTrack &t)
{
  // SimTrack selection
  if (t.noVertex()) return false;
  if (t.noGenpart()) return false;
  if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
  if (t.momentum().pt() < minPt_) return false;
  float eta = std::abs(t.momentum().eta());
  if (eta > 2.18 || eta < 1.55) return false; // no GEMs could be in such eta
  return true;
}


void GEMTrackMatch::buildLUT()
{
  std::vector<int> pos_ids;
  pos_ids.push_back(GEMDetId(1,1,1,1,36,1).rawId());

  std::vector<int> neg_ids;
  neg_ids.push_back(GEMDetId(-1,1,1,1,36,1).rawId());

  // VK: I would really suggest getting phis from GEMGeometry
  
  std::vector<float> phis;
  phis.push_back(0.);
  for(int i=1; i<37; ++i)
  {
    pos_ids.push_back(GEMDetId(1,1,1,1,i,1).rawId());
    neg_ids.push_back(GEMDetId(-1,1,1,1,i,1).rawId());
    phis.push_back(i*10.);
  }
  positiveLUT_ = std::make_pair(phis,pos_ids);
  negativeLUT_ = std::make_pair(phis,neg_ids);
}




std::pair<int,int> GEMTrackMatch::getClosestChambers(int region, float phi)
{
  auto& phis(positiveLUT_.first);
  auto upper = std::upper_bound(phis.begin(), phis.end(), phi);
  std::cout << "lower = " << upper - phis.begin() << std::endl;
  std::cout << "upper = " << upper - phis.begin() + 1 << std::endl;
  auto& LUT = (region == 1 ? positiveLUT_.second : negativeLUT_.second);
  return std::make_pair(LUT.at(upper - phis.begin()), (LUT.at((upper - phis.begin() + 1)%36)));
}


void GEMTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  MySimTrack track_;


  iEvent.getByLabel(simInputLabel_, sim_tracks);
  iEvent.getByLabel(simInputLabel_, sim_vertices);

  const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer & sim_trks = *sim_tracks.product();


  for (auto& t: sim_trks)
  {
    if (!isSimTrackGood(t)) continue;
    
    // match hits and digis to this SimTrack
    SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup);

    const SimHitMatcher& match_sh = match.simhits();
    const GEMDigiMatcher& match_gd = match.gemDigis();

    track_.pt = t.momentum().pt();
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    track_.charge = t.charge();
    track_.endcap = (track_.eta > 0.) ? 1 : -1;
    track_.gem_sh_layer1 = 0;
    track_.gem_sh_layer2 = 0;
    track_.gem_dg_layer1 = 0;
    track_.gem_dg_layer2 = 0;
    track_.gem_pad_layer1 = 0;
    track_.gem_pad_layer2 = 0;
    track_.gem_sh_eta = -9.;
    track_.gem_sh_phi = -9.;
    track_.gem_sh_x = -999;
    track_.gem_sh_y = -999;
    track_.gem_dg_eta = -9.;
    track_.gem_dg_phi = -9.;
    track_.gem_pad_eta = -9.;
    track_.gem_pad_phi = -9.;
    track_.gem_trk_rho = -999.;
    track_.gem_lx_even = -999.;
    track_.gem_ly_even = -999.;
    track_.gem_lx_odd = -999.;
    track_.gem_ly_odd = -999.;
    track_.has_gem_sh_l1 = 0;
    track_.has_gem_sh_l2 = 0;
    track_.has_gem_dg_l1 = 0;
    track_.has_gem_dg_l2 = 0;
    track_.has_gem_pad_l1 = 0;
    track_.has_gem_pad_l2 = 0;





    // Calculation of the localXY efficiency
    GlobalPoint gp_track(match_sh.propagatedPositionGEM());
    track_.gem_trk_eta = gp_track.eta();
    track_.gem_trk_phi = gp_track.phi();
    track_.gem_trk_rho = gp_track.perp();
    
    float track_angle = gp_track.phi().degrees();
    if (track_angle < 0.) track_angle += 360.;
    std::cout << "track angle = " << track_angle << std::endl;
    const int track_region = (gp_track.z() > 0 ? 1 : -1);
    
    // closest chambers in phi
    const std::pair<int,int>  mypair = getClosestChambers(track_region, track_angle);
    
    // chambers
    GEMDetId detId_first(mypair.first);
    GEMDetId detId_second(mypair.second);

    // assignment of local even and odd chambers (there is always an even and an odd chamber)
    bool firstIsOdd = detId_first.chamber() & 1;    // "i & 1" is same as "i % 2"
    
    GEMDetId detId_even_L1(firstIsOdd ? detId_second : detId_first);
    GEMDetId detId_odd_L1(firstIsOdd ? detId_first : detId_second);

    auto even_partition = theGEMGeometry->idToDetUnit(detId_even_L1)->surface();
    auto odd_partition = theGEMGeometry->idToDetUnit(detId_odd_L1)->surface();

    // global positions of partitions' centers
    LocalPoint p0(0.,0.,0.);
    GlobalPoint gp_even_partition = even_partition.toGlobal(p0);
    GlobalPoint gp_odd_partition = odd_partition.toGlobal(p0);
    
    LocalPoint lp_track_even_partition = even_partition.toLocal(gp_track);
    LocalPoint lp_track_odd_partition = odd_partition.toLocal(gp_track);

    // track chamber local x is the same as track partition local x
    track_.gem_lx_even = lp_track_even_partition.x();
    track_.gem_lx_odd = lp_track_odd_partition.x();

    // track chamber local y is the same as track partition local y
    // corrected for partition's local y WRT chamber
    track_.gem_ly_even = lp_track_even_partition.y() + (gp_even_partition.perp() - radiusCenter_);
    track_.gem_ly_odd = lp_track_odd_partition.y() + (gp_odd_partition.perp() - radiusCenter_);



        // ** GEM SimHits ** //
    auto gem_sh_ids_sch = match_sh.superChamberIdsGEM();
    for(auto d: gem_sh_ids_sch)
    {
      auto gem_simhits = match_sh.hitsInSuperChamber(d);
      auto gem_simhits_gp = match_sh.simHitsMeanPosition(gem_simhits);

      track_.gem_sh_eta = gem_simhits_gp.eta();
      track_.gem_sh_phi = gem_simhits_gp.phi();
      track_.gem_sh_x = gem_simhits_gp.x();
      track_.gem_sh_y = gem_simhits_gp.y();

      std::cout<<track_.gem_sh_eta<<"   "<<track_.gem_sh_phi<<"   "<<track_.gem_sh_x<<"   "<<track_.gem_sh_y<<std::endl;
    }

    auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      GEMDetId id(d);
      bool odd(id.chamber() & 1);
      
      if (id.layer() == 1)
      {
        if (odd) track_.gem_sh_layer1 |= 1;
        else track_.gem_sh_layer1 |= 2;
      }
      else if (id.layer() == 2)
      {
        if (odd) track_.gem_sh_layer2 |= 1;
        else track_.gem_sh_layer2 |= 2;
      }
    }

    // ** GEM Digis, Pads and CoPads ** //
    auto gem_dg_ids_sch = match_gd.superChamberIds();
    for(auto d: gem_dg_ids_sch)
    {
      auto gem_digis = match_gd.digisInSuperChamber(d);
      auto gem_dg_gp = match_gd.digisMeanPosition(gem_digis);

      track_.gem_dg_eta = gem_dg_gp.eta();
      track_.gem_dg_phi = gem_dg_gp.phi();        

      auto gem_pads = match_gd.padsInSuperChamber(d);
      auto gem_pad_gp = match_gd.digisMeanPosition(gem_pads);        
      
      track_.gem_pad_eta = gem_pad_gp.eta();
      track_.gem_pad_phi = gem_pad_gp.phi();        
    }

    auto gem_dg_ids_ch = match_gd.chamberIds();
    for(auto d: gem_dg_ids_ch)
    {
      GEMDetId id(d);
      bool odd(id.chamber() & 1);
      
      if (id.layer() == 1)
      {
        if (odd)
        {
         track_.gem_dg_layer1 |= 1;
         track_.gem_pad_layer1 |= 1;
        }
        else
        {
         track_.gem_dg_layer1 |= 2;
         track_.gem_pad_layer1 |= 2;
        }
      }
      else if (id.layer() == 2)
      {
        if (odd)
        {
         track_.gem_dg_layer2 |= 1;
         track_.gem_pad_layer2 |= 1;
        }
        else
        {
         track_.gem_dg_layer2 |= 2;
         track_.gem_pad_layer2 |= 2;
        }
      }
    }

    // Construct Chamber DetIds from the "projected" ids:
    GEMDetId id_ch_even_L1(detId_even_L1.region(), detId_even_L1.ring(), detId_even_L1.station(), 1, detId_even_L1.chamber(), 0);
    GEMDetId id_ch_odd_L1(detId_odd_L1.region(), detId_odd_L1.ring(), detId_odd_L1.station(), 1, detId_odd_L1.chamber(), 0);
    GEMDetId id_ch_even_L2(detId_even_L1.region(), detId_even_L1.ring(), detId_even_L1.station(), 2, detId_even_L1.chamber(), 0);
    GEMDetId id_ch_odd_L2(detId_odd_L1.region(), detId_odd_L1.ring(), detId_odd_L1.station(), 2, detId_odd_L1.chamber(), 0);

    // check if track has sh
    if(gem_sh_ids_ch.count(id_ch_even_L1)!=0) track_.has_gem_sh_l1 |= 2;
    if(gem_sh_ids_ch.count(id_ch_odd_L1)!=0) track_.has_gem_sh_l1 |= 1;
    if(gem_sh_ids_ch.count(id_ch_even_L2)!=0) track_.has_gem_sh_l2 |= 2;
    if(gem_sh_ids_ch.count(id_ch_odd_L2)!=0) track_.has_gem_sh_l2 |= 1;

    // check if track has dg
    if(gem_dg_ids_ch.count(id_ch_even_L1)!=0){
      track_.has_gem_dg_l1 |= 2;
      track_.has_gem_pad_l1 |= 2;
    }
    if(gem_dg_ids_ch.count(id_ch_odd_L1)!=0){
      track_.has_gem_dg_l1 |= 1;
      track_.has_gem_pad_l1 |= 1;
    }
    if(gem_dg_ids_ch.count(id_ch_even_L2)!=0){
      track_.has_gem_dg_l2 |= 2;
      track_.has_gem_pad_l2 |= 2;
    }
    if(gem_dg_ids_ch.count(id_ch_odd_L2)!=0){
      track_.has_gem_dg_l2 |= 1;
      track_.has_gem_pad_l2 |= 1;
    }
  track_eta->Fill( track_.eta );
  if ( track_.gem_dg_layer1 > 0 ) {
    dg_eta[0]->Fill ( track_.gem_dg_eta );
    dg_eta[2]->Fill( track_.gem_dg_eta);
  }
  else if ( track_.gem_dg_layer2 > 0 ) { 
    dg_eta[1]->Fill ( track_.gem_dg_eta );
    dg_eta[2]->Fill( track_.gem_dg_eta);
  }
  else if ( track_.gem_dg_layer1 > 0 && track_.gem_dg_layer2>0 ) {
    dg_eta[3]->Fill( track_.gem_dg_eta);
  }
  else { std::cout<<"dg_eta : "<<track_.gem_dg_eta; }
   






 

    




  }
}
