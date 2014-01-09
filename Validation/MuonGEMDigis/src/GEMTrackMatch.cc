#include "Validation/MuonGEMDigis/interface/GEMTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>






GEMTrackMatch::GEMTrackMatch(DQMStore* dbe, std::string simInputLabel , edm::ParameterSet cfg)
{
   const float PI=TMath::Pi();
   cfg_= cfg; 
   simInputLabel_= simInputLabel;
   dbe_= dbe;
   minPt_  = cfg_.getUntrackedParameter<double>("gemDigiMinPt",5.0);
   minEta_ = cfg_.getUntrackedParameter<double>("gemDigiMinEta",1.55);
   maxEta_ = cfg_.getUntrackedParameter<double>("gemDigiMaxEta",2.18);
   buildLUT();
   track_eta =  dbe_->book1D("track_eta", "track_eta;SimTrack |#eta|;# of tracks", 140,1.5,2.2);
   track_phi =  dbe_->book1D("track_phi", "track_phi;SimTrack |#eta|;# of tracks", 100,-PI,PI);

   dg_eta[0] = dbe_->book1D("dg_eta_l1","dg_eta_l1",140,1.5,2.2);
   dg_eta[1] = dbe_->book1D("dg_eta_l2","dg_eta_l2",140,1.5,2.2);
   dg_eta[2] = dbe_->book1D("dg_eta_l1or2","dg_eta_l1or2",140,1.5,2.2);
   dg_eta[3] = dbe_->book1D("dg_eta_l1and2","dg_eta_l1and2",140,1.5,2.2);

   dg_sh_eta[0] = dbe_->book1D("dg_sh_eta_l1","dg_sh_eta_l1",140,1.5,2.2);
   dg_sh_eta[1] = dbe_->book1D("dg_sh_eta_l2","dg_sh_eta_l2",140,1.5,2.2);
   dg_sh_eta[2] = dbe_->book1D("dg_sh_eta_l1or2","dg_sh_eta_l1or2",140,1.5,2.2);
   dg_sh_eta[3] = dbe_->book1D("dg_sh_eta_l1and2","dg_sh_eta_l1and2",140,1.5,2.2);

   dg_phi[0] = dbe_->book1D("dg_phi_l1","dg_phi_l1",100,-PI,PI);
   dg_phi[1] = dbe_->book1D("dg_phi_l2","dg_phi_l2",100,-PI,PI);
   dg_phi[2] = dbe_->book1D("dg_phi_l1or2","dg_phi_l1or2",100,-PI,PI);
   dg_phi[3] = dbe_->book1D("dg_phi_l1and2","dg_phi_l1and2",100,-PI,PI);
  
   dg_sh_phi[0] = dbe_->book1D("dg_sh_phi_l1","dg_sh_phi_l1",100,-PI,PI);
   dg_sh_phi[1] = dbe_->book1D("dg_sh_phi_l2","dg_sh_phi_l2",100,-PI,PI);
   dg_sh_phi[2] = dbe_->book1D("dg_sh_phi_l1or2","dg_sh_phi_l1or2",100,-PI,PI);
   dg_sh_phi[3] = dbe_->book1D("dg_sh_phi_l1and2","dg_sh_phi_l1and2",100,-PI,PI);

   pad_eta[0] = dbe_->book1D("pad_eta_l1","pad_eta_l1",140,1.5,2.2);
   pad_eta[1] = dbe_->book1D("pad_eta_l2","pad_eta_l2",140,1.5,2.2);
   pad_eta[2] = dbe_->book1D("pad_eta_l1or2","pad_eta_l1or2",140,1.5,2.2);
   pad_eta[3] = dbe_->book1D("copad_eta","copad_eta",140,1.5,2.2);

   pad_phi[0] = dbe_->book1D("pad_phi_l1","pad_phi_l1",100,-PI,PI);
   pad_phi[1] = dbe_->book1D("pad_phi_l2","pad_phi_l2",100,-PI,PI);
   pad_phi[2] = dbe_->book1D("pad_phi_l1or2","pad_phi_l1or2",100,-PI,PI);
   pad_phi[3] = dbe_->book1D("copad_phi","copad_phi",100,-PI,PI);

   dg_lx_even        = dbe_->book1D("dg_lx_even","dg_lx_even",100,-100,100); 
   dg_lx_even_l1     = dbe_->book1D("dg_lx_even_l1","dg_lx_even_l1",100,-100,100);
   dg_lx_even_l2     = dbe_->book1D("dg_lx_even_l2","dg_lx_even_l2",100,-100,100);
   dg_lx_even_l1or2  = dbe_->book1D("dg_lx_even_l1or2","dg_lx_even_l1or2",100,-100,100);
   dg_lx_even_l1and2 = dbe_->book1D("dg_lx_even_l1and2","dg_lx_even_l1and2",100,-100,100);

   dg_ly_even        = dbe_->book1D("dg_ly_even","dg_ly_even",100,-100,100);
   dg_ly_even_l1     = dbe_->book1D("dg_ly_even_l1","dg_ly_even_l1",100,-100,100);
   dg_ly_even_l2     = dbe_->book1D("dg_ly_even_l2","dg_ly_even_l2",100,-100,100);
   dg_ly_even_l1or2  = dbe_->book1D("dg_ly_even_l1or2","dg_ly_even_l1or2",100,-100,100);
   dg_ly_even_l1and2 = dbe_->book1D("dg_ly_even_l1and2","dg_ly_even_l1and2",100,-100,100);

   dg_lx_odd        = dbe_->book1D("dg_lx_odd","dg_lx_odd",100,-100,100);
   dg_lx_odd_l1     = dbe_->book1D("dg_lx_odd_l1","dg_lx_odd_l1",100,-100,100);
   dg_lx_odd_l2     = dbe_->book1D("dg_lx_odd_l2","dg_lx_odd_l2",100,-100,100);
   dg_lx_odd_l1or2  = dbe_->book1D("dg_lx_odd_l1or2","dg_lx_odd_l1or2",100,-100,100);
   dg_lx_odd_l1and2 = dbe_->book1D("dg_lx_odd_l1and2","dg_lx_odd_l1and2",100,-100,100);
  
   dg_ly_odd        = dbe_->book1D("dg_ly_odd","dg_ly_odd",100,-100,100);
   dg_ly_odd_l1     = dbe_->book1D("dg_ly_odd_l1","dg_ly_odd_l1",100,-100,100);
   dg_ly_odd_l2     = dbe_->book1D("dg_ly_odd_l2","dg_ly_odd_l2",100,-100,100);
   dg_ly_odd_l1or2  = dbe_->book1D("dg_ly_odd_l1or2","dg_ly_odd_l1or2",100,-100,100);
   dg_ly_odd_l1and2 = dbe_->book1D("dg_ly_odd_l1and2","dg_ly_odd_l1and2",100,-100,100);
}





GEMTrackMatch::~GEMTrackMatch() {
}

bool GEMTrackMatch::isSimTrackGood(const SimTrack &t)
{

  // SimTrack selection
  if (t.noVertex())   return false; 
  if (t.noGenpart()) return false;
  if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
  if (t.momentum().pt() < 5 ) return false;
  float eta = fabs(t.momentum().eta());
  if (eta > maxEta_ || eta < minEta_ ) return false; // no GEMs could be in such eta

  return true;
}


void GEMTrackMatch::buildLUT()
{
  
  const int maxChamberId_ = GEMDetId().maxChamberId; 
  std::vector<int> pos_ids;
  pos_ids.push_back(GEMDetId(1,1,1,1,maxChamberId_,1).rawId());

  std::vector<int> neg_ids;
  neg_ids.push_back(GEMDetId(-1,1,1,1,maxChamberId_,1).rawId());

  // VK: I would really suggest getting phis from GEMGeometry
  
  std::vector<float> phis;
  phis.push_back(0.);
  for(int i=1; i<maxChamberId_+1; ++i)
  {
    pos_ids.push_back(GEMDetId(1,1,1,1,i,1).rawId());
    neg_ids.push_back(GEMDetId(-1,1,1,1,i,1).rawId());
    phis.push_back(i*10.);
  }
  positiveLUT_ = std::make_pair(phis,pos_ids);
  negativeLUT_ = std::make_pair(phis,neg_ids);

}


void GEMTrackMatch::setGeometry(const GEMGeometry* geom)
{
  theGEMGeometry = geom;
  const auto top_chamber = static_cast<const GEMEtaPartition*>(theGEMGeometry->idToDetUnit(GEMDetId(1,1,1,1,1,1)));
  const int nEtaPartitions(theGEMGeometry->chamber(GEMDetId(1,1,1,1,1,1))->nEtaPartitions());
  const auto bottom_chamber = static_cast<const GEMEtaPartition*>(theGEMGeometry->idToDetUnit(GEMDetId(1,1,1,1,1,nEtaPartitions)));
  const float top_half_striplength = top_chamber->specs()->specificTopology().stripLength()/2.;
  const float bottom_half_striplength = bottom_chamber->specs()->specificTopology().stripLength()/2.;
  const LocalPoint lp_top(0., top_half_striplength, 0.);
  const LocalPoint lp_bottom(0., -bottom_half_striplength, 0.);
  const GlobalPoint gp_top = top_chamber->toGlobal(lp_top);
  const GlobalPoint gp_bottom = bottom_chamber->toGlobal(lp_bottom);

  radiusCenter_ = (gp_bottom.perp() + gp_top.perp())/2.;
  chamberHeight_ = gp_top.perp() - gp_bottom.perp();

}  


std::pair<int,int> GEMTrackMatch::getClosestChambers(int region, float phi)
{
  
  auto& phis(positiveLUT_.first);
  auto upper = std::upper_bound(phis.begin(), phis.end(), phi);
  auto& LUT = (region == 1 ? positiveLUT_.second : negativeLUT_.second);


  return std::make_pair(LUT.at(upper - phis.begin()), (LUT.at((upper - phis.begin() + 1)%36)));
}


void GEMTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  struct MySimTrack
  {
    Float_t pt, eta, phi;
    Char_t gem_sh_layer1, gem_sh_layer2;
    Char_t gem_dg_layer1, gem_dg_layer2;
    Char_t gem_pad_layer1, gem_pad_layer2;
    Float_t gem_lx_even, gem_ly_even;
    Float_t gem_lx_odd, gem_ly_odd;
    Char_t has_gem_dg_l1, has_gem_dg_l2;
    Char_t has_gem_pad_l1, has_gem_pad_l2;
    Char_t has_gem_sh_l1, has_gem_sh_l2;
  };
  MySimTrack track_;

  iEvent.getByLabel(simInputLabel_, sim_tracks);
  iEvent.getByLabel(simInputLabel_, sim_vertices);

  const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer & sim_trks = *sim_tracks.product();


  for (auto& t: sim_trks)
  {
    if (!isSimTrackGood(t)) 
    { continue; } 
    //{ printf("skip!!\n"); continue; }
    
    // match hits and digis to this SimTrack
    SimTrackDigiMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup);

    const SimHitMatcher& match_sh = match.simhits();
    const GEMDigiMatcher& match_gd = match.gemDigis();

    track_.pt = t.momentum().pt();
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    track_.gem_sh_layer1 = 0;
    track_.gem_sh_layer2 = 0;
    track_.gem_dg_layer1 = 0;
    track_.gem_dg_layer2 = 0;
    track_.gem_pad_layer1 = 0;
    track_.gem_pad_layer2 = 0;
    track_.gem_lx_even =0;
    track_.gem_ly_even =0;
    track_.gem_lx_odd  =0;
    track_.gem_ly_odd  =0;
    track_.has_gem_dg_l1 = 0;
    track_.has_gem_dg_l2 = 0;
    track_.has_gem_pad_l1 = 0;
    track_.has_gem_pad_l2 = 0;

    // ** GEM SimHits ** //
    auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      GEMDetId id(d);
      
      if (id.layer() == 1)
      {
        track_.gem_sh_layer1 = 1;
      }
      else if (id.layer() == 2)
      {
        track_.gem_sh_layer2 = 1;       
      }
    }


    // ** GEM Digis, Pads and CoPads ** //


    auto gem_dg_ids_ch = match_gd.chamberIds();

    for(auto d: gem_dg_ids_ch)
    {
      GEMDetId id(d);
      if (id.layer() == 1)
      {
         track_.gem_dg_layer1 = 1;
         track_.gem_pad_layer1 = 1;
      }
      else if (id.layer() == 2)
      {
         track_.gem_dg_layer2 = 1;
         track_.gem_pad_layer2 = 1;
      }
      //else { edm::LogInfo("GEMDIGI")<<"GEM Digi did not found on any layer." }
    }

    track_eta->Fill( fabs( track_.eta)  );
    if ( track_.gem_dg_layer1 > 0 ) {
      dg_eta[0]->Fill ( fabs( track_.eta ) );
    }
    if ( track_.gem_dg_layer2 > 0 ) { 
      dg_eta[1]->Fill ( fabs( track_.eta ) );
    }
    if ( track_.gem_dg_layer1 > 0 || track_.gem_dg_layer2>0 ) { 
      dg_eta[2]->Fill ( fabs( track_.eta ) );
    }
    if ( track_.gem_dg_layer1 > 0 && track_.gem_dg_layer2>0 ) {
      dg_eta[3]->Fill( fabs(track_.eta) );
    }
    if ( track_.gem_dg_layer1 ==0 && track_.gem_dg_layer2==0) {
      edm::LogInfo("GEMDIGI")<<"it has no layer on digi!";
    }


    if ( track_.gem_sh_layer1 > 0 ) {
      dg_sh_eta[0]->Fill ( fabs(track_.eta)); 
    }
    if ( track_.gem_sh_layer2 > 0 ) {
      dg_sh_eta[1]->Fill( fabs(track_.eta));
    }
    if (track_.gem_sh_layer1 >0 || track_.gem_sh_layer2>0 ) {
      dg_sh_eta[2]->Fill( fabs(track_.eta));
    }
    if (track_.gem_sh_layer1 >0 && track_.gem_sh_layer2>0 ) {
      dg_sh_eta[3]->Fill( fabs(track_.eta));
    }
    if ( track_.gem_sh_layer1 ==0 && track_.gem_sh_layer2==0) {
      edm::LogInfo("GEMSIM")<<"it has no layer on sh hit!";
    }


    if ( track_.gem_pad_layer1 > 0 ) {
      pad_eta[0]->Fill ( fabs(track_.eta) );
    }
    if ( track_.gem_pad_layer2 > 0 ) { 
      pad_eta[1]->Fill ( fabs(track_.eta) );
    }
    if ( track_.gem_pad_layer1 > 0 || track_.gem_pad_layer2>0 ) {
      pad_eta[2]->Fill( fabs(track_.eta));
    }
    if ( track_.gem_pad_layer1 > 0 && track_.gem_pad_layer2>0 ) {
      pad_eta[3]->Fill( fabs(track_.eta));
    }
    if ( track_.gem_pad_layer1==0 && track_.gem_pad_layer2==0) {
      edm::LogInfo("GEMPAD")<<"it has no layer on pad!";
    }

    // phi efficiency. 
    if( fabs(track_.eta) < 2.12 && fabs( track_.eta) > 1.64 ) {
      track_phi->Fill( track_.phi);
      if ( track_.gem_dg_layer1 > 0 ) {
          dg_phi[0]->Fill ( track_.phi );
      }
      if ( track_.gem_dg_layer2 > 0 ) { 
        dg_phi[1]->Fill ( track_.phi );
      }
      if ( track_.gem_dg_layer1 > 0 || track_.gem_dg_layer2 > 0 ) {
        dg_phi[2]->Fill( track_.phi );
      }
      if ( track_.gem_dg_layer1 > 0 && track_.gem_dg_layer2 > 0 ) {
        dg_phi[3]->Fill( track_.phi );
      }
      
      if ( track_.gem_sh_layer1 > 0 ) {
        dg_sh_phi[0]->Fill ( track_.phi);
      }
      if ( track_.gem_sh_layer2 > 0 ) {
        dg_sh_phi[1]->Fill( track_.phi);
      }
      if (track_.gem_sh_layer1 >0 || track_.gem_sh_layer2>0 ) {
        dg_sh_phi[2]->Fill( track_.phi);
      }
      if (track_.gem_sh_layer1 >0 && track_.gem_sh_layer2>0 ) {
        dg_sh_phi[3]->Fill( track_.phi);
      }


      if ( track_.gem_pad_layer1 > 0 ) {
        pad_phi[0]->Fill ( track_.phi );
      }
      if ( track_.gem_pad_layer2 > 0 ) { 
        pad_phi[1]->Fill ( track_.phi );
      }
      if ( track_.gem_pad_layer1 > 0 || track_.gem_pad_layer2>0 ) {
        pad_phi[2]->Fill( track_.phi );
      }
      if ( track_.gem_pad_layer1 > 0 && track_.gem_pad_layer2>0 ) {
        pad_phi[3]->Fill( track_.phi );
      }

    }  


    // Calculation of the localXY efficiency
    GlobalPoint gp_track(match_sh.propagatedPositionGEM());
    
    float track_angle = gp_track.phi().degrees();
    if (track_angle < 0.) track_angle += 360.;
    const int track_region = (gp_track.z() > 0 ? 1 : -1);

    // closest chambers in phi
    const auto mypair = getClosestChambers(track_region, track_angle);
 
    GEMDetId detId_first(mypair.first);
    GEMDetId detId_second(mypair.second);
 
    // assignment of local even and odd chambers (there is always an even and an odd chamber)
    bool firstIsOdd = detId_first.chamber() & 1;

    GEMDetId detId_even_L1(firstIsOdd ? detId_second : detId_first);
    GEMDetId detId_odd_L1(firstIsOdd ? detId_first : detId_second);

    auto even_partition = theGEMGeometry->idToDetUnit(detId_even_L1)->surface();
    auto odd_partition = theGEMGeometry->idToDetUnit(detId_odd_L1)->surface();

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

    GEMDetId id_ch_even_L1(detId_even_L1.region(), detId_even_L1.ring(), detId_even_L1.station(), 1, detId_even_L1.chamber(), 0);
    GEMDetId id_ch_odd_L1(detId_odd_L1.region(), detId_odd_L1.ring(), detId_odd_L1.station(), 1, detId_odd_L1.chamber(), 0);
    GEMDetId id_ch_even_L2(detId_even_L1.region(), detId_even_L1.ring(), detId_even_L1.station(), 2, detId_even_L1.chamber(), 0);
    GEMDetId id_ch_odd_L2(detId_odd_L1.region(), detId_odd_L1.ring(), detId_odd_L1.station(), 2, detId_odd_L1.chamber(), 0);

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

    dg_lx_even->Fill( track_.gem_lx_even);
    dg_lx_odd->Fill( track_.gem_lx_odd);
    dg_ly_even->Fill( track_.gem_ly_even);
    dg_ly_odd->Fill( track_.gem_ly_odd);
    
    if ( track_.has_gem_dg_l1 > 0 ) {
      dg_lx_even_l1->Fill ( track_.gem_lx_even);
      dg_ly_even_l1->Fill ( track_.gem_ly_even);
      dg_lx_odd_l1->Fill ( track_.gem_lx_odd);
      dg_ly_odd_l1->Fill ( track_.gem_ly_odd);
    }
    if ( track_.has_gem_dg_l2 > 0 ) { 
      dg_lx_even_l2->Fill ( track_.gem_lx_even);
      dg_ly_even_l2->Fill ( track_.gem_ly_even);
      dg_lx_odd_l2->Fill ( track_.gem_lx_odd);
      dg_ly_odd_l2->Fill ( track_.gem_ly_odd);
    }
    if ( track_.has_gem_dg_l1 > 0 || track_.has_gem_dg_l2 > 0 ) {
      dg_lx_even_l1or2->Fill ( track_.gem_lx_even);
      dg_ly_even_l1or2->Fill ( track_.gem_ly_even);
      dg_lx_odd_l1or2->Fill ( track_.gem_lx_odd);
      dg_ly_odd_l1or2->Fill ( track_.gem_ly_odd);
    }
    if ( track_.has_gem_dg_l1 > 0 && track_.has_gem_dg_l2 > 0 ) {
      dg_lx_even_l1and2->Fill ( track_.gem_lx_even);
      dg_ly_even_l1and2->Fill ( track_.gem_ly_even);
      dg_lx_odd_l1and2->Fill ( track_.gem_lx_odd);
      dg_ly_odd_l1and2->Fill ( track_.gem_ly_odd);
    }
  }
}
