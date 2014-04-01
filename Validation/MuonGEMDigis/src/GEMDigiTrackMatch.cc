#include "Validation/MuonGEMDigis/interface/GEMDigiTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>

GEMDigiTrackMatch::GEMDigiTrackMatch(DQMStore* dbe, std::string simInputLabel , edm::ParameterSet cfg) : GEMTrackMatch(dbe,simInputLabel,cfg)
{
   minPt_  = cfg_.getUntrackedParameter<double>("gemDigiMinPt",5.0);
   minEta_ = cfg_.getUntrackedParameter<double>("gemDigiMinEta",1.55);
   maxEta_ = cfg_.getUntrackedParameter<double>("gemDigiMaxEta",2.18);
}

void GEMDigiTrackMatch::bookHisto(){
   const float PI=TMath::Pi();
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

GEMDigiTrackMatch::~GEMDigiTrackMatch() {  }

void GEMDigiTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
    SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup,theGEMGeometry);

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
      edm::LogInfo("GEMDigiTrackMatch")<<"it has no layer on sh hit!";
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
      edm::LogInfo("GEMDigiTrackMatch")<<"it has no layer on pad!";
    }

    // phi efficiency. 
    if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) {
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
    
    if ( track_.has_gem_dg_l1 /2 >= 1 ) {
      dg_lx_even_l1->Fill ( track_.gem_lx_even);
      dg_ly_even_l1->Fill ( track_.gem_ly_even);
    }
    if ( track_.has_gem_dg_l1 %2 == 1 ) { 
      dg_lx_odd_l1->Fill ( track_.gem_lx_odd);
      dg_ly_odd_l1->Fill ( track_.gem_ly_odd);
    }
    if ( track_.has_gem_dg_l2 /2  >=1 ) { 
      dg_lx_even_l2->Fill ( track_.gem_lx_even);
      dg_ly_even_l2->Fill ( track_.gem_ly_even);
    }
    if ( track_.has_gem_dg_l2 %2 == 1 ) {
      dg_lx_odd_l2->Fill ( track_.gem_lx_odd);
      dg_ly_odd_l2->Fill ( track_.gem_ly_odd);
    }
    if ( track_.has_gem_dg_l1 /2 >=1  || track_.has_gem_dg_l2 /2 >=1 ) {
      dg_lx_even_l1or2->Fill ( track_.gem_lx_even);
      dg_ly_even_l1or2->Fill ( track_.gem_ly_even);
    }
    if ( track_.has_gem_dg_l1 %2 ==1  || track_.has_gem_dg_l2 %2 ==1 ) {
      dg_lx_odd_l1or2->Fill ( track_.gem_lx_odd);
      dg_ly_odd_l1or2->Fill ( track_.gem_ly_odd);
    }
    if ( track_.has_gem_dg_l1 /2 >=1 && track_.has_gem_dg_l2 /2 >=1 ) {
      dg_lx_even_l1and2->Fill ( track_.gem_lx_even);
      dg_ly_even_l1and2->Fill ( track_.gem_ly_even);
    }
    if ( track_.has_gem_dg_l1 %2 ==1 && track_.has_gem_dg_l2 %2 ==1 ) {
      dg_lx_odd_l1and2->Fill ( track_.gem_lx_odd);
      dg_ly_odd_l1and2->Fill ( track_.gem_ly_odd);
    }
  }
}
