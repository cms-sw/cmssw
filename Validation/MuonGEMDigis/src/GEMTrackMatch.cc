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
  Char_t gem_sh_layer1, gem_sh_layer2;
  Char_t gem_dg_layer1, gem_dg_layer2;
  Char_t gem_pad_layer1, gem_pad_layer2;
};



GEMTrackMatch::GEMTrackMatch(DQMStore* dbe, std::string simInputLabel , edm::ParameterSet cfg )
{
   //theEff_eta_dg[0]  =  dbe_->book1D("eff_eta_track_dg_gem_l1", "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack |#eta|;Eff.", 140,1.5,2.2);


   cfg_= cfg; 
   simInputLabel_= simInputLabel;
   dbe_= dbe;



   buildLUT();


  track_eta =  new TH1F("track_eta", "track_eta;SimTrack |#eta|;# of tracks", 140,1.5,2.2);
  track_phi =  new TH1F("track_phi", "track_phi;SimTrack |#eta|;# of tracks", 100,-PI,PI);

  track_dg_eta =  new TH1F("track_dg_eta", "track_eta;SimTrack |#eta|;# of tracks", 140,1.5,2.2);
  track_sh_eta =  new TH1F("track_sh_eta", "track_eta;SimTrack |#eta|;# of tracks", 140,1.5,2.2);


  dg_eta[0] = new TH1F("dg_eta_l1","dg_eta_l1",140,1.5,2.2);
  dg_eta[1] = new TH1F("dg_eta_l2","dg_eta_l2",140,1.5,2.2);
  dg_eta[2] = new TH1F("dg_eta_l1or2","dg_eta_l1or2",140,1.5,2.2);
  dg_eta[3] = new TH1F("dg_eta_l1and2","dg_eta_l1and2",140,1.5,2.2);


  dg_sh_eta[0] = new TH1F("dg_sh_eta_l1","dg_sh_eta_l1",140,1.5,2.2);
  dg_sh_eta[1] = new TH1F("dg_sh_eta_l2","dg_sh_eta_l2",140,1.5,2.2);
  dg_sh_eta[2] = new TH1F("dg_sh_eta_l1or2","dg_sh_eta_l1or2",140,1.5,2.2);
  dg_sh_eta[3] = new TH1F("dg_sh_eta_l1and2","dg_sh_eta_l1and2",140,1.5,2.2);


  dg_phi[0] = new TH1F("dg_phi_l1","dg_phi_l1",100,-PI,PI);
  dg_phi[1] = new TH1F("dg_phi_l2","dg_phi_l2",100,-PI,PI);
  dg_phi[2] = new TH1F("dg_phi_l1or2","dg_phi_l1or2",100,-PI,PI);
  dg_phi[3] = new TH1F("dg_phi_l1and2","dg_phi_l1and2",100,-PI,PI);
  
  dg_sh_phi[0] = new TH1F("dg_sh_phi_l1","dg_sh_phi_l1",100,-PI,PI);
  dg_sh_phi[1] = new TH1F("dg_sh_phi_l2","dg_sh_phi_l2",100,-PI,PI);
  dg_sh_phi[2] = new TH1F("dg_sh_phi_l1or2","dg_sh_phi_l1or2",100,-PI,PI);
  dg_sh_phi[3] = new TH1F("dg_sh_phi_l1and2","dg_sh_phi_l1and2",100,-PI,PI);

  pad_eta[0] = new TH1F("pad_eta_l1","pad_eta_l1",140,1.5,2.2);
  pad_eta[1] = new TH1F("pad_eta_l2","pad_eta_l2",140,1.5,2.2);
  pad_eta[2] = new TH1F("pad_eta_l1or2","pad_eta_l1or2",140,1.5,2.2);
  pad_eta[3] = new TH1F("copad_eta","copad_eta",140,1.5,2.2);

  pad_phi[0] = new TH1F("pad_phi_l1","pad_phi_l1",100,-PI,PI);
  pad_phi[1] = new TH1F("pad_phi_l2","pad_phi_l2",100,-PI,PI);
  pad_phi[2] = new TH1F("pad_phi_l1or2","pad_phi_l1or2",100,-PI,PI);
  pad_phi[3] = new TH1F("copad_phi","copad_phi",100,-PI,PI);


  



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
    if (!isSimTrackGood(t)) 
    { continue; } 
    //{ printf("skip!!\n"); continue; }
    
    // match hits and digis to this SimTrack
    SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup);

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





    
    
    

    // ** GEM SimHits ** //

    auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      GEMDetId id(d);
      
      if (id.layer() == 1)
      {
        track_.gem_sh_layer1 =1;
      }
      else if (id.layer() == 2)
      {
        track_.gem_sh_layer2 =1;       // 10
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
      else { printf("no layer?? please, check.\n"); }
    }




  track_eta->Fill( fabs( track_.eta)  );
//  if( track_.gem_dg_eta!=-9.) {
    track_dg_eta->Fill( fabs( track_.eta));
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
      printf("it has no layer on digi!\n");
    }

//  }

//  if( track_.gem_sh_eta != -9.) {
    track_sh_eta->Fill( fabs( track_.eta));
  
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
      printf("it has no layer on sh hit!\n");
    }
//  }


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
      printf("it has no layer on pad!\n");
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





 }




}
