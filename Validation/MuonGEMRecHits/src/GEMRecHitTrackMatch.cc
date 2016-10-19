#include "Validation/MuonGEMRecHits/interface/GEMRecHitTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>

using namespace std;
GEMRecHitTrackMatch::GEMRecHitTrackMatch(const edm::ParameterSet& ps) : GEMTrackMatch(ps)
{
  std::string simInputLabel_ = ps.getUntrackedParameter<std::string>("simInputLabel");
  simHitsToken_ = consumes<edm::PSimHitContainer>(edm::InputTag(simInputLabel_,"MuonGEMHits"));
  simTracksToken_ = consumes< edm::SimTrackContainer >(ps.getParameter<edm::InputTag>("simTrackCollection"));
  simVerticesToken_ = consumes< edm::SimVertexContainer >(ps.getParameter<edm::InputTag>("simVertexCollection"));

  gem_recHitToken_ = consumes<GEMRecHitCollection>(ps.getParameter<edm::InputTag>("gemRecHitInput"));

  detailPlot_ = ps.getParameter<bool>("detailPlot");
  cfg_ = ps;
}

void GEMRecHitTrackMatch::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& run, edm::EventSetup const & iSetup){
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry& geom = *hGeom;
  setGeometry(geom);

  const float PI=TMath::Pi();
  const char* l_suffix[4] = {"_l1","_l2","_l1or2","_l1and2"};
  const char* s_suffix[2] = {"_st1","_st2"};
  const char* c_suffix[3] = {"_even","_odd","_all"};

  nstation = geom.regions()[0]->stations().size(); 
  for( unsigned int j=0 ; j<nstation ; j++) {
      string track_eta_name  = string("track_eta")+s_suffix[j];
      string track_eta_title = string("track_eta")+";SimTrack |#eta|;# of tracks";
      track_eta[j] = ibooker.book1D(track_eta_name.c_str(), track_eta_title.c_str(),140,minEta_,maxEta_);

      for ( unsigned int k = 0 ; k<3 ; k++) {
          string suffix = string(s_suffix[j])+ string(c_suffix[k]);
          string track_phi_name  = string("track_phi")+suffix;
          string track_phi_title = string("track_phi")+suffix+";SimTrack #phi;# of tracks";
          track_phi[j][k] = ibooker.book1D(track_phi_name.c_str(), track_phi_title.c_str(),200,-PI,PI);
      }


      for( unsigned int i=0 ; i< 4; i++) {
         string suffix = string(s_suffix[j])+string(l_suffix[i]);
         string rh_eta_name = string("rh_eta")+suffix;
         string rh_eta_title = rh_eta_name+"; tracks |#eta|; # of tracks";
         rh_eta[i][j] = ibooker.book1D( rh_eta_name.c_str(), rh_eta_title.c_str(), 140, minEta_, maxEta_) ;

         string rh_sh_eta_name = string("rh_sh_eta")+suffix;
         string rh_sh_eta_title = rh_sh_eta_name+"; tracks |#eta|; # of tracks";
         rh_sh_eta[i][j] = ibooker.book1D( rh_sh_eta_name.c_str(), rh_sh_eta_title.c_str(), 140, minEta_, maxEta_) ;

         for ( unsigned int k = 0 ; k<3 ; k++) {
          suffix = string(s_suffix[j])+string(l_suffix[i])+ string(c_suffix[k]);
          string rh_phi_name = string("rh_phi")+suffix;
          string rh_phi_title = rh_phi_name+"; tracks #phi; # of tracks";
          rh_phi[i][j][k] = ibooker.book1D( (rh_phi_name).c_str(), rh_phi_title.c_str(), 200, -PI,PI) ;

          string rh_sh_phi_name = string("rh_sh_phi")+suffix;
          string rh_sh_phi_title = rh_sh_phi_name+"; tracks #phi; # of tracks";
          rh_sh_phi[i][j][k] = ibooker.book1D( (rh_sh_phi_name).c_str(), rh_sh_phi_title.c_str(), 200,-PI,PI) ;

         }
      }
   }
}

GEMRecHitTrackMatch::~GEMRecHitTrackMatch() {  }

void GEMRecHitTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry& geom = *hGeom;

  edm::Handle<edm::PSimHitContainer> simhits;
  edm::Handle<edm::SimTrackContainer> sim_tracks;
  //edm::Handle<edm::SimVertexContainer> sim_vertices;
  iEvent.getByToken(simHitsToken_, simhits);
  iEvent.getByToken(simTracksToken_, sim_tracks);
  //iEvent.getByToken(simVerticesToken_, sim_vertices);
  //if ( !simhits.isValid() || !sim_tracks.isValid() || !sim_vertices.isValid()) return;
  if ( !simhits.isValid() || !sim_tracks.isValid() ) return;

  //const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer & sim_trks = *sim_tracks.product();

  MySimTrack track_; 
  for (auto& t: sim_trks)
  {
    if (!isSimTrackGood(t)) 
    { continue; } 
    
    // match hits and rechits to this SimTrack

    const SimHitMatcher& match_sh = SimHitMatcher( t, iEvent, geom, cfg_, simHitsToken_, simTracksToken_, simVerticesToken_);
    const GEMRecHitMatcher& match_rh = GEMRecHitMatcher( match_sh, iEvent, geom, cfg_ ,gem_recHitToken_);

    track_.pt = t.momentum().pt();
    //std::cout << "SimTrack pt value :  " << track_.pt << "\n"; 
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    std::fill( std::begin(track_.hitOdd), std::end(track_.hitOdd),false);
    std::fill( std::begin(track_.hitEven), std::end(track_.hitEven),false);

    for ( int i= 0 ; i< 3 ; i++) {
      for ( int j= 0 ; j<2 ; j++) {
        track_.gem_sh[i][j]  = false;
        track_.gem_rh[i][j]  = false;
      }
    }

    // ** GEM SimHits ** //
    const auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      const GEMDetId id(d);
      if ( id.chamber() %2 ==0 ) track_.hitEven[id.station()-1] = true;
      else if ( id.chamber() %2 ==1 ) track_.hitOdd[id.station()-1] = true;
      else { std::cout<<"Error to get chamber id"<<std::endl;}

      track_.gem_sh[ id.station()-1][ (id.layer()-1)] = true;

    }
    // ** GEM RecHits ** //
    const auto gem_rh_ids_ch = match_rh.chamberIds();

    for(auto d: gem_rh_ids_ch)
    {
      const GEMDetId id(d);
      track_.gem_rh[ id.station()-1][ (id.layer()-1)] = true;
    }
    
    FillWithTrigger( track_eta, fabs(track_.eta)) ;
    FillWithTrigger( track_phi, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);


    FillWithTrigger( rh_sh_eta, track_.gem_sh  , fabs( track_.eta) );
    FillWithTrigger( rh_eta,    track_.gem_rh  , fabs( track_.eta) );
  
    // Separate station.

    FillWithTrigger( rh_sh_phi, track_.gem_sh  ,fabs(track_.eta), track_.phi , track_.hitOdd, track_.hitEven);
    FillWithTrigger( rh_phi,    track_.gem_rh  ,fabs(track_.eta), track_.phi , track_.hitOdd, track_.hitEven);
  
   
    /*  

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
*/

  }
}
