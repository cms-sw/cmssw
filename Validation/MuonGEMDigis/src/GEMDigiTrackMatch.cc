#include "Validation/MuonGEMDigis/interface/GEMDigiTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>
#include "Validation/MuonGEMHits/interface/GEMDetLabel.h"

using namespace std;
using namespace GEMDetLabel;
GEMDigiTrackMatch::GEMDigiTrackMatch(const edm::ParameterSet& ps) : GEMTrackMatch(ps)
{
  std::string simInputLabel_ = ps.getUntrackedParameter<std::string>("simInputLabel");
  simHitsToken_ = consumes<edm::PSimHitContainer>(edm::InputTag(simInputLabel_,"MuonGEMHits"));
  simTracksToken_ = consumes< edm::SimTrackContainer >(ps.getParameter<edm::InputTag>("simTrackCollection"));
  simVerticesToken_ = consumes< edm::SimVertexContainer >(ps.getParameter<edm::InputTag>("simVertexCollection"));

  gem_digiToken_ = consumes<GEMDigiCollection>(ps.getParameter<edm::InputTag>("gemDigiInput"));
  gem_padToken_  = consumes<GEMPadDigiCollection>(ps.getParameter<edm::InputTag>("gemPadDigiInput"));
  gem_copadToken_ = consumes<GEMCoPadDigiCollection>(ps.getParameter<edm::InputTag>("gemCoPadDigiInput")); 
  detailPlot_ = ps.getParameter<bool>("detailPlot");
  cfg_ = ps;
}

void GEMDigiTrackMatch::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& run, edm::EventSetup const & iSetup){
  const GEMGeometry* geom = nullptr;
  try {
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  geom = &*hGeom;
  setGeometry(*geom);
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("GEMDigiTrackMatch") << "+++ Error : GEM geometry is unavailable on histogram booking. +++\n";
    return;
  }

  const float PI=TMath::Pi();

  nstation = geom->regions()[0]->stations().size();
  for( auto& region : geom->regions() ){
    int region_num = region->region();
    TString title_suffix = TString::Format(" at Region%d", region_num);
    TString histname_suffix = TString::Format("_r%d", region_num);
    for( auto& station : region->stations() ){
      if( station->station()==2 ) continue;
      int station_num = (station->station()==1) ? 1 : 2;
      TString title_suffix2 = title_suffix + TString::Format(" Station%d", station_num);
      TString histname_suffix2 = histname_suffix + TString::Format("_st%d", station_num);

      TString dcEta_title_sim = TString::Format("SimHit Occupancy for detector component %s;;#eta-partition",title_suffix2.Data());
      TString dcEta_histname_sim = TString::Format("sim_dcEta_trk%s", histname_suffix2.Data());
      TString dcEta_title_strip = TString::Format("Occupancy for detector component %s;;#eta-partition",title_suffix2.Data());
      TString dcEta_histname_strip = TString::Format("strip_dcEta_trk%s", histname_suffix2.Data());
      TString dcEta_title_pad = TString::Format("Pad's occupancy for detector component %s;;#eta-partition", title_suffix2.Data());
      TString dcEta_histname_pad = TString::Format("pad_dcEta_trk%s", histname_suffix2.Data());
      TString dcEta_title_copad = TString::Format("CoPad's occupancy for detector component %s;;#eta-partition", title_suffix2.Data());
      TString dcEta_histname_copad = TString::Format("copad_dcEta_trk%s", histname_suffix2.Data());

      int nXbins = station->rings()[0]->nSuperChambers()*2;
      int nRoll1 = station->rings()[0]->superChambers()[0]->chambers()[0]->etaPartitions().size();
      int nRoll2 = station->rings()[0]->superChambers()[0]->chambers()[1]->etaPartitions().size();
      int nYbins = ( nRoll1 > nRoll2 ) ? nRoll1 : nRoll2;

      theSim_dcEta[ dcEta_histname_sim.Hash() ] = ibooker.book2D(dcEta_histname_sim, dcEta_title_strip, nXbins, 0, nXbins, nYbins, 1, nYbins+1);
      theStrip_dcEta[ dcEta_histname_strip.Hash() ] = ibooker.book2D(dcEta_histname_strip, dcEta_title_strip, nXbins, 0, nXbins, nYbins, 1, nYbins+1);
      thePad_dcEta[ dcEta_histname_pad.Hash() ] = ibooker.book2D(dcEta_histname_pad, dcEta_title_pad, nXbins, 0, nXbins, nYbins, 1, nYbins+1);
      theCoPad_dcEta[ dcEta_histname_copad.Hash() ] = ibooker.book2D(dcEta_histname_copad, dcEta_title_copad, nXbins, 0, nXbins, nYbins, 1, nYbins+1);
      int idx = 0;
      for(unsigned int sCh=1; sCh <= station->superChambers().size(); sCh++ ){
        for(unsigned int Ch=1; Ch<=2; Ch++){
          idx++;
          TString label = TString::Format("ch%d_la%d", sCh, Ch);
          theSim_dcEta[ dcEta_histname_sim.Hash() ]->setBinLabel(idx, label.Data());
          theStrip_dcEta[ dcEta_histname_strip.Hash() ]->setBinLabel(idx, label.Data());
          thePad_dcEta[ dcEta_histname_pad.Hash() ]->setBinLabel(idx, label.Data());
          theCoPad_dcEta[ dcEta_histname_copad.Hash() ]->setBinLabel(idx, label.Data());
        }
      }
    }
  }
  if ( detailPlot_) { 
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
        string dg_eta_name = string("dg_eta")+suffix;
        string dg_eta_title = dg_eta_name+"; tracks |#eta|; # of tracks";
        dg_eta[i][j] = ibooker.book1D( dg_eta_name.c_str(), dg_eta_title.c_str(), 140, minEta_, maxEta_) ;

        string dg_sh_eta_name = string("dg_sh_eta")+suffix;
        string dg_sh_eta_title = dg_sh_eta_name+"; tracks |#eta|; # of tracks";
        dg_sh_eta[i][j] = ibooker.book1D( dg_sh_eta_name.c_str(), dg_sh_eta_title.c_str(), 140, minEta_, maxEta_) ;

        string pad_eta_name = string("pad_eta")+suffix;
        string pad_eta_title = pad_eta_name+"; tracks |#eta|; # of tracks";
        pad_eta[i][j] = ibooker.book1D( pad_eta_name.c_str(), pad_eta_title.c_str(), 140, minEta_, maxEta_) ;

        string copad_eta_name = string("copad_eta")+suffix;
        string copad_eta_title = copad_eta_name+"; tracks |#eta|; # of tracks";
        copad_eta[i][j] = ibooker.book1D( copad_eta_name.c_str(), copad_eta_title.c_str(), 140, minEta_, maxEta_) ;

        for ( unsigned int k = 0 ; k<3 ; k++) {
          suffix = string(s_suffix[j])+string(l_suffix[i])+ string(c_suffix[k]);
          string dg_phi_name = string("dg_phi")+suffix;
          string dg_phi_title = dg_phi_name+"; tracks #phi; # of tracks";
          dg_phi[i][j][k] = ibooker.book1D( (dg_phi_name).c_str(), dg_phi_title.c_str(), 200, -PI,PI) ;

          string dg_sh_phi_name = string("dg_sh_phi")+suffix;
          string dg_sh_phi_title = dg_sh_phi_name+"; tracks #phi; # of tracks";
          dg_sh_phi[i][j][k] = ibooker.book1D( (dg_sh_phi_name).c_str(), dg_sh_phi_title.c_str(), 200,-PI,PI) ;

          string pad_phi_name = string("pad_phi")+suffix;
          string pad_phi_title = pad_phi_name+"; tracks #phi; # of tracks";
          pad_phi[i][j][k] = ibooker.book1D( (pad_phi_name).c_str(), pad_phi_title.c_str(), 200, -PI,PI) ;

          string copad_phi_name = string("copad_phi")+suffix;
          string copad_phi_title = copad_phi_name+"; tracks #phi; # of tracks";
          copad_phi[i][j][k] = ibooker.book1D( (copad_phi_name).c_str(), copad_phi_title.c_str(), 200, -PI,PI) ;

        }
      }
    }
  }
}

GEMDigiTrackMatch::~GEMDigiTrackMatch() {  }


void GEMDigiTrackMatch::fillMatchedHitID( const char* hTitle_prefix, std::unordered_map< UInt_t , MonitorElement* >& hist_map, const GEMDetId id){
      int binX = (id.chamber()-1)*2 + (id.layer()-1);
      int binY = id.roll();
      Int_t station = id.station();
      bool copad = true ; 
      if ( std::string(hTitle_prefix).find("copad") == string::npos ) {  
        copad = false;
      }
      if ( 2== station ) return; // remove st2_short hits.
      if ( 3== station ) station=2 ; // Just to labeling.
      TString histname = TString::Format("%s_r%d_st%d", hTitle_prefix, id.region(), station);
      LogDebug("GEMDigiTrackMatch")<<histname;
      hist_map[histname.Hash()]->Fill(binX, binY);
      if ( copad ) hist_map[histname.Hash()]->Fill(binX+1, binY); 
}

void GEMDigiTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry& geom = *hGeom;

  edm::Handle<edm::PSimHitContainer> simhits;
  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;
  iEvent.getByToken(simHitsToken_, simhits);
  iEvent.getByToken(simTracksToken_, sim_tracks);
  iEvent.getByToken(simVerticesToken_, sim_vertices);
  if ( !simhits.isValid() || !sim_tracks.isValid() || !sim_vertices.isValid()) return;

  // Simple Plot //
    //const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
    const edm::SimTrackContainer & sim_trks = *sim_tracks.product();
    MySimTrack track_; 
    for (auto& t: sim_trks)
    {
      if (!isSimTrackGood(t)) 
      { continue; } 

      // match hits and digis to this SimTrack

      const SimHitMatcher& match_sh = SimHitMatcher( t, iEvent, geom, cfg_, simHitsToken_, simTracksToken_, simVerticesToken_);
      const GEMDigiMatcher& match_gd = GEMDigiMatcher( match_sh, iEvent, geom, cfg_ ,gem_digiToken_, gem_padToken_, gem_copadToken_);

      track_.pt = t.momentum().pt();
      track_.phi = t.momentum().phi();
      track_.eta = t.momentum().eta();
      std::fill( std::begin(track_.hitOdd), std::end(track_.hitOdd),false);
      std::fill( std::begin(track_.hitEven), std::end(track_.hitEven),false);

      for ( int i= 0 ; i< 3 ; i++) {
        for ( int j= 0 ; j<2 ; j++) {
          track_.gem_sh[i][j]  = false;
          track_.gem_dg[i][j]  = false;
          track_.gem_pad[i][j] = false;
        }
      }
 
      // Simple plots for efficiency
      const auto gem_sh_ids = match_sh.detIdsGEM(); 
      const auto gem_strip_ids = match_gd.detIds(); 
      const auto gem_pad_ids = match_gd.detIdsForPad(); 
      const auto gem_copad_ids = match_gd.detIdsForCoPad(); 

      for(auto d: gem_sh_ids )    { fillMatchedHitID( "sim_dcEta_trk"  , theSim_dcEta   , GEMDetId(d)); }
      for(auto d: gem_strip_ids ) { fillMatchedHitID( "strip_dcEta_trk", theStrip_dcEta , GEMDetId(d)); }
      for(auto d: gem_pad_ids )   { fillMatchedHitID( "pad_dcEta_trk",   thePad_dcEta   , GEMDetId(d)); }
      for(auto d: gem_copad_ids ) { fillMatchedHitID( "copad_dcEta_trk", theCoPad_dcEta , GEMDetId(d)); }

      // Detail plots 
      if ( detailPlot_ ) {
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
      // ** GEM Digis, Pads and CoPads ** //
      const auto gem_dg_ids_ch = match_gd.chamberIds();

      for(auto d: gem_dg_ids_ch)
      {
        const GEMDetId id(d);
        track_.gem_dg[ id.station()-1][ (id.layer()-1)] = true;
      }
      const auto gem_pad_ids_ch = match_gd.chamberIdsWithPads();
      for(auto d: gem_pad_ids_ch)
      {
        const GEMDetId id(d);
        track_.gem_pad[ id.station()-1][ (id.layer()-1)] = true;
      }


      //FillWithTrigger( track_eta, fabs(track_.eta)) ;
      //FillWithTrigger( track_phi, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);


        FillWithTrigger( dg_sh_eta, track_.gem_sh  , fabs( track_.eta) );
        FillWithTrigger( dg_eta,    track_.gem_dg  , fabs( track_.eta) );
        FillWithTrigger( pad_eta,   track_.gem_pad , fabs( track_.eta) );
        FillWithTrigger( copad_eta,   track_.gem_pad , fabs( track_.eta) );

        // Separate station.

        FillWithTrigger( dg_sh_phi, track_.gem_sh  ,fabs(track_.eta), track_.phi , track_.hitOdd, track_.hitEven);
        FillWithTrigger( dg_phi,    track_.gem_dg  ,fabs(track_.eta), track_.phi , track_.hitOdd, track_.hitEven);
        FillWithTrigger( pad_phi,   track_.gem_pad ,fabs(track_.eta), track_.phi , track_.hitOdd, track_.hitEven);
        FillWithTrigger( copad_phi,   track_.gem_pad ,fabs(track_.eta), track_.phi , track_.hitOdd, track_.hitEven);
    }
  }
}
