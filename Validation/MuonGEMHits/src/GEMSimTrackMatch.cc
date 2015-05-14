#include "Validation/MuonGEMHits/interface/GEMSimTrackMatch.h"
#include "Validation/MuonGEMHits/interface/GEMDetLabel.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>

using namespace std;
using namespace GEMDetLabel;
GEMSimTrackMatch::GEMSimTrackMatch(const edm::ParameterSet& ps) : GEMTrackMatch(ps)
{
  std::string simInputLabel_ = ps.getUntrackedParameter<std::string>("simInputLabel");

  simHitsToken_ = consumes<edm::PSimHitContainer>(edm::InputTag(simInputLabel_,"MuonGEMHits"));
  simTracksToken_ = consumes< edm::SimTrackContainer >(ps.getParameter<edm::InputTag>("simTrackCollection"));
  simVerticesToken_ = consumes< edm::SimVertexContainer >(ps.getParameter<edm::InputTag>("simVertexCollection"));

  cfg_ = ps;
}

void GEMSimTrackMatch::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & run, edm::EventSetup const & iSetup) 
{
  // Mandantory
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry& geom = *hGeom;
  setGeometry(geom);

  const float PI=TMath::Pi();

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
         string suffix = string(s_suffix[j])+l_suffix[i];

         string sh_eta_name = string("sh_eta")+suffix;
         string sh_eta_title = sh_eta_name+"; tracks |#eta|; # of tracks";
         sh_eta[i][j] = ibooker.book1D( sh_eta_name.c_str(), sh_eta_title.c_str(), 140, minEta_, maxEta_) ;

         for ( unsigned int k = 0 ; k<3 ; k++) {
          suffix = string(s_suffix[j])+ string(l_suffix[i])+string(c_suffix[k]);
          string sh_phi_name = string("sh_phi")+suffix;
          string sh_phi_title = sh_phi_name+"; tracks #phi; # of tracks";
          sh_phi[i][j][k] = ibooker.book1D( (sh_phi_name).c_str(), sh_phi_title.c_str(), 200,-PI,PI) ;

         }
      }
  }
}





GEMSimTrackMatch::~GEMSimTrackMatch() {
}

void GEMSimTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iSetup.get<MuonGeometryRecord>().get(hGeom);
  const GEMGeometry& geom = *hGeom;

  MySimTrack track_;
  edm::Handle<edm::PSimHitContainer> simhits;
  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;

  iEvent.getByToken(simHitsToken_, simhits);
  iEvent.getByToken(simTracksToken_, sim_tracks);
  iEvent.getByToken(simVerticesToken_, sim_vertices);
  if ( !simhits.isValid() || !sim_tracks.isValid() || !sim_vertices.isValid()) return;

  //const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer & sim_trks = *sim_tracks.product();



  for (auto& t: sim_trks)
  {
    if (!isSimTrackGood(t)) 
    { continue; } 
    
    track_.pt = t.momentum().pt();
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    std::fill( std::begin(track_.hitOdd), std::end(track_.hitOdd),false);
    std::fill( std::begin(track_.hitEven), std::end(track_.hitEven),false);
    for( int i=0 ; i<3 ; i++) {
      for (int j=0 ; j<2 ; j++) {
        track_.gem_sh[i][j] = false;
      }
    }

    // match hits to this SimTrack
    const SimHitMatcher match_sh = SimHitMatcher( t, iEvent, geom, cfg_, simHitsToken_, simTracksToken_, simVerticesToken_ );

    // check for hit chambers
    const auto gem_sh_ids_ch = match_sh.chamberIdsGEM();

    for(auto d: gem_sh_ids_ch)
    {
      const GEMDetId id(d);
      if ( id.chamber() %2 ==0 ) track_.hitEven[id.station()-1] = true;
      else if ( id.chamber() %2 ==1 ) track_.hitOdd[id.station()-1] = true;
      else { std::cout<<"Error to get chamber id"<<std::endl;}
      track_.gem_sh[ id.station()-1][ (id.layer()-1)] = true;
    }
    FillWithTrigger( track_eta, fabs(track_.eta)) ;
    FillWithTrigger( track_phi, fabs(track_.eta), track_.phi, track_.hitOdd, track_.hitEven);
    FillWithTrigger( sh_eta, track_.gem_sh  , fabs( track_.eta) );
    FillWithTrigger( sh_phi, track_.gem_sh ,fabs(track_.eta), track_.phi , track_.hitOdd, track_.hitEven);
  }
}

