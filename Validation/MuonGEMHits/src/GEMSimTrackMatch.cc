#include "Validation/MuonGEMHits/interface/GEMSimTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>

using namespace std;
GEMSimTrackMatch::GEMSimTrackMatch(DQMStore* dbe, edm::EDGetToken& track, edm::EDGetToken& vertex, edm::ParameterSet cfg) : GEMTrackMatch(dbe,track, vertex,  cfg)
{
   minPt_  = cfg_.getUntrackedParameter<double>("gemMinPt",5.0);
   minEta_ = cfg_.getUntrackedParameter<double>("gemMinEta",1.55);
   maxEta_ = cfg_.getUntrackedParameter<double>("gemMaxEta",2.45);
}

void GEMSimTrackMatch::bookHisto(const GEMGeometry* geom) 
{
  const float PI=TMath::Pi();
  dbe_->setCurrentFolder("MuonGEMHitsV/GEMHitsTask");

	const char* l_suffix[4] = { "_l1","_l2","_l1or2","_l1and2"};
	const char* s_suffix[3] = { "_st1","_st2_short","_st2_long"};
	const char* c_suffix[2] = { "_even","_odd" };
	nstation = geom->regions()[0]->stations().size();
   
	for( unsigned int st=0 ; st<nstation ; st++) {
			string suffix = string(s_suffix[st]);
			string track_eta_name = string("track_eta")+ suffix;
			string track_eta_title = track_eta_name+";SimTrack |#eta|;# of tracks";
			track_eta[st]     = dbe_->book1D(track_eta_name.c_str(), track_eta_title.c_str(), 140,minEta_,maxEta_);
			string track_phi_name = string("track_phi")+ suffix;
			string track_phi_title = track_phi_name+";SimTrack #phi;# of tracks";
			track_phi[st] = dbe_->book1D(track_phi_name.c_str(), track_phi_title.c_str(),100, -PI,PI);

   	for( unsigned int layer=0 ; layer< 4; layer++) {
			suffix = string(l_suffix[layer])+s_suffix[st];
		
			string sh_eta_name = string("sh_eta")+suffix;
			string sh_eta_title = sh_eta_name+"; tracks |#eta|; # of tracks";
			sh_eta[layer][st] = dbe_->book1D( sh_eta_name.c_str(), sh_eta_title.c_str(), 140, minEta_, maxEta_) ;

			string sh_phi_name = string("sh_phi")+suffix;
			string sh_phi_title = sh_phi_name+"; tracks #phi; # of tracks";
			sh_phi[layer][st] = dbe_->book1D( sh_phi_name.c_str(), sh_phi_title.c_str(), 100, -PI,PI) ;
		}
		for ( int chamber = 0 ; chamber <2 ; chamber++) {
			suffix =string(s_suffix[st])+c_suffix[chamber];
			string gem_lx_name  = string("gem_lx")+ suffix;
			string gem_lx_title = gem_lx_name+";SimTrack localX [cm] ; Entries";
			gem_lx[st][chamber] = dbe_->book1D(gem_lx_name.c_str(), gem_lx_title.c_str(), 100,-100,100);

			string gem_ly_name  = string("gem_ly")+ suffix;
			string gem_ly_title = gem_ly_name+";SimTrack localY [cm] ; Entries";
			gem_ly[st][chamber] = dbe_->book1D(gem_ly_name.c_str(), gem_ly_title.c_str(), 100,-100,100);
		}
	}
}





GEMSimTrackMatch::~GEMSimTrackMatch() {
}

void GEMSimTrackMatch::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  struct MySimTrack
  {
    Float_t pt, eta, phi;
    Char_t endcap;
    Float_t gem_sh_eta, gem_sh_phi;
    Float_t gem_trk_eta, gem_trk_phi, gem_trk_rho;
    Float_t gem_lx_even, gem_ly_even;
    Float_t gem_lx_odd, gem_ly_odd;
    Char_t has_gem_sh_l1, has_gem_sh_l2;
		bool track_sh[3][2][2];
		bool station[3];
		bool layer[2];
		bool chamber[2];
  };
  MySimTrack track_;

  iEvent.getByToken(simTracksToken_, sim_tracks);
  iEvent.getByToken(simVerticesToken_, sim_vertices);
  
  if ( !sim_tracks.isValid() || !sim_vertices.isValid()) return;

  const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer & sim_trks = *sim_tracks.product();



  for (auto& t: sim_trks)
  {
    if (!isSimTrackGood(t)) 
    { continue; } 
    
    // match hits to this SimTrack
    SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup, theGEMGeometry);
    const SimHitMatcher& match_sh = match.simhits();

    track_.pt = t.momentum().pt();
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    track_.gem_sh_eta = -9.;
    track_.gem_sh_phi = -9.;
    track_.gem_trk_eta = -999.;
    track_.gem_trk_phi = -999.;
    track_.gem_trk_rho = -999.;
    track_.gem_lx_even =0;
    track_.gem_ly_even =0;
    track_.gem_lx_odd  =0;
    track_.gem_ly_odd  =0;
		
		for( int i=0 ; i<3 ; i++) track_.station[i] = false;
		for( int i=0 ; i<2 ; i++) track_.layer[i]   = false;
		for( int i=0 ; i<2 ; i++) track_.chamber[i] = false; 

    // check for hit chambers
    const auto gem_sh_ids_ch = match_sh.chamberIdsGEM();

    for(auto d: gem_sh_ids_ch)
    {
      const GEMDetId id(d);
			//std::cout<<d<<" to "<<id<<std::endl;
			track_.station[id.station()-1] = true;
			track_.layer[ id.layer()-1] = true;
			track_.chamber[ id.chamber()%2] = true;
			auto& gem_sh_list = match_sh.hitsInChamber(id);
			for( auto sh: gem_sh_list) {
				gem_lx[id.station()-1][id.chamber()%2]->Fill( sh.localPosition().x());
				gem_ly[id.station()-1][id.chamber()%2]->Fill( sh.localPosition().y());
			}
			
		}
		track_eta[0]->Fill( fabs( track_.eta)) ;
		if ( fabs(track_.eta) > getEtaRangeForPhi(0).first && fabs(track_.eta)< getEtaRangeForPhi(0).second   ) track_phi[0]->Fill( track_.phi ) ;

		if ( nstation >1 ) { 
			track_eta[1]->Fill ( fabs( track_.eta)) ;   // station2_short
			track_eta[2]->Fill ( fabs( track_.eta)) ;   // station2_long
			if ( fabs(track_.eta) > getEtaRangeForPhi(1).first && fabs(track_.eta)< getEtaRangeForPhi(1).second   ) track_phi[1]->Fill( track_.phi ) ;
			if ( fabs(track_.eta) > getEtaRangeForPhi(2).first && fabs(track_.eta)< getEtaRangeForPhi(2).second   ) track_phi[2]->Fill( track_.phi ) ;
		}
		
		for( unsigned int station =0 ; station<nstation; station++) {
			if ( track_.station[station] ) {
				if ( track_.layer[0] ) sh_eta[0][station]->Fill( track_.eta);
				if ( track_.layer[1] ) sh_eta[1][station]->Fill( track_.eta);
				if ( track_.layer[0] || track_.layer[1] ) sh_eta[2][station]->Fill( track_.eta);
				if ( track_.layer[0] && track_.layer[1] ) sh_eta[3][station]->Fill( track_.eta);
			}
			if( track_.station[station] && fabs(track_.eta) > getEtaRangeForPhi(station).first && fabs(track_.eta)< getEtaRangeForPhi(station).second ) {
        if ( track_.layer[0] ) sh_phi[0][station]->Fill( track_.phi);
        if ( track_.layer[1] ) sh_phi[1][station]->Fill( track_.phi);
        if ( track_.layer[0] || track_.layer[1] ) sh_phi[2][station]->Fill( track_.phi);
        if ( track_.layer[0] && track_.layer[1] ) sh_phi[3][station]->Fill( track_.phi);
			}
		}
	}
}

		/*

		for( int station =0 ; station<3; station++) { 
			for( int layer = 0 ; layer <2; layer++) {
				for( int chamber = 0 ; chamber<2; chamber++) {
					if ( track_.station[i] ) {
				if ( track_.

    // Calculation of the localXY efficiency
    GlobalPoint gp_track(match_sh.propagatedPositionGEM());
    track_.gem_trk_eta = gp_track.eta();
    track_.gem_trk_rho = gp_track.perp();
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

    bool isHitLayer1 = false;
    bool isHitLayer2 = false;
    bool isHitOdd = false;
    bool isHitEven = false;

    if(gem_sh_ids_ch.count(id_ch_even_L1)!=0) { isHitLayer1 = true; isHitEven = true; }
    if(gem_sh_ids_ch.count(id_ch_odd_L1)!=0) { isHitLayer1 = true;  isHitOdd  = true; }
    if(gem_sh_ids_ch.count(id_ch_even_L2)!=0) { isHitLayer2 = true; isHitEven = true; }
    if(gem_sh_ids_ch.count(id_ch_odd_L2)!=0) { isHitLayer2 = true;  isHitOdd  = true; }

    bool isOddClosed  = TMath::Abs( TMath::ASin( track_.gem_lx_odd/track_.gem_trk_rho))  < 5*TMath::DegToRad();  
    bool isEvenClosed = TMath::Abs( TMath::ASin( track_.gem_lx_even/track_.gem_trk_rho)) < 5*TMath::DegToRad(); 


    gem_lx[0][0]->Fill( track_.gem_lx_even); 
    gem_lx[0][1]->Fill( track_.gem_lx_odd);  
		if( isEvenClosed) gem_ly[0][0]->Fill( track_.gem_ly_even);
		if( lsOddClosed ) gem_ly[0][1]->Fill( track_.gem_ly_odd);

		if( isHitEven) { 
	    if ( isHitLayer1 ) {
	      gem_lx[1][0]->Fill ( track_.gem_lx_even);
				if( lx__even) gem_ly[1][0]->Fill ( track_.gem_ly_even);
	    }
    	if ( isHitLayer2 ) {
	      gem_lx[2][0]->Fill ( track_.gem_lx_even);
				if( lx__even) gem_ly[2][0]->Fill ( track_.gem_ly_even);
	    }
    	if ( isHitLayer1 || isHitLayer2 ) {
	      gem_lx[3][0]->Fill ( track_.gem_lx_even);
				if( lx__even) gem_ly[3][0]->Fill ( track_.gem_ly_even);
	    }
			if (isHitLayer1 && isHitLayer2 ) {
	      gem_lx[4][0]->Fill ( track_.gem_lx_even);
				if( lx__even) gem_ly[4][0]->Fill ( track_.gem_ly_even);
			}
		}	
		if( isHitOdd) { 
	    if ( isHitLayer1 ) {
	      gem_lx[1][1]->Fill ( track_.gem_lx_odd);
				if( lx__odd) gem_ly[1][1]->Fill ( track_.gem_ly_odd);
	    }
    	if ( isHitLayer2 ) {
	      gem_lx[2][1]->Fill ( track_.gem_lx_odd);
				if( lx__odd) gem_ly[2][1]->Fill ( track_.gem_ly_odd);
	    }
    	if ( isHitLayer1 || isHitLayer2 ) {
	      gem_lx[3][1]->Fill ( track_.gem_lx_odd);
				if( lx__odd) gem_ly[3][1]->Fill ( track_.gem_ly_odd);
	    }
			if (isHitLayer1 && isHitLayer2 ) {
	      gem_lx[4][1]->Fill ( track_.gem_lx_odd);
				if( lx__odd) gem_ly[4][1]->Fill ( track_.gem_ly_odd);
			}
		}
  }
  }
  
}
*/
