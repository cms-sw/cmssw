#include "Validation/MuonGEMHits/interface/GEMSimTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>

using namespace std;
GEMSimTrackMatch::GEMSimTrackMatch(DQMStore* dbe, std::string simInputLabel , edm::ParameterSet cfg) : GEMTrackMatch(dbe, simInputLabel , cfg)
{
   minPt_  = cfg_.getUntrackedParameter<double>("gemMinPt",5.0);
   minEta_ = cfg_.getUntrackedParameter<double>("gemMinEta",1.55);
   maxEta_ = cfg_.getUntrackedParameter<double>("gemMaxEta",2.45);
}

void GEMSimTrackMatch::bookHisto() 
{
   const float PI=TMath::Pi();
   dbe_->setCurrentFolder("MuonGEMHitsV/GEMHitsTask");

	 const char* l_suffix[5] = { "","_l1","_l2","_l1or2","_l1and2"};
	 const char* c_suffix[2] = { "_even","_odd" };
   
   for( unsigned int i=0 ; i< 5; i++) {
		for( unsigned int j=0 ; j<2 ; j++) {
			string suffix = string(c_suffix[j]) + string(l_suffix[i]);

			string track_eta_name = string("track_eta")+ suffix;
			string track_eta_title = track_eta_name+";SimTrack |#eta|;# of tracks";
			track_eta[i][j]     = dbe_->book1D(track_eta_name.c_str(), track_eta_title.c_str(), 140,1.5,2.5);
	
			string track_phi_name = string("track_phi")+ suffix;
			string track_phi_title = track_phi_name+";SimTrack |#phi|;# of tracks";
			track_phi[i][j] = dbe_->book1D(track_phi_name.c_str(), track_phi_title.c_str(),100, -PI,PI);
			
			string gem_lx_even_name  = string("gem_lx")+ suffix;
			string gem_lx_even_title = gem_lx_even_name+";SimTrack localX [cm] ; Entries";
			gem_lx[i][j] = dbe_->book1D(gem_lx_even_name.c_str(), gem_lx_even_title.c_str(), 100,-100,100);

			string gem_ly_even_name  = string("gem_ly")+ suffix;
			string gem_ly_even_title = gem_ly_even_name+";SimTrack localY [cm] ; Entries";
			gem_ly[i][j] = dbe_->book1D(gem_ly_even_name.c_str(), gem_ly_even_title.c_str(), 100,-100,100);

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
    track_.has_gem_sh_l1 = 0;
    track_.has_gem_sh_l2 = 0;


    // check for hit chambers
    const auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
		bool isHitLayer1 = false;
		bool isHitLayer2 = false;
		bool isHitOddChamber  = false;
		bool isHitEvenChamber = false;


    for(auto d: gem_sh_ids_ch)
    {
      const GEMDetId id(d);
      if ( id.chamber() & 1 ) isHitOddChamber = true;  // true : odd, false : even
			else isHitEvenChamber = true;
			if ( id.layer() == 1 ) isHitLayer1 = true;
			if ( id.layer() == 2 ) isHitLayer2 = true;
		}		 
		if ( isHitEvenChamber ) {	
	    track_eta[0][0]->Fill( fabs( track_.eta)  );
    	if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[0][0]->Fill( track_.phi);
		  if ( isHitLayer1 ) {
		    track_eta[1][0]->Fill ( fabs(track_.eta));
    		if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[1][0]->Fill( track_.phi);
		  }
		  if ( isHitLayer2 ) {
	     track_eta[2][0]->Fill( fabs(track_.eta));
    		if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[2][0]->Fill( track_.phi);
	  	}
	    if ( isHitLayer1 || isHitLayer2 ) {
	 	    track_eta[3][0]->Fill( fabs(track_.eta));
    		if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[3][0]->Fill( track_.phi);
	    }
	    if ( isHitLayer1 && isHitLayer2 ) {
	      track_eta[4][0]->Fill( fabs(track_.eta));
    		if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[4][0]->Fill( track_.phi);
	    }
		}
		if ( isHitOddChamber ) {	
	    track_eta[0][1]->Fill( fabs( track_.eta)  );
    	if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[0][1]->Fill( track_.phi);
		  if ( isHitLayer1 ) {
		    track_eta[1][1]->Fill ( fabs(track_.eta));
    		if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[1][1]->Fill( track_.phi);
		  }
		  if ( isHitLayer2 ) {
	      track_eta[2][1]->Fill( fabs(track_.eta));
    		if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[2][1]->Fill( track_.phi);
	  	}
	    if ( isHitLayer1 || isHitLayer2 ) {
	 	    track_eta[3][1]->Fill( fabs(track_.eta));
    		if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[3][1]->Fill( track_.phi);
	    }
	    if ( isHitLayer1 && isHitLayer2 ) {
	      track_eta[4][1]->Fill( fabs(track_.eta));
    		if( fabs(track_.eta) < maxEta_ && fabs( track_.eta) > minEta_ ) track_phi[4][1]->Fill( track_.phi);
	    }
		}
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

		// Re-use flags
    isHitLayer1 = false;
    isHitLayer2 = false;
    bool isHitOdd = false;
    bool isHitEven = false;

    if(gem_sh_ids_ch.count(id_ch_even_L1)!=0) { isHitLayer1 = true; isHitEven = true; }
    if(gem_sh_ids_ch.count(id_ch_odd_L1)!=0) { isHitLayer1 = true;  isHitOdd  = true; }
    if(gem_sh_ids_ch.count(id_ch_even_L2)!=0) { isHitLayer2 = true; isHitEven = true; }
    if(gem_sh_ids_ch.count(id_ch_odd_L2)!=0) { isHitLayer2 = true;  isHitOdd  = true; }

    bool lx__odd  = TMath::Abs( TMath::ASin( track_.gem_lx_odd/track_.gem_trk_rho)) < 5*TMath::Pi()/180.;  
    bool lx__even = TMath::Abs( TMath::ASin( track_.gem_lx_even/track_.gem_trk_rho)) < 5*TMath::Pi()/180.; 

    gem_lx[0][0]->Fill( track_.gem_lx_even); // Full even
    gem_lx[0][1]->Fill( track_.gem_lx_odd);  // Full odd
		if( lx__even) gem_ly[0][0]->Fill( track_.gem_ly_even);
		if( lx__even) gem_ly[0][1]->Fill( track_.gem_ly_odd);

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
