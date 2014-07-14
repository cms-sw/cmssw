#include "Validation/MuonGEMDigis/interface/GEMDigiTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>

using namespace std;
GEMDigiTrackMatch::GEMDigiTrackMatch(DQMStore* dbe, edm::EDGetToken& track, edm::EDGetToken& vertex , edm::ParameterSet cfg) : GEMTrackMatch(dbe,track,vertex,cfg)
{
   minPt_  = cfg_.getUntrackedParameter<double>("gemDigiMinPt",5.0);
   minEta_ = cfg_.getUntrackedParameter<double>("gemDigiMinEta",1.55);
   maxEta_ = cfg_.getUntrackedParameter<double>("gemDigiMaxEta",2.18);
}

void GEMDigiTrackMatch::FillWithTrigger( MonitorElement* hist[4][3], bool array[3][2], Float_t value)
{
	for( unsigned int i=0 ; i<nstation ; i++) {
		if ( array[i][0] ) hist[0][i]->Fill(value);
		if ( array[i][1] ) hist[1][i]->Fill(value);
		if ( array[i][0] || array[i][1] ) hist[2][i]->Fill(value);
		if ( array[i][0] && array[i][1] ) hist[3][i]->Fill(value);
	} 
	return;
}


void GEMDigiTrackMatch::bookHisto(const GEMGeometry* geom){
	theGEMGeometry = geom;
  const float PI=TMath::Pi();
	const char* l_suffix[4] = {"_l1","_l2","_l1or2","_l1and2"};
	const char* s_suffix[3] = {"_st1","_st2_short","_st2_long"};
	const char* c_suffix[2] = {"_even","_odd"};

  nstation = theGEMGeometry->regions()[0]->stations().size(); 
	for( unsigned int j=0 ; j<nstation ; j++) {
			string track_eta_name  = string("track_eta")+s_suffix[j];
			string track_eta_title = string("track_eta")+";SimTrack |#eta|;# of tracks";
			string track_phi_name  = string("track_phi")+s_suffix[j];
			string track_phi_title = string("track_phi")+";SimTrack #phi;# of tracks";
	 		track_eta[j] = dbe_->book1D(track_eta_name.c_str(), track_eta_title.c_str(),140,minEta_,maxEta_);
			track_phi[j] = dbe_->book1D(track_phi_name.c_str(), track_phi_title.c_str(),100,-PI,PI);
	 		for( unsigned int i=0 ; i< 4; i++) {
				 string suffix = string(l_suffix[i])+string(s_suffix[j]);
				 string dg_eta_name = string("dg_eta")+suffix;
				 string dg_eta_title = dg_eta_name+"; tracks |#eta|; # of tracks";
			   dg_eta[i][j] = dbe_->book1D( dg_eta_name.c_str(), dg_eta_title.c_str(), 140, minEta_, maxEta_) ;

				 string dg_sh_eta_name = string("dg_sh_eta")+suffix;
				 string dg_sh_eta_title = dg_sh_eta_name+"; tracks |#eta|; # of tracks";
			   dg_sh_eta[i][j] = dbe_->book1D( dg_sh_eta_name.c_str(), dg_sh_eta_title.c_str(), 140, minEta_, maxEta_) ;

				 string dg_phi_name = string("dg_phi")+suffix;
				 string dg_phi_title = dg_phi_name+"; tracks #phi; # of tracks";
			   dg_phi[i][j] = dbe_->book1D( dg_phi_name.c_str(), dg_phi_title.c_str(), 100, -PI,PI) ;

				 string dg_sh_phi_name = string("dg_sh_phi")+suffix;
				 string dg_sh_phi_title = dg_sh_phi_name+"; tracks #phi; # of tracks";
			   dg_sh_phi[i][j] = dbe_->book1D( dg_sh_phi_name.c_str(), dg_sh_phi_title.c_str(), 100,-PI,PI) ;

				 string pad_eta_name = string("pad_eta")+suffix;
				 string pad_eta_title = pad_eta_name+"; tracks |#eta|; # of tracks";
			   pad_eta[i][j] = dbe_->book1D( pad_eta_name.c_str(), pad_eta_title.c_str(), 140, minEta_, maxEta_) ;

				 string pad_phi_name = string("pad_phi")+suffix;
				 string pad_phi_title = pad_phi_name+"; tracks #phi; # of tracks";
			   pad_phi[i][j] = dbe_->book1D( pad_phi_name.c_str(), pad_phi_title.c_str(), 100, -PI,PI) ;
				 for ( unsigned int k = 0 ; k<2 ; k++) {
					 suffix = suffix+ string(c_suffix[k]);
					 string dg_lx_name = string("dg_lx")+suffix;
					 string dg_lx_title = dg_lx_name+"; local X[cm]; Entries";
					 //dg_lx[i][j][k] = dbe_->book1D( dg_lx_name.c_str(), dg_lx_title.c_str(), 100,-100,100);  

					 string dg_ly_name = string("dg_ly")+suffix;
					 string dg_ly_title = dg_ly_name+"; local Y[cm]; Entries";
					 //dg_ly[i][j][k] = dbe_->book1D( dg_ly_name.c_str(), dg_ly_title.c_str(), 100,-100,100);  
				 }
			}
	 }
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
		bool gem_sh[3][2];
		bool gem_dg[3][2];
		bool gem_pad[3][2];
  };
  MySimTrack track_;

  iEvent.getByToken(simTracksToken_, sim_tracks);
  iEvent.getByToken(simVerticesToken_, sim_vertices);

  if ( !sim_tracks.isValid() || !sim_vertices.isValid()) {
    LogDebug("GEMDigiTrackMath")<<"GEMDigiTrackMatch can not load sim_track or sim vertex by token\n";  
    return;
  }

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
    track_.gem_lx_even =0;
    track_.gem_ly_even =0;
    track_.gem_lx_odd  =0;
    track_.gem_ly_odd  =0;
    track_.has_gem_dg_l1 = 0;
    track_.has_gem_dg_l2 = 0;
    track_.has_gem_pad_l1 = 0;
    track_.has_gem_pad_l2 = 0;
		for ( int i= 0 ; i< 3 ; i++) {
			for ( int j= 0 ; j<2 ; j++) {
		    track_.gem_sh[i][j]  = false;
    		track_.gem_dg[i][j]  = false;
		    track_.gem_pad[i][j] = false;
			}
		}

    // ** GEM SimHits ** //
    const auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      const GEMDetId id(d);
			track_.gem_sh[ id.station()-1][ (id.layer()-1)] = true;

    }
    // ** GEM Digis, Pads and CoPads ** //
    auto gem_dg_ids_ch = match_gd.chamberIds();

    for(auto d: gem_dg_ids_ch)
    {
      GEMDetId id(d);
			track_.gem_dg[ id.station()-1][ (id.layer()-1)] = true;
			track_.gem_pad[ id.station()-1][ (id.layer()-1)] = true;
    }
	  
 
    // if this track enter thought station, 
		track_eta[0]->Fill ( fabs( track_.eta)) ;   // station1
		if ( fabs(track_.eta) > getEtaRangeForPhi(0).first && fabs(track_.eta)< getEtaRangeForPhi(0).second   ) track_phi[0]->Fill( track_.phi ) ;

		if ( nstation >1 ) { 
			track_eta[1]->Fill ( fabs( track_.eta)) ;   // station2_short
			track_eta[2]->Fill ( fabs( track_.eta)) ;   // station2_long
			if ( fabs(track_.eta) > getEtaRangeForPhi(1).first && fabs(track_.eta)< getEtaRangeForPhi(1).second   ) track_phi[1]->Fill( track_.phi ) ;
			if ( fabs(track_.eta) > getEtaRangeForPhi(2).first && fabs(track_.eta)< getEtaRangeForPhi(2).second   ) track_phi[2]->Fill( track_.phi ) ;
		}
		

		FillWithTrigger( dg_sh_eta, track_.gem_sh  , fabs( track_.eta) );
		FillWithTrigger( dg_eta,    track_.gem_dg  , fabs( track_.eta) );
		FillWithTrigger( pad_eta,   track_.gem_pad , fabs( track_.eta) );
	
    // Separate station.

		FillWithTrigger( dg_sh_phi, track_.gem_sh  ,  track_.phi );
		FillWithTrigger( dg_phi,    track_.gem_dg  ,  track_.phi );
		FillWithTrigger( pad_phi,   track_.gem_pad ,  track_.phi );
	
   
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
