#include "Validation/MuonGEMRecHits/interface/GEMRecHitsValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>

using namespace std;

GEMRecHitsValidation::GEMRecHitsValidation(const edm::ParameterSet& cfg): GEMBaseValidation(cfg)
{
  InputTagToken_   = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagToken_RH = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitsInputLabel"));
}

MonitorElement* GEMRecHitsValidation::BookHist1D( DQMStore::IBooker& ibooker, const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num, const unsigned int Nbin, const Float_t xMin, const Float_t xMax) {                                                                             

  string hist_name  = name+string("_r") + regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];

  string hist_label = label+string(" : region")+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num];  

  return ibooker.book1D( hist_name, hist_label,Nbin,xMin,xMax ); 
}

void GEMRecHitsValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
  const GEMGeometry* GEMGeometry_ ;
  
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("MuonGEMRechHits") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }

  LogDebug("GEMRecHitsValidation")<<"Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("MuonGEMRecHitsV/GEMRecHitsTask");
  LogDebug("GEMRecHitsValidation")<<"ibooker set current folder\n";

  int nregions = GEMGeometry_->regions().size();
  LogDebug("GEMRecHitsValidation")<<"nregions set.\n";
  int nstations = GEMGeometry_->regions()[0]->stations().size(); 
  LogDebug("GEMRecHitsValidation")<<"nstations set.\n";
//  int nstripsGE11  = 384;
//GEMGeometry_->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->nstrips();
//  int nstripsGE21 = 768;
 
  /* 
  if ( nstations > 1 ) {
    nstripsGE21  = GEMGeometry_->regions()[0]->stations()[1]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->nstrips();
  }
  else LogDebug("GEMStripDIGIValidation")<<"Info : Only 1 station is existed.\n";
  */
  LogDebug("GEMRecHitsValidation")<<"Successfully binning set.\n";

  gem_cls_tot = ibooker.book1D("gem_cls_tot","ClusterSize Distribution",11,-0.5,10.5);
  for( int region_num = 0 ; region_num <nregions ; region_num++ ) {
    for( int layer_num = 0 ; layer_num < 2 ; layer_num++) {
      for( int station_num = 0 ; station_num < nstations ; station_num++) {
  //      if ( station_num == 0 ) nstrips = nstripsGE11;
  //      else nstrips = nstripsGE21;
  //      std::string name_prefix = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num] + "_l"+layerLabel[layer_num];
  //      std::string label_prefix = "region"+regionLabel[region_num]+" station "+stationLabel[station_num] +" layer "+layerLabel[layer_num];
       /* theStrip_phistrip[region_num][station_num][layer_num] = ibooker.book2D( ("strip_dg_phistrip"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+"; phi [rad];strip number").c_str(), 280, -TMath::Pi(), TMath::Pi(), nstrips/2,0,nstrips);
        theStrip[region_num][station_num][layer_num] = ibooker.book1D( ("strip_dg"+name_prefix).c_str(), ("Digi occupancy per stip number: "+label_prefix+";strip number; entries").c_str(), nstrips,0.5,nstrips+0.5);
        theStrip_bx[region_num][station_num][layer_num] = ibooker.book1D( ("strip_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5); */
        gem_cls[region_num][station_num][layer_num] = BookHist1D(ibooker,"cls","ClusterSize Distribution",region_num,station_num,layer_num,11,-0.5,10.5);
        gem_pullX[region_num][station_num][layer_num] = BookHist1D(ibooker,"pullX","Pull Of X",region_num,station_num,layer_num,100,-50,50);
 	gem_rh_zr[region_num][station_num][layer_num] = BookHistZR(ibooker,"rh","RecHits",region_num,station_num,layer_num);
        gem_rh_xy[region_num][station_num][layer_num] = BookHistXY(ibooker,"rh","RecHits",region_num,station_num,layer_num);
      }
    }
  }
  LogDebug("GEMRecHitsValidation")<<"Booking End.\n";
}


GEMRecHitsValidation::~GEMRecHitsValidation() {
}

void GEMRecHitsValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup& iSetup)
{
  const GEMGeometry* GEMGeometry_ ;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("MuonGEMRecHits") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }
  
  edm::Handle<GEMRecHitCollection> gemRecHits;
  edm::Handle<edm::PSimHitContainer> gemSimHits;
  e.getByToken( this->InputTagToken_, gemSimHits);
  e.getByToken( this->InputTagToken_RH,gemRecHits);
  if (!gemRecHits.isValid()) {
    edm::LogError("GEMRecHitsValidation") << "Cannot get strips by Token RecHits Token.\n";
    return ;
  }
  
  for (edm::PSimHitContainer::const_iterator hits = gemSimHits->begin(); hits!=gemSimHits->end(); ++hits) {
    
    const GEMDetId id(hits->detUnitId());
    
    Int_t sh_region = id.region();
    //Int_t sh_ring = id.ring();
    Int_t sh_roll = id.roll();
    Int_t sh_station = id.station();
    Int_t sh_layer = id.layer();
    Int_t sh_chamber = id.chamber();

    if ( GEMGeometry_->idToDet(hits->detUnitId()) == nullptr) {
      std::cout<<"simHit did not matched with GEMGeometry."<<std::endl;
      continue;
    }
   
//    if (!(abs(hits-> particleType())) == 13) continue;
    
    //const LocalPoint p0(0., 0., 0.);
    //const GlobalPoint Gp0(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(p0));
    const LocalPoint hitLP(hits->localPosition());

    const LocalPoint hitEP(hits->entryPoint());
    Int_t sh_strip = GEMGeometry_->etaPartition(hits->detUnitId())->strip(hitEP);
    
    //const GlobalPoint hitGP(GEMGeometry_->idToDet(hits->detUnitId())->surface().toGlobal(hitLP));
    //Float_t sh_l_r = hitLP.perp();
    Float_t sh_l_x = hitLP.x();
    //Float_t sh_l_y = hitLP.y();
    //Float_t sh_l_z = hitLP.z();

	
	for (GEMRecHitCollection::const_iterator recHit = gemRecHits->begin(); recHit != gemRecHits->end(); ++recHit){
		Float_t  rh_l_x = recHit->localPosition().x();
	 	Float_t  rh_l_xErr = recHit->localPositionError().xx();
	  	//Float_t  rh_l_y = recHit->localPosition().y();
	  	//Int_t  detId = (Short_t) (*recHit).gemId();
	  	//Int_t  bx = recHit->BunchX();
		Int_t  clusterSize = recHit->clusterSize();
   		Int_t  firstClusterStrip = recHit->firstClusterStrip();

  		GEMDetId id((*recHit).gemId());
	      
  		Short_t rh_region = (Short_t) id.region();
  		//Int_t rh_ring = (Short_t) id.ring();
  		Short_t rh_station = (Short_t) id.station();
  		Short_t rh_layer = (Short_t) id.layer();
  		Short_t rh_chamber = (Short_t) id.chamber();
  		Short_t rh_roll = (Short_t) id.roll();

  		LocalPoint recHitLP = recHit->localPosition();
	    	if ( GEMGeometry_->idToDet((*recHit).gemId()) == nullptr) {
		      	std::cout<<"This gem recHit did not matched with GEMGeometry."<<std::endl;
		       	continue;
	  	}
	       	GlobalPoint recHitGP = GEMGeometry_->idToDet((*recHit).gemId())->surface().toGlobal(recHitLP);

	  	Float_t     rh_g_R = recHitGP.perp();
	      	//Float_t rh_g_Eta = recHitGP.eta();
	  	//Float_t rh_g_Phi = recHitGP.phi();
	  	Float_t     rh_g_X = recHitGP.x();
	  	Float_t     rh_g_Y = recHitGP.y();
	  	Float_t     rh_g_Z = recHitGP.z();
		Float_t   rh_pullX = (Float_t)(rh_l_x - sh_l_x)/(rh_l_xErr);
      
		std::vector<int> stripsFired;
  		for(int i = firstClusterStrip; i < (firstClusterStrip + clusterSize); i++){
  		  stripsFired.push_back(i);
 		}
    
 		const bool cond1( sh_region == rh_region and sh_layer == rh_layer and sh_station == rh_station);
  		const bool cond2(sh_chamber == rh_chamber and sh_roll == rh_roll);
  		const bool cond3(std::find(stripsFired.begin(), stripsFired.end(), (sh_strip + 1)) != stripsFired.end());

  		if(cond1 and cond2 and cond3){
		//	Float_t x_pull = (sh_l_x - rh_l_x) / rh_l_xErr;
		 	 LogDebug("GEMRecHitsValidation")<< " Region : " << rh_region << "\t Station : " << rh_station 
<< "\t Layer : "<< rh_layer << "\n Radius: " << rh_g_R << "\t X : " << rh_g_X << "\t Y : "<< rh_g_Y << "\t Z : " << rh_g_Z << std::endl;	
			gem_cls_tot->Fill(clusterSize);
			gem_cls[(int)(rh_region/2.+0.5)][rh_station-1][rh_layer-1]->Fill(clusterSize);
			gem_pullX[(int)(rh_region/2.+0.5)][rh_station-1][rh_layer-1]->Fill(rh_pullX);
			gem_rh_zr[(int)(rh_region/2.+0.5)][rh_station-1][rh_layer-1]->Fill(rh_g_Z ,rh_g_R);
			gem_rh_xy[(int)(rh_region/2.+0.5)][rh_station-1][rh_layer-1]->Fill(rh_g_X ,rh_g_Y);
		}

	} //End loop on RecHits
  } //End loop on SimHits
 
}
