#include "Validation/MuonGEMRecHits/interface/GEMRecHitsValidation.h"
#include <iomanip>
GEMRecHitsValidation::GEMRecHitsValidation(const edm::ParameterSet& cfg): GEMBaseValidation(cfg)
{
  InputTagToken_   = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagToken_RH = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitsInputLabel"));
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

  for( int region_num = 0 ; region_num <nregions ; region_num++ ) {
    for( int layer_num = 0 ; layer_num < 2 ; layer_num++) {
      for( int station_num = 0 ; station_num < nstations ; station_num++) {
  //      if ( station_num == 0 ) nstrips = nstripsGE11;
  //      else nstrips = nstripsGE21;
        std::string name_prefix = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num] + "_l"+layerLabel[layer_num];
        std::string label_prefix = "region"+regionLabel[region_num]+" station "+stationLabel[station_num] +" layer "+layerLabel[layer_num];
       /* theStrip_phistrip[region_num][station_num][layer_num] = ibooker.book2D( ("strip_dg_phistrip"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+"; phi [rad];strip number").c_str(), 280, -TMath::Pi(), TMath::Pi(), nstrips/2,0,nstrips);
        theStrip[region_num][station_num][layer_num] = ibooker.book1D( ("strip_dg"+name_prefix).c_str(), ("Digi occupancy per stip number: "+label_prefix+";strip number; entries").c_str(), nstrips,0.5,nstrips+0.5);
        theStrip_bx[region_num][station_num][layer_num] = ibooker.book1D( ("strip_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5); */
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
  for (edm::PSimHitContainer::const_iterator itHit = gemSimHits->begin(); itHit!=gemSimHits->end(); ++itHit) {
	for (GEMRecHitCollection::const_iterator recHit = gemRecHits->begin(); recHit != gemRecHits->end(); ++recHit){
		std::cout<<"Loooooooooping ------->>  "<<std::endl;
	}
  }
 /* for (GEMDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {
    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = GEMGeometry_->idToDet(id);
    if ( gdet == nullptr) { 
      std::cout<<"Getting DetId failed. Discard this gem strip hit.Maybe it comes from unmatched geometry."<<std::endl;
      continue; 
    }
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = GEMGeometry_->etaPartition(id);

    Short_t region = (Short_t) id.region();
    Short_t layer = (Short_t) id.layer();
    Short_t station = (Short_t) id.station();
    //Short_t chamber = (Short_t) id.chamber();
    //Short_t nroll = (Short_t) id.roll();

    GEMDigiCollection::const_iterator digiItr;
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      Short_t strip = (Short_t) digiItr->strip();
      Short_t bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t) gp.perp();
      //Float_t g_eta = (Float_t) gp.eta();
      Float_t g_phi = (Float_t) gp.phi();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
      Float_t g_z = (Float_t) gp.z();
      edm::LogInfo("GEMStripDIGIValidation")<<"Global x "<<g_x<<"Global y "<<g_y<<std::endl;  
      edm::LogInfo("GEMStripDIGIValidation")<<"Global strip "<<strip<<"Global phi "<<g_phi<<std::endl;  
      edm::LogInfo("GEMStripDIGIValidation")<<"Global bx "<<bx<<std::endl;  
      // fill hist
      int region_num=0 ;
      if ( region ==-1 ) region_num = 0 ;
      else if ( region==1) region_num = 1;  
      int station_num = station-1;
      int layer_num = layer-1;
    
      if ( theStrip_xy[region_num][station_num][layer_num] != nullptr) {
        theStrip_xy[region_num][station_num][layer_num]->Fill(g_x,g_y);     
        theStrip_phistrip[region_num][station_num][layer_num]->Fill(g_phi,strip);
        theStrip[region_num][station_num][layer_num]->Fill(strip);
        theStrip_bx[region_num][station_num][layer_num]->Fill(bx);
        theStrip_zr[region_num][station_num][layer_num]->Fill(g_z,g_r);
      }
      else {
        std::cout<<"Error is occued when histograms is called."<<std::endl;
      }
    }    
  }*/
}
