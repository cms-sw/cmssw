#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"
#include <iomanip>
GEMStripDigiValidation::GEMStripDigiValidation(const edm::ParameterSet& cfg): GEMBaseValidation(cfg)
{
  InputTagToken_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("stripLabel"));
}

void GEMStripDigiValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
  const GEMGeometry* GEMGeometry_ ;
  
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("MuonGEMStripDigis") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }

  LogDebug("GEMStripDIGIValidation")<<"Geometry is acquired from MuonGeometryRecord\n";
  ibooker.setCurrentFolder("MuonGEMDigisV/GEMDigisTask");
  LogDebug("GEMStripDIGIValidation")<<"ibooker set current folder\n";

  int nregions = GEMGeometry_->regions().size();
  LogDebug("GEMStripDIGIValidation")<<"nregions set.\n";
  int nstations = GEMGeometry_->regions()[0]->stations().size(); 
  LogDebug("GEMStripDIGIValidation")<<"nstations set.\n";
  int nstripsGE11  = 384;
  int nstripsGE21 = 768;
 
  LogDebug("GEMStripDIGIValidation")<<"Successfully binning set.\n";


  int nstrips = 0;

  for( int region_num = 0 ; region_num <nregions ; region_num++ ) {
    for( int layer_num = 0 ; layer_num < 2 ; layer_num++) {
      for( int station_num = 0 ; station_num < nstations ; station_num++) {
        if ( station_num == 0 ) nstrips = nstripsGE11;
        else nstrips = nstripsGE21;
        std::string name_prefix = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num] + "_l"+layerLabel[layer_num];
        std::string label_prefix = "region"+regionLabel[region_num]+" station "+stationLabel[station_num] +" layer "+layerLabel[layer_num];
        theStrip_phistrip[region_num][station_num][layer_num] = ibooker.book2D( ("strip_dg_phistrip"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+"; phi [rad];strip number").c_str(), 280, -TMath::Pi(), TMath::Pi(), nstrips/2,0,nstrips);
        theStrip[region_num][station_num][layer_num] = ibooker.book1D( ("strip_dg"+name_prefix).c_str(), ("Digi occupancy per stip number: "+label_prefix+";strip number; entries").c_str(), nstrips,0.5,nstrips+0.5);
        theStrip_bx[region_num][station_num][layer_num] = ibooker.book1D( ("strip_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5);
        theStrip_zr[region_num][station_num][layer_num] = BookHistZR(ibooker,"strip_dg","Strip Digi",region_num,station_num,layer_num);
        theStrip_xy[region_num][station_num][layer_num] = BookHistXY(ibooker,"strip_dg","Strip Digi",region_num,station_num,layer_num);
        TString xy_name = TString::Format("strip_dg_xy%s_odd",name_prefix.c_str());
        TString xy_title = TString::Format("Digi XY occupancy %s at odd chambers",label_prefix.c_str());
        theStrip_xy_ch[ xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360,360, 360, -360, 360);
        xy_name = TString::Format("strip_dg_xy%s_even",name_prefix.c_str());
        xy_title = TString::Format("Digi XY occupancy %s at even chambers",label_prefix.c_str());
        theStrip_xy_ch[ xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360,360, 360, -360, 360);
      }
    }
  }
  LogDebug("GEMStripDIGIValidation")<<"Booking End.\n";
}


GEMStripDigiValidation::~GEMStripDigiValidation() {
}

void GEMStripDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup& iSetup)
{
  const GEMGeometry* GEMGeometry_ ;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("MuonGEMStripDigis") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }

  edm::Handle<GEMDigiCollection> gem_digis;
  e.getByToken( this->InputTagToken_, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMStripDigiValidation") << "Cannot get strips by Token stripToken.\n";
    return ;
  }
  for (GEMDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {
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
    Short_t chamber = (Short_t) id.chamber();
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

        std::string name_prefix = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num] + "_l"+layerLabel[layer_num];
        TString hname;
        if ( chamber %2 == 0 ) { hname = TString::Format("strip_dg_xy%s_even",name_prefix.c_str()); }
        else { hname = TString::Format("strip_dg_xy%s_odd",name_prefix.c_str()); }
        theStrip_xy_ch[hname.Hash()]->Fill(g_x,g_y);
      }
      else {
        std::cout<<"Error is occued when histograms is called."<<std::endl;
      }
    }    
  }
}
