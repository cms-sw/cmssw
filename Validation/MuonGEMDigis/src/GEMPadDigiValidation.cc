#include "Validation/MuonGEMDigis/interface/GEMPadDigiValidation.h"

GEMPadDigiValidation::GEMPadDigiValidation(const edm::ParameterSet& cfg): GEMBaseValidation(cfg)
{
  InputTagToken_ = consumes<GEMPadDigiCollection>(cfg.getParameter<edm::InputTag>("PadLabel"));
}
void GEMPadDigiValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
  const GEMGeometry* GEMGeometry_ ;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("GEMPadDigiValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }
  int npadsGE11 = GEMGeometry_->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  int npadsGE21 = 0;
  int nPads = 0;

  int nregions = GEMGeometry_->regions().size();
  int nstations = GEMGeometry_->regions()[0]->stations().size(); 
  if ( nstations > 1 ) {
    npadsGE21  = GEMGeometry_->regions()[0]->stations()[1]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  }

  for( int region_num = 0 ; region_num < nregions ; region_num++ ) {
    for( int layer_num = 0 ; layer_num < 2 ; layer_num++) {
      std::string name_prefix  = std::string("_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
      std::string label_prefix = "region "+regionLabel[region_num]+" layer "+layerLabel[layer_num];
      for( int station_num = 0 ; station_num < nstations ; station_num++) {
        if ( station_num == 0 ) nPads = npadsGE11;
        else nPads = npadsGE21;
        if ( station_num ==1 ) continue;
        name_prefix  = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
        label_prefix = "region"+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num];
        theCSCPad_phipad[region_num][station_num][layer_num] = ibooker.book2D( ("pad_dg_phipad"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+"; phi [rad]; Pad number").c_str(), 280,-TMath::Pi(),TMath::Pi(), nPads/2,0,nPads );
        theCSCPad[region_num][station_num][layer_num] = ibooker.book1D( ("pad_dg"+name_prefix).c_str(), ("Digi occupancy per pad number: "+label_prefix+";Pad number; entries").c_str(), nPads,0.5,nPads+0.5);
        theCSCPad_bx[region_num][station_num][layer_num] = ibooker.book1D( ("pad_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5);
        theCSCPad_zr[region_num][station_num][layer_num] = BookHistZR(ibooker,"pad_dg","Pad Digi",region_num,station_num,layer_num);
        theCSCPad_xy[region_num][station_num][layer_num] = BookHistXY(ibooker,"pad_dg","Pad Digi",region_num,station_num,layer_num);
				TString xy_name = TString::Format("pad_dg_xy%s_odd",name_prefix.c_str());
        TString xy_title = TString::Format("Digi XY occupancy %s at odd chambers",label_prefix.c_str());
        theCSCPad_xy_ch[ xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360,360, 360, -360, 360);
        xy_name = TString::Format("pad_dg_xy%s_even",name_prefix.c_str());
        xy_title = TString::Format("Digi XY occupancy %s at even chambers",label_prefix.c_str());
        theCSCPad_xy_ch[ xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360,360, 360, -360, 360);
      }
    }
  }
}


GEMPadDigiValidation::~GEMPadDigiValidation() {
 

}


void GEMPadDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup& iSetup)
{
  const GEMGeometry* GEMGeometry_ ;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("GEMPadDigiValidaation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }
  edm::Handle<GEMPadDigiCollection> gem_digis;
  e.getByToken(InputTagToken_, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMPadDigiValidation") << "Cannot get pads by label GEMPadToken.";
  }

  for (GEMPadDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {

    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = GEMGeometry_->idToDet(id);
    if ( gdet == nullptr) { 
      std::cout<<"Getting DetId failed. Discard this gem pad hit.Maybe it comes from unmatched geometry."<<std::endl;
      continue; 
    }
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = GEMGeometry_->etaPartition(id);

    Short_t region = (Short_t) id.region();
    Short_t layer = (Short_t) id.layer();
    Short_t station = (Short_t) id.station();
		Short_t chamber = (Short_t) id.chamber();
    GEMPadDigiCollection::const_iterator digiItr;

    if ( station ==2 ) continue;
    //loop over digis of given roll
    //
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      Short_t pad = (Short_t) digiItr->pad();
      Short_t bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfPad(digiItr->pad());

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t) gp.perp();
      Float_t g_phi = (Float_t) gp.phi();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
      Float_t g_z = (Float_t) gp.z();
      edm::LogInfo("GEMPadDIGIValidation")<<"Global x "<<g_x<<"Global y "<<g_y<<"\n";  
      edm::LogInfo("GEMPadDIGIValidation")<<"Global pad "<<pad<<"Global phi "<<g_phi<<std::endl; 
      edm::LogInfo("GEMPadDIGIValidation")<<"Global bx "<<bx<<std::endl; 

      int region_num=0;
      int station_num = station-1;
      int layer_num = layer-1;
      if ( region == -1 ) region_num = 0 ; 
      else if (region == 1 ) region_num = 1; 

      theCSCPad_xy[region_num][station_num][layer_num]->Fill(g_x,g_y);     
      theCSCPad_phipad[region_num][station_num][layer_num]->Fill(g_phi,pad);
      theCSCPad[region_num][station_num][layer_num]->Fill(pad);
      theCSCPad_bx[region_num][station_num][layer_num]->Fill(bx);
      theCSCPad_zr[region_num][station_num][layer_num]->Fill(g_z,g_r);
		  std::string name_prefix = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num] + "_l"+layerLabel[layer_num];
      TString hname;
      if ( chamber %2 == 0 ) { hname = TString::Format("pad_dg_xy%s_even",name_prefix.c_str()); }
      else { hname = TString::Format("pad_dg_xy%s_odd",name_prefix.c_str()); }
      theCSCPad_xy_ch[hname.Hash()]->Fill(g_x,g_y);
    }
  }
}
