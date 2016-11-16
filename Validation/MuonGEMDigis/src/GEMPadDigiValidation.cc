#include "Validation/MuonGEMDigis/interface/GEMPadDigiValidation.h"
#include <TMath.h>
GEMPadDigiValidation::GEMPadDigiValidation(const edm::ParameterSet& cfg): GEMBaseValidation(cfg)
{
  InputTagToken_ = consumes<GEMPadDigiCollection>(cfg.getParameter<edm::InputTag>("PadLabel"));
  detailPlot_ = cfg.getParameter<bool>("detailPlot");
}
void GEMPadDigiValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
  const GEMGeometry* GEMGeometry_ = initGeometry(iSetup);

  if ( GEMGeometry_ == nullptr) return ;
  int npadsGE11 = GEMGeometry_->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  int npadsGE21 = 0;
  int nPads = 0;
  if ( GEMGeometry_->regions()[0]->stations()[1]->superChambers().size() != 0 ) {
    npadsGE21 = GEMGeometry_->regions()[0]->stations()[1]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  }

  for( auto& region : GEMGeometry_->regions()  ){
    int re = region->region();
    TString title_suffix = getSuffixTitle(re) ;
    TString histname_suffix = getSuffixName( re ) ;
    TString simpleZR_title    = TString::Format("ZR Occupancy%s; |Z|(cm) ; R(cm)",title_suffix.Data());
    TString simpleZR_histname = TString::Format("pad_simple_zr%s",histname_suffix.Data());

    auto* simpleZR = getSimpleZR(ibooker, simpleZR_title, simpleZR_histname);
    if ( simpleZR != nullptr) {
      thePad_simple_zr[simpleZR_histname.Hash() ] = simpleZR;
    }
    for( auto& station : region->stations()) {
      int st = station->station();
      TString title_suffix2 = getSuffixTitle( re, st) ;
      TString histname_suffix2 = getSuffixName( re, st) ;

      TString dcEta_title    = TString::Format("Occupancy for detector component %s;;#eta-partition",title_suffix2.Data());
      TString dcEta_histname = TString::Format("pad_dcEta%s",histname_suffix2.Data());

      auto* dcEta = getDCEta(ibooker, station, dcEta_title, dcEta_histname);
      if ( dcEta != nullptr) {
        thePad_dcEta[ dcEta_histname.Hash() ] = dcEta;
      }
    }
  }


  if ( detailPlot_ ) {
    for( auto& region : GEMGeometry_->regions() ) {
      int re = region->region();
      int region_num = (re+1)/2;
      for( auto& station : region->stations() ) {
      int st = station->station();
      int station_num = st-1;
         if ( station_num == 0 ) nPads = npadsGE11;
         else nPads = npadsGE21;
        for( int la = 1 ; la <= 2 ; la++) {
          int layer_num = la-1;
          std::string name_prefix  = getSuffixName( re, st, la);
          std::string label_prefix = getSuffixTitle( re, st, la) ;
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

    int re = id.region();
    int la = id.layer();
    int st = id.station();
		Short_t chamber = (Short_t) id.chamber();
    Short_t nroll = (Short_t) id.roll();
    GEMPadDigiCollection::const_iterator digiItr;

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

      int region_num  = (re+1)/2;
      int station_num = st-1;
      int layer_num   = la-1;
      int binX = (chamber-1)*2+layer_num;
      int binY = nroll;

      // Fill normal plots.
      TString histname_suffix = getSuffixName( re);
      TString simple_zr_histname = TString::Format("pad_simple_zr%s",histname_suffix.Data());
      thePad_simple_zr[simple_zr_histname.Hash()]->Fill( fabs(g_z), g_r);

      histname_suffix = getSuffixName( re, st);
      TString dcEta_histname = TString::Format("pad_dcEta%s",histname_suffix.Data());
      thePad_dcEta[dcEta_histname.Hash()]->Fill( binX, binY); 

      if ( detailPlot_) {
        theCSCPad_xy[region_num][station_num][layer_num]->Fill(g_x,g_y);     
        theCSCPad_phipad[region_num][station_num][layer_num]->Fill(g_phi,pad);
        theCSCPad[region_num][station_num][layer_num]->Fill(pad);
        theCSCPad_bx[region_num][station_num][layer_num]->Fill(bx);
        theCSCPad_zr[region_num][station_num][layer_num]->Fill(g_z,g_r);
        std::string name_prefix = getSuffixName( re, st, la);
        TString hname;
        if ( chamber %2 == 0 ) { hname = TString::Format("pad_dg_xy%s_even",name_prefix.c_str()); }
        else { hname = TString::Format("pad_dg_xy%s_odd",name_prefix.c_str()); }
        theCSCPad_xy_ch[hname.Hash()]->Fill(g_x,g_y);
      }
    }
  }
}
