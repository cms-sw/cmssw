#include "Validation/MuonGEMDigis/interface/GEMCoPadDigiValidation.h"
#include <TMath.h>

GEMCoPadDigiValidation::GEMCoPadDigiValidation(const edm::ParameterSet& cfg): GEMBaseValidation(cfg)
{
  InputTagToken_ = consumes<GEMCoPadDigiCollection>(cfg.getParameter<edm::InputTag>("CopadLabel"));
  detailPlot_ = cfg.getParameter<bool>("detailPlot");
  minBXGEM_ = cfg.getParameter<int>("minBXGEM"); 
  maxBXGEM_ = cfg.getParameter<int>("maxBXGEM"); 
}
void GEMCoPadDigiValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
  const GEMGeometry* GEMGeometry_ = initGeometry(iSetup);

  const double PI = TMath::Pi();

  int npadsGE11 = GEMGeometry_->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  int npadsGE21 = 0;
  int nPads = 0;

  if ( GEMGeometry_->regions()[0]->stations()[1]->superChambers().size() != 0 ) {
    npadsGE21  = GEMGeometry_->regions()[0]->stations()[1]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  }

  for( auto& region : GEMGeometry_->regions()  ){
    int re = region->region();
    TString title_suffix = getSuffixTitle(re); 
    TString histname_suffix = getSuffixName( re) ;
    TString simpleZR_title    = TString::Format("Copad ZR Occupancy%s; |Z|(cm) ; R(cm)",title_suffix.Data());
    TString simpleZR_histname = TString::Format("copad_simple_zr%s",histname_suffix.Data());

    auto* simpleZR = getSimpleZR(ibooker, simpleZR_title, simpleZR_histname);
    if ( simpleZR != nullptr) {
      theCoPad_simple_zr[simpleZR_histname.Hash() ] = simpleZR;
    }
    for( auto& station : region->stations()) {
      int st = station->station();
      TString title_suffix2 = getSuffixTitle( re , st) ;
      TString histname_suffix2 = getSuffixName( re, st) ;

      TString dcEta_title    = TString::Format("Copad's occupancy for detector component %s; # of sub-detector ;#eta-partition",title_suffix2.Data());
      TString dcEta_histname = TString::Format("copad_dcEta%s",histname_suffix2.Data());

      auto* dcEta = getDCEta(ibooker, station, dcEta_title, dcEta_histname);
      if ( dcEta != nullptr) {
        theCoPad_dcEta[ dcEta_histname.Hash() ] = dcEta;
      }
    }

    if ( detailPlot_) {
      for( auto&  region : GEMGeometry_->regions() ) {
        int re = region->region();
        int region_num = (re+1)/2;
        std::string name_prefix  = getSuffixName( re);
        std::string label_prefix = getSuffixTitle(re);
        for( auto& station : region->stations() ) {
          int st = station->station();
          int station_num = st-1;
            
          if ( st == 1 ) nPads = npadsGE11;
          else nPads = npadsGE21;
          name_prefix  = getSuffixName( re, st) ;
          label_prefix = getSuffixTitle( re, st) ;
          theCSCCoPad_phipad[region_num][station_num] = ibooker.book2D( ("copad_dg_phipad"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+"; phi [rad]; Pad number").c_str(), 280,-PI,PI, nPads/2,0,nPads );
          theCSCCoPad[region_num][station_num] = ibooker.book1D( ("copad_dg"+name_prefix).c_str(), ("Digi occupancy per pad number: "+label_prefix+";Pad number; entries").c_str(), nPads,0.5,nPads+0.5);
          theCSCCoPad_bx[region_num][station_num] = ibooker.book1D( ("copad_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5);
          theCSCCoPad_zr[region_num][station_num] = BookHistZR( ibooker, "copad_dg","CoPad Digi",region_num,station_num);
          theCSCCoPad_xy[region_num][station_num] = BookHistXY( ibooker, "copad_dg","CoPad Digi",region_num,station_num);
          TString xy_name = TString::Format("copad_dg_xy%s_odd",name_prefix.c_str());
          TString xy_title = TString::Format("Digi XY occupancy %s at odd chambers",label_prefix.c_str());
          theCSCCoPad_xy_ch[ xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360,360, 360, -360, 360);
          xy_name = TString::Format("copad_dg_xy%s_even",name_prefix.c_str());
          xy_title = TString::Format("Digi XY occupancy %s at even chambers",label_prefix.c_str());
          theCSCCoPad_xy_ch[ xy_name.Hash()] = ibooker.book2D(xy_name, xy_title, 360, -360,360, 360, -360, 360);
        }
      }
    }
  }
}


GEMCoPadDigiValidation::~GEMCoPadDigiValidation() {


}


void GEMCoPadDigiValidation::analyze(const edm::Event& e,
    const edm::EventSetup& iSetup)
{
  const GEMGeometry* GEMGeometry_;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("GEMCoPadDigiValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return;
  }
  edm::Handle<GEMCoPadDigiCollection> gem_digis;
  e.getByToken(InputTagToken_, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMCoPadDigiValidation") << "Cannot get pads by token.";
    return ;
  }

  for (GEMCoPadDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {
    GEMDetId id = (*cItr).first;
    int re = id.region();
    int st = id.station();
    int la = id.layer();
    Short_t chamber = (Short_t) id.chamber();
    GEMCoPadDigiCollection::const_iterator digiItr;
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      GEMDetId roId = GEMDetId(re, id.ring(), st, la, chamber, digiItr->roll());
      Short_t nroll = roId.roll();  
      LogDebug("GEMCoPadDigiValidation")<<"roId : "<<roId;
      const GeomDet* gdet = GEMGeometry_->idToDet(roId);
      if ( gdet == nullptr) {
        edm::LogError("GEMCoPadDigiValidation")<<roId<<" : This part can not load from GEMGeometry // Original"<<id<<" station : "<<st;
        edm::LogError("GEMCoPadDigiValidation")<<"Getting DetId failed. Discard this gem copad hit.Maybe it comes from unmatched geometry between GEN and DIGI.";
        continue; 
      }
      const BoundPlane & surface = gdet->surface();
      const GEMEtaPartition * roll = GEMGeometry_->etaPartition(roId);
      LogDebug("GEMCoPadDigiValidation")<<" roll's n pad : "<<roll->npads();

      Short_t pad1 = (Short_t) digiItr->pad(1);
      Short_t pad2 = (Short_t) digiItr->pad(2);
      Short_t bx1  = (Short_t) digiItr->bx(1);
      Short_t bx2  = (Short_t) digiItr->bx(2);
      LogDebug("GEMCoPadDigiValidation")<<" copad #1 pad : "<<pad1<<"  bx : "<<bx1;
      LogDebug("GEMCoPadDigiValidation")<<" copad #2 pad : "<<pad2<<"  bx : "<<bx2;

      // Filtered using BX
      if ( bx1 < (Short_t)minBXGEM_ || bx1 > (Short_t)maxBXGEM_) continue;
      if ( bx2 < (Short_t)minBXGEM_ || bx2 > (Short_t)maxBXGEM_) continue;

      LocalPoint lp1 = roll->centreOfPad(pad1);
      LocalPoint lp2 = roll->centreOfPad(pad2);

      GlobalPoint gp1 = surface.toGlobal(lp1);
      GlobalPoint gp2 = surface.toGlobal(lp2);
      Float_t g_r1 = (Float_t) gp1.perp();
      Float_t g_r2 = (Float_t) gp2.perp();
      Float_t g_z1 = (Float_t) gp1.z();
      Float_t g_z2 = (Float_t) gp2.z();

      Float_t g_phi = (Float_t) gp1.phi();
      Float_t g_x = (Float_t) gp1.x();
      Float_t g_y = (Float_t) gp1.y();

      int region_num=0;
      if ( re == -1 ) region_num = 0 ; 
      else if (re == 1 ) region_num = 1; 
      else {
        edm::LogError("GEMCoPadDIGIValidation")<<"region : "<<re<<std::endl;
      }
      int binX = (chamber-1)*2+(la-1);
      int binY = nroll;
      int station_num = st-1;

      // Fill normal plots.
      TString histname_suffix = getSuffixName( re); 
      TString simple_zr_histname = TString::Format("copad_simple_zr%s",histname_suffix.Data());
      theCoPad_simple_zr[simple_zr_histname.Hash()]->Fill( fabs(g_z1), g_r1);
      theCoPad_simple_zr[simple_zr_histname.Hash()]->Fill( fabs(g_z2), g_r2);

      histname_suffix = getSuffixName( re, st ) ;
      TString dcEta_histname = TString::Format("copad_dcEta%s",histname_suffix.Data());
      theCoPad_dcEta[dcEta_histname.Hash()]->Fill( binX, binY); 
      theCoPad_dcEta[dcEta_histname.Hash()]->Fill( binX+1, binY); 

      // Fill detail plots.
      if ( detailPlot_) {
        theCSCCoPad_xy[region_num][station_num]->Fill(g_x,g_y);     
        theCSCCoPad_phipad[region_num][station_num]->Fill(g_phi,pad1);
        theCSCCoPad[region_num][station_num]->Fill(pad1);
        theCSCCoPad_bx[region_num][station_num]->Fill(bx1);
        theCSCCoPad_bx[region_num][station_num]->Fill(bx2);
        theCSCCoPad_zr[region_num][station_num]->Fill(g_z1,g_r1);
        theCSCCoPad_zr[region_num][station_num]->Fill(g_z2,g_r2);
        std::string name_prefix = getSuffixName( re, st) ;
        TString hname;
        if ( chamber %2 == 0 ) { hname = TString::Format("copad_dg_xy%s_even",name_prefix.c_str()); }
        else { hname = TString::Format("copad_dg_xy%s_odd",name_prefix.c_str()); }
        theCSCCoPad_xy_ch[hname.Hash()]->Fill(g_x,g_y);
      }
    }
  }
}
