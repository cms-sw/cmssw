#include "Validation/MuonGEMDigis/interface/GEMCoPadDigiValidation.h"

GEMCoPadDigiValidation::GEMCoPadDigiValidation(const edm::ParameterSet& cfg): GEMBaseValidation(cfg)
{
  InputTagToken_ = consumes<GEMCoPadDigiCollection>(cfg.getParameter<edm::InputTag>("CopadLabel"));
}
void GEMCoPadDigiValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
  const GEMGeometry* GEMGeometry_ ;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("GEMCoPadDigiValidation") << "+++ Error : GEM geometry is unavailable on histogram booking. +++\n";
    return;
  }

  const double PI = TMath::Pi();


  int npadsGE11 = GEMGeometry_->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  int npadsGE21 = 0;
  int nPads = 0;

  int nregions = GEMGeometry_->regions().size();
  int nstations = GEMGeometry_->regions()[0]->stations().size(); 
  if ( nstations > 1 ) {
    npadsGE21  = GEMGeometry_->regions()[0]->stations()[2]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  }
  for( int region_num = 0 ; region_num < nregions ; region_num++ ) {
    std::string name_prefix  = std::string("_r")+regionLabel[region_num];
    std::string label_prefix = "region "+regionLabel[region_num];
    for( int station_num = 0 ; station_num < nstations ; station_num++) {
      if ( station_num == 0 ) nPads = npadsGE11;
      else nPads = npadsGE21;
      if ( station_num ==1 ) continue;
      name_prefix  = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num];
      label_prefix = "region"+regionLabel[region_num]+" station "+stationLabel[station_num];
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
    Short_t region  = (Short_t)  id.region();
    Short_t station = (Short_t) id.station();
    // Not yet remove station 2_long. station 2 mean station 2_long. So, set it to 3.
    if ( station ==2 ) {
      station=3; // fix wrong assignment.
    }
		Short_t chamber = (Short_t) id.chamber();
    GEMCoPadDigiCollection::const_iterator digiItr;
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      GEMDetId roId = GEMDetId(region, id.ring(), station, id.layer(), chamber,digiItr->roll());
      LogDebug("GEMCoPadDigiValidation")<<"roId : "<<roId;
      const GeomDet* gdet = GEMGeometry_->idToDet(roId);
      if ( gdet == nullptr) { 
        edm::LogError("GEMCoPadDigiValidation")<<"Getting DetId failed. Discard this gem copad hit.Maybe it comes from unmatched geometry between GEN and DIGI.";
        continue; 
      }
      const BoundPlane & surface = gdet->surface();
      LogDebug("GEMCoPadDigiValidation")<<" ID : "<<roId;
      const GEMEtaPartition * roll = GEMGeometry_->etaPartition(roId);
      LogDebug("GEMCoPadDigiValidation")<<" roll's n pad : "<<roll->npads();

      Short_t pad = (Short_t) digiItr->pad(1);
      Short_t bx  = (Short_t) digiItr->bx(1);
      LogDebug("GEMCoPadDigiValidation")<<" copad #1 pad : "<<pad<<"  bx : "<<bx;
      LogDebug("GEMCoPadDigiValidation")<<" copad #2 pad : "<<digiItr->pad(2)<<"  bx : "<<digiItr->bx(2);

      LocalPoint lp = roll->centreOfPad(pad);

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t) gp.perp();
      Float_t g_phi = (Float_t) gp.phi();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
      Float_t g_z = (Float_t) gp.z();

      int region_num=0;
      int station_num = station-1;

      if ( region == -1 ) region_num = 0 ; 
      else if (region == 1 ) region_num = 1; 
      else {
        edm::LogError("GEMCOPadDIGIValidation")<<"region : "<<region<<std::endl;
      }


      theCSCCoPad_xy[region_num][station_num]->Fill(g_x,g_y);     
      theCSCCoPad_phipad[region_num][station_num]->Fill(g_phi,pad);
      theCSCCoPad[region_num][station_num]->Fill(pad);
      theCSCCoPad_bx[region_num][station_num]->Fill(bx);
      theCSCCoPad_zr[region_num][station_num]->Fill(g_z,g_r);
			std::string name_prefix = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num];
			TString hname;
      if ( chamber %2 == 0 ) { hname = TString::Format("copad_dg_xy%s_even",name_prefix.c_str()); }
      else { hname = TString::Format("copad_dg_xy%s_odd",name_prefix.c_str()); }
      theCSCCoPad_xy_ch[hname.Hash()]->Fill(g_x,g_y);
		}
  }
}
