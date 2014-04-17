#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"
GEMStripDigiValidation::GEMStripDigiValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag)
:  GEMBaseValidation(dbe, inputTag)
{}

void GEMStripDigiValidation::bookHisto(const GEMGeometry* geom) { 
  theGEMGeometry = geom;  
  std::string region[2]= { "-1","1" } ;
  std::string station[3]= { "1","2","3" } ;
  std::string layer[2]= { "1","2" } ;

  theStrip_zr_rm1 = dbe_->book2D("strip_dg_zr_rm1", "Digi occupancy: region-1; globalZ [cm] ; globalR [cm] ", 200,-573,-564, 55,130,240);
  theStrip_zr_rp1 = dbe_->book2D("strip_dg_zr_rp1", "Digi occupancy: region 1; globalZ [cm] ; globalR [cm] ", 200,564,573,55,130,240);

  int nstripsGE11  = theGEMGeometry->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->nstrips();
  int nstripsGE21 = 0;

  int nregions = theGEMGeometry->regions().size();
  int nstations = theGEMGeometry->regions()[0]->stations().size(); 
  
  if ( nstations > 1 ) {
    nstripsGE21  = theGEMGeometry->regions()[0]->stations()[1]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->nstrips();
  }
  else LogDebug("GEMStripDIGIValidation")<<"Info : Only 1 station is existed.\n";


  int nstrips = 0;


  for( int region_num = 0 ; region_num <nregions ; region_num++ ) {
    for( int layer_num = 0 ; layer_num < 2 ; layer_num++) {
      std::string name_prefix  = std::string("_r")+region[region_num]+"_l"+layer[layer_num];
      std::string label_prefix = "region "+region[region_num]+" layer "+layer[layer_num];
      theStrip_bx[region_num][layer_num] = dbe_->book1D( ("strip_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5);
      for( int station_num = 0 ; station_num < nstations ; station_num++) {
        if ( station_num == 0 ) nstrips = nstripsGE11;
        else nstrips = nstripsGE21; 
        name_prefix  = std::string("_r")+region[region_num]+"_st"+station[station_num]+"_l"+layer[layer_num];
        label_prefix = "region"+region[region_num]+" station "+station[station_num]+" layer "+layer[layer_num];

        theStrip_phistrip[region_num][station_num][layer_num] = dbe_->book2D( ("strip_dg_phistrip"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+"; phi [rad];strip number").c_str(), 280, -TMath::Pi(), TMath::Pi(), nstrips/2,0,nstrips);
        theStrip[region_num][station_num][layer_num] = dbe_->book1D( ("strip_dg"+name_prefix).c_str(), ("Digi occupancy per stip number: "+label_prefix+";strip number; entries").c_str(), nstrips,0.5,nstrips+0.5);
        theStrip_xy[region_num][station_num][layer_num] = dbe_->book2D( ("strip_dg_xy"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+";globalX [cm]; globalY[cm]").c_str(), 260, -260,260,260,-260,260);
      }
    }
  }
}


GEMStripDigiValidation::~GEMStripDigiValidation() {
 

}


void GEMStripDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{
  edm::Handle<GEMDigiCollection> gem_digis;
  e.getByLabel(theInputTag, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMStripDigiValidation") << "Cannot get strips by label "
                                       << theInputTag.encode();
  }
  for (GEMDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {
    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = theGEMGeometry->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = theGEMGeometry->etaPartition(id);

    Short_t region = (Short_t) id.region();
    Short_t layer = (Short_t) id.layer();
    Short_t station = (Short_t) id.station();

    GEMDigiCollection::const_iterator digiItr;
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      Short_t strip = (Short_t) digiItr->strip();
      Short_t bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t) gp.perp();
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

      theStrip_xy[region_num][station_num][layer_num]->Fill(g_x,g_y);     
      theStrip_phistrip[region_num][station_num][layer_num]->Fill(g_phi,strip);
      theStrip[region_num][station_num][layer_num]->Fill(strip);
      theStrip_bx[region_num][layer_num]->Fill(bx);
 
      if ( region== -1 ) {
                theStrip_zr_rm1->Fill(g_z,g_r);
      }
      else if ( region == 1 ) {
                theStrip_zr_rp1->Fill(g_z,g_r);
      }
      else {
        edm::LogInfo("StripDIGIValidation")<<"region : "<<region<<std::endl;
      }
   }
  }
}
