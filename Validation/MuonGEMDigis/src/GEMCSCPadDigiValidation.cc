#include "Validation/MuonGEMDigis/interface/GEMCSCPadDigiValidation.h"

GEMCSCPadDigiValidation::GEMCSCPadDigiValidation(DQMStore* dbe,
                                               edm::EDGetToken& inputToken, const edm::ParameterSet& pbInfo)
:  GEMBaseValidation(dbe, inputToken, pbInfo)
{}
void GEMCSCPadDigiValidation::bookHisto(const GEMGeometry* geom) {
  theGEMGeometry = geom;

  int npadsGE11 = theGEMGeometry->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  int npadsGE21 = 0;
  int nPads = 0;

  int nregions = theGEMGeometry->regions().size();
  int nstations = theGEMGeometry->regions()[0]->stations().size(); 
  if ( nstations > 1 ) {
    npadsGE21  = theGEMGeometry->regions()[0]->stations()[1]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
  }

  for( int region_num = 0 ; region_num < nregions ; region_num++ ) {
    for( int layer_num = 0 ; layer_num < 2 ; layer_num++) {
      std::string name_prefix  = std::string("_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
      std::string label_prefix = "region "+regionLabel[region_num]+" layer "+layerLabel[layer_num];
      for( int station_num = 0 ; station_num < nstations ; station_num++) {
        if ( station_num == 0 ) nPads = npadsGE11;
        else nPads = npadsGE21;
        name_prefix  = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
        label_prefix = "region"+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num];
        theCSCPad_phipad[region_num][station_num][layer_num] = dbe_->book2D( ("pad_dg_phipad"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+"; phi [rad]; Pad number").c_str(), 280,-TMath::Pi(),TMath::Pi(), nPads/2,0,nPads );
        theCSCPad[region_num][station_num][layer_num] = dbe_->book1D( ("pad_dg"+name_prefix).c_str(), ("Digi occupancy per pad number: "+label_prefix+";Pad number; entries").c_str(), nPads,0.5,nPads+0.5);
        theCSCPad_bx[region_num][station_num][layer_num] = dbe_->book1D( ("pad_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5);
        theCSCPad_zr[region_num][station_num][layer_num] = BookHistZR("pad_dg","Pad Digi",region_num,station_num,layer_num);
        theCSCPad_xy[region_num][station_num][layer_num] = BookHistXY("pad_dg","Pad Digi",region_num,station_num,layer_num);
      }
    }
  }
}


GEMCSCPadDigiValidation::~GEMCSCPadDigiValidation() {
 

}


void GEMCSCPadDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{
  edm::Handle<GEMCSCPadDigiCollection> gem_digis;
  e.getByToken(inputToken_, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMCSCPadDigiValidation") << "Cannot get pads by label GEMCSCPadToken.";
  }

  for (GEMCSCPadDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {

    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = theGEMGeometry->idToDet(id);
    if ( gdet == nullptr) { 
      std::cout<<"Getting DetId failed. Discard this gem pad hit.Maybe it comes from unmatched geometry."<<std::endl;
      continue; 
    }
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = theGEMGeometry->etaPartition(id);

    Short_t region = (Short_t) id.region();
    Short_t layer = (Short_t) id.layer();
    Short_t station = (Short_t) id.station();
    GEMCSCPadDigiCollection::const_iterator digiItr;
    //loop over digis of given roll
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
      edm::LogInfo("GEMCSCPadDIGIValidation")<<"Global x "<<g_x<<"Global y "<<g_y<<"\n";  
      edm::LogInfo("GEMCSCPadDIGIValidation")<<"Global pad "<<pad<<"Global phi "<<g_phi<<std::endl; 
      edm::LogInfo("GEMCSCPadDIGIValidation")<<"Global bx "<<bx<<std::endl; 

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
    }
  }
}
