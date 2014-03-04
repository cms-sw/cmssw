#include "Validation/MuonGEMDigis/interface/GEMCSCCoPadDigiValidation.h"

GEMCSCCoPadDigiValidation::GEMCSCCoPadDigiValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag)
:  GEMBaseValidation(dbe, inputTag) { }

void GEMCSCCoPadDigiValidation::bookHisto() {

   int npadsGE11 = theGEMGeometry->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->npads();
   int nPads = npadsGE11 ;
 
   theCSCCoPad_xy_rm1 = dbe_->book2D("copad_dg_xy_rm1", "Digi occupancy: region -1;globalX [cm]; globalY[cm]", 260,-260,260,260,-260,260);
   theCSCCoPad_xy_rp1 = dbe_->book2D("copad_dg_xy_rp1", "Digi occupancy: region  1;globalX [cm]; globalY[cm]", 260,-260,260,260,-260,260);

   theCSCCoPad_phipad_rm1 =  dbe_->book2D("copad_dg_phipad_rm1", "Digi occupancy: region -1; phi [rad];pad number ", 280, -TMath::Pi(),TMath::Pi(),nPads/2,0,nPads);
   theCSCCoPad_phipad_rp1 =  dbe_->book2D("copad_dg_phipad_rp1", "Digi occupancy: region  1; phi [rad];pad number ", 280, -TMath::Pi(),TMath::Pi(),nPads/2,0,nPads);

   theCSCCoPad_rm1 =  dbe_->book1D("copad_dg_rm1", "Digi occupancy per stip number: region -1;pad number; entries", nPads,0.5,nPads+0.5);
   theCSCCoPad_rp1 =  dbe_->book1D("copad_dg_rp1", "Digi occupancy per stip number: region  1;pad number; entries", nPads,0.5,nPads+0.5);


   theCSCCoPad_bx_rm1 = dbe_->book1D("copad_dg_bx_rm1", "Bunch crossing: region -1; bunch crossing ; entries", 11,-5.5,5.5);
   theCSCCoPad_bx_rp1 = dbe_->book1D("copad_dg_bx_rp1", "Bunch crossing: region  1; bunch crossing ; entries", 11,-5.5,5.5);

   theCSCCoPad_zr_rm1 =  dbe_->book2D("copad_dg_zr_rm1", "Digi occupancy: region-1; globalZ [cm] ; globalR [cm] ", 200,-573,-564,55,130,240);
   theCSCCoPad_zr_rp1 =  dbe_->book2D("copad_dg_zr_rp1", "Digi occupancy: region 1; globalZ [cm] ; globalR [cm] ", 200, 564, 573,55,130,240);
}


GEMCSCCoPadDigiValidation::~GEMCSCCoPadDigiValidation() {
 

}


void GEMCSCCoPadDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{
  edm::Handle<GEMCSCPadDigiCollection> gem_digis;
  e.getByLabel(theInputTag, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMCSCCoPadDigiValidation") << "Cannot get pads by label "
                                       << theInputTag.encode();
  }
  for (GEMCSCPadDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {

    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = theGEMGeometry->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = theGEMGeometry->etaPartition(id);

    Short_t region = (Short_t) id.region();

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
      edm::LogInfo("GEMCSCCoPadDIGIValidation")<<"Global x "<<g_x<<"Global y "<<g_y<<"\n";	
      edm::LogInfo("GEMCSCCoPadDIGIValidation")<<"Global pad "<<pad<<"Global phi "<<g_phi<<std::endl;	
      edm::LogInfo("GEMCSCCoPadDIGIValidation")<<"Global bx "<<bx<<std::endl;	

      // fill hist
      if ( region== -1 ) {
                theCSCCoPad_zr_rm1->Fill(g_z,g_r);
	        theCSCCoPad_xy_rm1->Fill(g_x,g_y); 
            theCSCCoPad_phipad_rm1->Fill(g_phi , pad);
                   theCSCCoPad_rm1->Fill(pad);
                theCSCCoPad_bx_rm1->Fill(bx);
      }
      else if ( region == 1 ) {
                theCSCCoPad_zr_rp1->Fill(g_z,g_r);
                theCSCCoPad_xy_rp1->Fill(g_x,g_y);
            theCSCCoPad_phipad_rp1->Fill(g_phi, pad);
                   theCSCCoPad_rp1->Fill(pad);
                theCSCCoPad_bx_rp1->Fill(bx);

      }
      else {
        edm::LogInfo("GEMCSCCOPadDIGIValidation")<<"region : "<<region<<std::endl;
      }
   }
  }
}
