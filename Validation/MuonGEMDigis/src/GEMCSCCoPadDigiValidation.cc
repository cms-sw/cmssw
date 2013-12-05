#include "Validation/MuonGEMDigis/interface/GEMCSCCoPadDigiValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>

const int nPads = 96;

GEMCSCCoPadDigiValidation::GEMCSCCoPadDigiValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag)
:  GEMBaseValidation(dbe, inputTag)
,  theCSCCoPad_xy_rm1( dbe_->book2D("copad_dg_xy_rm1", "Digi occupancy: region -1;globalX [cm]; globalY[cm]", 260,-260,260,260,-260,260))
,  theCSCCoPad_xy_rp1( dbe_->book2D("copad_dg_xy_rp1", "Digi occupancy: region  1;globalX [cm]; globalY[cm]", 260,-260,260,260,-260,260))

,  theCSCCoPad_phipad_rm1( dbe_->book2D("copad_dg_phipad_rm1", "Digi occupancy: region -1; pad number; phi [rad]", nPads/2,0,nPads,280,-TMath::Pi(),TMath::Pi()))
,  theCSCCoPad_phipad_rp1( dbe_->book2D("copad_dg_phipad_rp1", "Digi occupancy: region  1; pad number; phi [rad]", nPads/2,0,nPads,280,-TMath::Pi(),TMath::Pi()))


,  theCSCCoPad_rm1( dbe_->book1D("copad_dg_rm1", "Digi occupancy per stip number: region -1;pad number; entries", nPads,0.5,nPads+0.5))
,  theCSCCoPad_rp1( dbe_->book1D("copad_dg_rp1", "Digi occupancy per stip number: region  1;pad number; entries", nPads,0.5,nPads+0.5))


,  theCSCCoPad_bx_rm1( dbe_->book1D("copad_dg_bx_rm1", "Bunch crossing: region -1; bunch crossing ; entries", 11,-5.5,5.5))
,  theCSCCoPad_bx_rp1( dbe_->book1D("copad_dg_bx_rp1", "Bunch crossing: region  1; bunch crossing ; entries", 11,-5.5,5.5))


,  theCSCCoPad_zr_rm1( dbe_->book2D("copad_dg_zr_rm1", "Digi occupancy: region-1; globalZ [cm] ; globalR [cm] ", 200,-573,-564,55,130,240))
,  theCSCCoPad_zr_rp1( dbe_->book2D("copad_dg_zr_rp1", "Digi occupancy: region 1; globalZ [cm] ; globalR [cm] ", 200, 564, 573,55,130,240))
{

}


GEMCSCCoPadDigiValidation::~GEMCSCCoPadDigiValidation() {
 

}


void GEMCSCCoPadDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{
  edm::Handle<GEMCSCPadDigiCollection> gem_digis;
  e.getByLabel(theInputTag, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMDigiValidation") << "Cannot get pads by label "
                                       << theInputTag.encode();
  }
  //std::cout<<" Hello "<<std::endl;

  for (GEMCSCPadDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {

    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = theGEMGeometry->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = theGEMGeometry->etaPartition(id);

//    Int_t detId = id();
    Short_t region = (Short_t) id.region();
//    Short_t ring = (Short_t) id.ring();
//    Short_t station = (Short_t) id.station();
//    Short_t layer = (Short_t) id.layer();
//    Short_t chamber = (Short_t) id.chamber();
//    Short_t id_roll = (Short_t) id.roll();

    GEMCSCPadDigiCollection::const_iterator digiItr;
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      Short_t pad = (Short_t) digiItr->pad();
      Short_t bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfPad(digiItr->pad());
//      Float_t x = (Float_t) lp.x();
//      Float_t y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t) gp.perp();
//      Float_t g_eta = (Float_t) gp.eta();
      Float_t g_phi = (Float_t) gp.phi();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
      Float_t g_z = (Float_t) gp.z();
      edm::LogInfo("CSCCoPadDIGIValidation")<<"Global x "<<g_x<<"Global y "<<g_y<<"\n";	
      edm::LogInfo("CSCCoPadDIGIValidation")<<"Global pad "<<pad<<"Global phi "<<g_phi<<std::endl;	
      edm::LogInfo("CSCCoPadDIGIValidation")<<"Global bx "<<bx<<std::endl;	

      // fill hist
      if ( region== -1 ) {
                theCSCCoPad_zr_rm1->Fill(g_z,g_r);
	        theCSCCoPad_xy_rm1->Fill(g_x,g_y); 
            theCSCCoPad_phipad_rm1->Fill(pad,g_phi);
                   theCSCCoPad_rm1->Fill(pad);
                theCSCCoPad_bx_rm1->Fill(bx);
      }
      else if ( region == 1 ) {
                theCSCCoPad_zr_rp1->Fill(g_z,g_r);
                theCSCCoPad_xy_rp1->Fill(g_x,g_y);
            theCSCCoPad_phipad_rp1->Fill(pad,g_phi);
                   theCSCCoPad_rp1->Fill(pad);
                theCSCCoPad_bx_rp1->Fill(bx);

      }
      else {
        std::cout<<"region : "<<region<<std::endl;
      }
   }
  }
}
