#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
GEMStripDigiValidation::GEMStripDigiValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag)
:  GEMBaseValidation(dbe, inputTag)
,  theStrip_xy_rm1_l1( dbe_->book2D("strip_dg_xy_rm1_l1", "Digi occupancy: region -1, layer1;globalX [cm]; globalY[cm]", 260,-260,260,260,-260,260))
,  theStrip_xy_rm1_l2( dbe_->book2D("strip_dg_xy_rm1_l2", "Digi occupancy: region -1, layer2;globalX [cm]; globalY[cm]", 260,-260,260,260,-260,260))
,  theStrip_xy_rp1_l1( dbe_->book2D("strip_dg_xy_rp1_l1", "Digi occupancy: region  1, layer1;globalX [cm]; globalY[cm]", 260,-260,260,260,-260,260))
,  theStrip_xy_rp1_l2( dbe_->book2D("strip_dg_xy_rp1_l2", "Digi occupancy: region  1, layer2;globalX [cm]; globalY[cm]", 260,-260,260,260,-260,260))

,  theStrip_phistrip_rm1_l1( dbe_->book2D("strip_dg_phistrip_rm1_l1", "Digi occupancy: region -1, layer1; strip number; phi [rad]", 192,0,384,280,-TMath::Pi(),TMath::Pi()))
,  theStrip_phistrip_rm1_l2( dbe_->book2D("strip_dg_phistrip_rm1_l2", "Digi occupancy: region -1, layer2; strip number; phi [rad]", 192,0,384,280,-TMath::Pi(),TMath::Pi()))
,  theStrip_phistrip_rp1_l1( dbe_->book2D("strip_dg_phistrip_rp1_l1", "Digi occupancy: region  1, layer1; strip number; phi [rad]", 192,0,384,280,-TMath::Pi(),TMath::Pi()))
,  theStrip_phistrip_rp1_l2( dbe_->book2D("strip_dg_phistrip_rp1_l2", "Digi occupancy: region  1, layer2; strip number; phi [rad]", 192,0,384,280,-TMath::Pi(),TMath::Pi()))


,  theStrip_rm1_l1( dbe_->book1D("strip_dg_rm1_l1", "Digi occupancy per stip number: region -1, layer1;strip number; entries", 384,0.5,384.5))
,  theStrip_rm1_l2( dbe_->book1D("strip_dg_rm1_l2", "Digi occupancy per stip number: region -1, layer2;strip number; entries", 384,0.5,384.5))
,  theStrip_rp1_l1( dbe_->book1D("strip_dg_rp1_l1", "Digi occupancy per stip number: region  1, layer1;strip number; entries", 384,0.5,384.5))
,  theStrip_rp1_l2( dbe_->book1D("strip_dg_rp1_l2", "Digi occupancy per stip number: region  1, layer2;strip number; entries", 384,0.5,384.5))


,  theStrip_bx_rm1_l1( dbe_->book1D("strip_dg_bx_rm1_l1", "Bunch crossing: region -1, layer1; bunch crossing ; entries", 11,-5.5,5.5))
,  theStrip_bx_rm1_l2( dbe_->book1D("strip_dg_bx_rm1_l2", "Bunch crossing: region -1, layer2; bunch crossing ; entries", 11,-5.5,5.5))
,  theStrip_bx_rp1_l1( dbe_->book1D("strip_dg_bx_rp1_l1", "Bunch crossing: region  1, layer1; bunch crossing ; entries", 11,-5.5,5.5))
,  theStrip_bx_rp1_l2( dbe_->book1D("strip_dg_bx_rp1_l2", "Bunch crossing: region  1, layer2; bunch crossing ; entries", 11,-5.5,5.5))


,  theStrip_zr_rm1( dbe_->book2D("strip_dg_zr_rm1", "Digi occupancy: region-1; globalZ [cm] ; globalR [cm] ", 200,-573,-564, 55,130,240))
,  theStrip_zr_rp1( dbe_->book2D("strip_dg_zr_rp1", "Digi occupancy: region 1; globalZ [cm] ; globalR [cm] ", 200,564,573,55,130,240))
{

}


GEMStripDigiValidation::~GEMStripDigiValidation() {
 

}


void GEMStripDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{
  edm::Handle<GEMDigiCollection> gem_digis;
  e.getByLabel(theInputTag, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMDigiValidation") << "Cannot get strips by label "
                                       << theInputTag.encode();
  }
  //std::cout<<" Hello "<<std::endl;

  for (GEMDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {

    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = theGEMGeometry->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = theGEMGeometry->etaPartition(id);

//    Int_t detId = id();
    Short_t region = (Short_t) id.region();
//    Short_t ring = (Short_t) id.ring();
//    Short_t station = (Short_t) id.station();
    Short_t layer = (Short_t) id.layer();
//    Short_t chamber = (Short_t) id.chamber();
//    Short_t id_roll = (Short_t) id.roll();

    GEMDigiCollection::const_iterator digiItr;
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      Short_t strip = (Short_t) digiItr->strip();
      Short_t bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());
//      Float_t x = (Float_t) lp.x();
//      Float_t y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t) gp.perp();
//      Float_t g_eta = (Float_t) gp.eta();
      Float_t g_phi = (Float_t) gp.phi();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
      Float_t g_z = (Float_t) gp.z();
      edm::LogInfo("StripDIGIValidation")<<"Global x "<<g_x<<"Global y "<<g_y<<"\n";	
      edm::LogInfo("StripDIGIValidation")<<"Global strip "<<strip<<"Global phi "<<g_phi<<std::endl;	
      edm::LogInfo("StripDIGIValidation")<<"Global bx "<<bx<<std::endl;	

      // fill hist
      if ( region== -1 ) {
                theStrip_zr_rm1->Fill(g_z,g_r);
	if ( layer == 1 ) {
	        theStrip_xy_rm1_l1->Fill(g_x,g_y); 
          theStrip_phistrip_rm1_l1->Fill(strip,g_phi);
                   theStrip_rm1_l1->Fill(strip);
                theStrip_bx_rm1_l1->Fill(bx);
        }
        else if ( layer ==2 ) {
                theStrip_xy_rm1_l2->Fill(g_x,g_y);
          theStrip_phistrip_rm1_l2->Fill(strip,g_phi);
                   theStrip_rm1_l2->Fill(strip);
                theStrip_bx_rm1_l2->Fill(bx);
        }
        else {
          edm::LogInfo("StripDIGIValidation")<<"layer : "<<layer<<std::endl;
	}
      }
      else if ( region == 1 ) {
                theStrip_zr_rp1->Fill(g_z,g_r);

        if ( layer == 1 ) {
                theStrip_xy_rp1_l1->Fill(g_x,g_y);
          theStrip_phistrip_rp1_l1->Fill(strip,g_phi);
                   theStrip_rp1_l1->Fill(strip);
                theStrip_bx_rp1_l1->Fill(bx);
        }
        else if ( layer == 2 ) {
                theStrip_xy_rp1_l2->Fill(g_x,g_y);
          theStrip_phistrip_rp1_l2->Fill(strip,g_phi);
                   theStrip_rp1_l2->Fill(strip);
                theStrip_bx_rp1_l2->Fill(bx);
        }
        else {
          edm::LogInfo("StripDIGIValidation")<<"layer : "<<layer<<std::endl;
        }
      }
      else {
        edm::LogInfo("StripDIGIValidation")<<"region : "<<region<<std::endl;
      }
   }
  }
}
