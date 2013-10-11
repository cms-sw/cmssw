#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"

GEMStripDigiValidation::GEMStripDigiValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag)
:GEMBaseValidation(dbe, inputTag),
  theStrip_XY_rm1_l1( dbe_->book2D("strip_dg_xy_rm1_l1", "Digi occupancy: region -1, layer1", 260,-260,260,260,-260,260)),
  theStrip_XY_rm1_l2( dbe_->book2D("strip_dg_xy_rm1_l2", "Digi occupancy: region -1, layer2", 260,-260,260,260,-260,260))
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
//      Short_t strip = (Short_t) digiItr->strip();
//      Short_t bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());
//      Float_t x = (Float_t) lp.x();
//      Float_t y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
//      Float_t g_r = (Float_t) gp.perp();
//      Float_t g_eta = (Float_t) gp.eta();
//      Float_t g_phi = (Float_t) gp.phi();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
//      Float_t g_z = (Float_t) gp.z();

      // fill hist
      if ( region== -1 ) {
	if ( layer == 1 ) {
	  theStrip_XY_rm1_l1->Fill(g_x,g_y); 
	  std::cout<<"Global x "<<g_x<<"Global y "<<g_y<<std::endl;	
        }
        else if ( layer ==2 ) {
          theStrip_XY_rm1_l2->Fill(g_x,g_y);
	  std::cout<<"Global x "<<g_x<<"Global y "<<g_y<<std::endl;	
        }
        else {
          std::cout<<"layer : "<<layer<<std::endl;
	}
       }


    }
  }
}
