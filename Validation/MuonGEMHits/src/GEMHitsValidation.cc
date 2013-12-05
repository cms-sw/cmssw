#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>



GEMHitsValidation::GEMHitsValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag)
:  GEMBaseValidation(dbe, inputTag)
{
  gem_sh_xy_rm1_l1 = dbe_->book2D("gem_sh_xy_rm1_l1", "SimHit occupancy : region -1, layer1;globalX [cm]; globalY[cm]", 100,-260,260,100,-260,260);
  gem_sh_xy_rm1_l2 = dbe_->book2D("gem_sh_xy_rm1_l2", "SimHit occupancy : region -1, layer2;globalX [cm]; globalY[cm]", 100,-260,260,100,-260,260);
  gem_sh_xy_rp1_l1 = dbe_->book2D("gem_sh_xy_rp1_l1", "SimHit occupancy : region  1, layer1;globalX [cm]; globalY[cm]", 100,-260,260,100,-260,260);
  gem_sh_xy_rp1_l2 = dbe_->book2D("gem_sh_xy_rp1_l2", "SimHit occupancy : region  1, layer2;globalX [cm]; globalY[cm]", 100,-260,260,100,-260,260);



  gem_sh_zr_rm1 =  dbe_->book2D("gem_sh_zr_rm1", "SimHit occupancy: region-1; globalZ [cm] ; globalR [cm] ", 200,-573,-564,110,130,240);
  gem_sh_zr_rp1 =  dbe_->book2D("gem_sh_zr_rp1", "SimHit occupancy: region 1; globalZ [cm] ; globalR [cm] ", 200, 564, 573,110,130,240);

}


GEMHitsValidation::~GEMHitsValidation() {
}


void GEMHitsValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{

  edm::Handle<edm::PSimHitContainer> GEMHits;
  e.getByLabel(theInputTag, GEMHits);
  if (!GEMHits.isValid()) {
    edm::LogError("GEMHitsValidation") << "Cannot get GEMHits by label "
                                       << theInputTag.encode();
  }

  //Int_t eventNumber = e.id().event();
  for (auto hits=GEMHits->begin(); hits!=GEMHits->end(); hits++) {
    //Int_t particleType = hits->particleType();
    //Float_t lx = hits->localPosition().x();
    //Float_t ly = hits->localPosition().y();
    //Float_t energyLoss = hits->energyLoss();
    //Float_t pabs = hits->pabs();
    //Float_t timeOfFlight = hits->timeOfFlight();
    
    const GEMDetId id(hits->detUnitId());
    
    Int_t region = id.region();
    //Int_t ring = id.ring();
    //Int_t station = id.station();
    Int_t layer = id.layer();
    //Int_t chamber = id.chamber();
    //Int_t roll = id.roll();

    const LocalPoint p0(0., 0., 0.);
    const GlobalPoint Gp0(theGEMGeometry->idToDet(hits->detUnitId())->surface().toGlobal(p0));

    //Float_t Phi_0 = Gp0.phi();
    //Float_t R_0 = Gp0.perp();
    //Float_t DeltaPhi = atan(-1*id.region()*pow(-1,id.chamber())*hits->localPosition().x()/(Gp0.perp() + hits->localPosition().y()));
 
    const LocalPoint hitLP(hits->localPosition());
    const GlobalPoint hitGP(theGEMGeometry->idToDet(hits->detUnitId())->surface().toGlobal(hitLP));
    Float_t g_r = hitGP.perp();
    //Float_t g_eta = hitGP.eta();
    //Float_t g_phi = hitGP.phi();
    Float_t g_x = hitGP.x();
    Float_t g_y = hitGP.y();
    Float_t g_z = hitGP.z();

    const LocalPoint hitEP(hits->entryPoint());
    //Int_t strip = theGEMGeometry->etaPartition(hits->detUnitId())->strip(hitEP);

      // fill hist
      if ( region== -1 ) {
        gem_sh_zr_rm1->Fill(g_z,g_r);
        std::cout<<"Z : "<<g_z<<"  Rho : "<<g_r<<std::endl;
	if ( layer == 1 ) {
	  gem_sh_xy_rm1_l1->Fill(g_x,g_y); 
        }
        else if ( layer ==2 ) {
          gem_sh_xy_rm1_l2->Fill(g_x,g_y);
        }
        else {
          //std::cout<<"layer : "<<layer<<std::endl;
	}
      }
      else if ( region == 1 ) {
        gem_sh_zr_rp1->Fill(g_z,g_r);
        if ( layer == 1 ) {
          gem_sh_xy_rp1_l1->Fill(g_x,g_y);
        }
        else if ( layer == 2 ) {
          gem_sh_xy_rp1_l2->Fill(g_x,g_y);
        }
        else {
          //std::cout<<"layer : "<<layer<<std::endl;
        }
      }
      else {
        //std::cout<<"region : "<<region<<std::endl;
      }
   }
}
