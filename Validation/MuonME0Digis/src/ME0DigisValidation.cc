#include "Validation/MuonME0Digis/interface/ME0DigisValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <TMath.h>

ME0DigisValidation::ME0DigisValidation(const edm::ParameterSet& cfg):  ME0BaseValidation(cfg)
{
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagToken_Digi = consumes<ME0DigiPreRecoCollection>(cfg.getParameter<edm::InputTag>("digiInputLabel"));
}

void ME0DigisValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
   //edm::ESHandle<ME0Geometry> hGeom;
   //iSetup.get<MuonGeometryRecord>().get(hGeom);
   //const ME0Geometry* ME0Geometry_ =( &*hGeom);
  
  LogDebug("MuonME0DigisValidation")<<"Info : Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonME0DigisV/ME0DigisTask");

  unsigned int nregion  = 2; 

  edm::LogInfo("MuonME0DigisValidation")<<"+++ Info : # of region : "<<nregion<<std::endl;

  LogDebug("MuonME0DigisValidation")<<"+++ Info : finish to get geometry information from ES.\n";


  for( unsigned int region_num = 0 ; region_num < nregion ; region_num++ ) {
      me0_strip_dg_zr_tot[region_num] = BookHistZR(ibooker,"me0_strip_dg_tot","Digi",region_num);
      me0_strip_dg_zr_tot_Muon[region_num] = BookHistZR(ibooker,"me0_strip_dg_tot","Digi Muon",region_num);
      for( unsigned int layer_num = 0 ; layer_num < 6 ; layer_num++) {
          //me0_strip_dg_zr[region_num][layer_num] = BookHistZR(ibooker,"me0_strip_dg","SimHit",region_num,layer_num);
          me0_strip_dg_xy[region_num][layer_num] = BookHistXY(ibooker,"me0_strip_dg","Digi",region_num,layer_num);
          me0_strip_dg_xy_Muon[region_num][layer_num] = BookHistXY(ibooker,"me0_strip_dg","Digi Muon",region_num,layer_num);

//          std::string histo_name_DeltaX = std::string("me0_strip_dg_DeltaX_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
//          std::string histo_name_DeltaY = std::string("me0_strip_dg_DeltaY_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
//          std::string histo_label_DeltaX = "DIGI Delta X : region"+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" "+" ; x_{SimHit} - x_{Digi} ; entries";
//          std::string histo_label_DeltaY = "DIGI Delta Y : region"+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" "+" ; y_{SimHit} - y_{Digi} ; entries";
//
//          me0_strip_digi_DeltaX[region_num][layer_num] = ibooker.book1D(histo_name_DeltaX.c_str(), histo_label_DeltaX.c_str(),100,-10,10);
//          me0_strip_digi_DeltaY[region_num][layer_num] = ibooker.book1D(histo_name_DeltaY.c_str(), histo_label_DeltaY.c_str(),100,-10,10);
      }
  }
}


ME0DigisValidation::~ME0DigisValidation() {
}


void ME0DigisValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup& iSetup)
{
 edm::ESHandle<ME0Geometry> hGeom;
 iSetup.get<MuonGeometryRecord>().get(hGeom);
 const ME0Geometry* ME0Geometry_ =( &*hGeom);
  edm::Handle<edm::PSimHitContainer> ME0Hits;
  e.getByToken(InputTagToken_, ME0Hits);

  edm::Handle<ME0DigiPreRecoCollection> ME0Digis;
  e.getByToken(InputTagToken_Digi, ME0Digis);

  if (!ME0Hits.isValid() | !ME0Digis.isValid() ) {
    edm::LogError("ME0DigisValidation") << "Cannot get ME0Hits/ME0Digis by Token simInputTagToken";
    return ;
  }



//  for (auto hits=ME0Hits->begin(); hits!=ME0Hits->end(); hits++) {

//    const ME0DetId id(hits->detUnitId());
//    Int_t region = id.region();
//    Int_t layer = id.layer();



//    //Int_t even_odd = id.chamber()%2;
//    if ( ME0Geometry_->idToDet(hits->detUnitId()) == nullptr) {
//      std::cout<<"simHit did not matched with GEMGeometry."<<std::endl;
//      continue;
//    }
//    const LocalPoint p0(0., 0., 0.);
//    const GlobalPoint Gp0(ME0Geometry_->idToDet(hits->detUnitId())->surface().toGlobal(p0));
//    const LocalPoint hitLP(hits->localPosition());
    
//    const GlobalPoint hitGP(ME0Geometry_->idToDet(hits->detUnitId())->surface().toGlobal(hitLP));
//    Float_t g_r = hitGP.perp();
//    Float_t g_x = hitGP.x();
//    Float_t g_y = hitGP.y();
//    Float_t g_z = hitGP.z();

//    const LocalPoint hitEP(hits->entryPoint());

////move these here in order to use a GEMDetId - layers, station...
//    Float_t energyLoss = hits->energyLoss();
//    Float_t timeOfFlight = hits->timeOfFlight();
//    if (abs(hits-> particleType()) == 13)
//    {
//      timeOfFlightMuon = hits->timeOfFlight();
//      energyLossMuon = hits->energyLoss();
////fill histos for Muons only
//      me0_sh_tofMu[(int)(region/2.+0.5)][layer-1]->Fill(timeOfFlightMuon);
//      me0_sh_elossMu[(int)(region/2.+0.5)][layer-1]->Fill(energyLossMuon*1.e9);
//    }
    
//      // fill hist
//    // First, fill variable has no condition.
//    me0_sh_zr[(int)(region/2.+0.5)][layer-1]->Fill(g_z,g_r);
//    me0_sh_xy[(int)(region/2.+0.5)][layer-1]->Fill(g_x,g_y);
//    me0_sh_tof[(int)(region/2.+0.5)][layer-1]->Fill(timeOfFlight);
//    me0_sh_eloss[(int)(region/2.+0.5)][layer-1]->Fill(energyLoss*1.e9);

//   }
  for (ME0DigiPreRecoCollection::DigiRangeIterator cItr=ME0Digis->begin(); cItr!=ME0Digis->end(); cItr++) {
    ME0DetId id = (*cItr).first;

    const GeomDet* gdet = ME0Geometry_->idToDet(id);
    if ( gdet == nullptr) {
      std::cout<<"Getting DetId failed. Discard this gem strip hit.Maybe it comes from unmatched geometry."<<std::endl;
      continue;
    }
    const BoundPlane & surface = gdet->surface();

    Short_t region = (Short_t) id.region();
    Short_t layer = (Short_t) id.layer();
//    Short_t station = (Short_t) id.station();
//    Short_t chamber = (Short_t) id.chamber();
//    Short_t nroll = (Short_t) id.roll();

    ME0DigiPreRecoCollection::const_iterator digiItr;
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      Short_t particleType = digiItr->pdgid();
//      Short_t strip = (Short_t) digiItr->strip();
//      Short_t bx = (Short_t) digiItr->bx();
      LocalPoint lp(digiItr->x(), digiItr->y(), 0);

      GlobalPoint gp = surface.toGlobal(lp);

      Float_t g_r = (Float_t) gp.perp();
//      Float_t g_eta = (Float_t) gp.eta();
//      Float_t g_phi = (Float_t) gp.phi();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
      Float_t g_z = (Float_t) gp.z();
      // fill hist
      int region_num=0 ;
      if ( region ==-1 ) region_num = 0 ;
      else if ( region==1) region_num = 1;
      int layer_num = layer-1;

      if ( abs(particleType) == 13) {
        me0_strip_dg_zr_tot_Muon[region_num]->Fill(g_z,g_r);
        me0_strip_dg_xy_Muon[region_num][layer_num]->Fill(g_x,g_y);
      }
      else {
        me0_strip_dg_zr_tot[region_num]->Fill(g_z,g_r);
        me0_strip_dg_xy[region_num][layer_num]->Fill(g_x,g_y);
      }
    }
}

}

