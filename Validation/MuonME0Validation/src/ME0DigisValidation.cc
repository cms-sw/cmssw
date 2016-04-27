#include "Validation/MuonME0Validation/interface/ME0DigisValidation.h"
#include <TMath.h>

ME0DigisValidation::ME0DigisValidation(const edm::ParameterSet& cfg):  ME0BaseValidation(cfg)
{
  InputTagToken_ = consumes<edm::PSimHitContainer>(cfg.getParameter<edm::InputTag>("simInputLabel"));
  InputTagToken_Digi = consumes<ME0DigiPreRecoCollection>(cfg.getParameter<edm::InputTag>("digiInputLabel"));
}

void ME0DigisValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {

  LogDebug("MuonME0DigisValidation")<<"Info : Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonME0DigisV/ME0DigisTask");

  unsigned int nregion  = 2;

  edm::LogInfo("MuonME0DigisValidation")<<"+++ Info : # of region : "<<nregion<<std::endl;

  LogDebug("MuonME0DigisValidation")<<"+++ Info : finish to get geometry information from ES.\n";


  for( unsigned int region_num = 0 ; region_num < nregion ; region_num++ ) {
      me0_strip_dg_zr_tot[region_num] = BookHistZR(ibooker,"me0_strip_dg_tot","Digi",region_num);
      me0_strip_dg_zr_tot_Muon[region_num] = BookHistZR(ibooker,"me0_strip_dg_tot","Digi Muon",region_num);
      for( unsigned int layer_num = 0 ; layer_num < 6 ; layer_num++) {

          me0_strip_dg_xy[region_num][layer_num] = BookHistXY(ibooker,"me0_strip_dg","Digi",region_num,layer_num);
          me0_strip_dg_xy_Muon[region_num][layer_num] = BookHistXY(ibooker,"me0_strip_dg","Digi Muon",region_num,layer_num);
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

    ME0DigiPreRecoCollection::const_iterator digiItr;
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      Short_t particleType = digiItr->pdgid();
      LocalPoint lp(digiItr->x(), digiItr->y(), 0);

      GlobalPoint gp = surface.toGlobal(lp);

      Float_t g_r = (Float_t) gp.perp();
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
