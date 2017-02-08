#include "Validation/MuonME0Validation/interface/ME0SegmentsValidation.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include <TMath.h>

ME0SegmentsValidation::ME0SegmentsValidation(const edm::ParameterSet& cfg):  ME0BaseValidation(cfg)
{
  InputTagToken_Segments = consumes<ME0SegmentCollection>(cfg.getParameter<edm::InputTag>("segmentInputLabel"));
}

void ME0SegmentsValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & Run, edm::EventSetup const & iSetup ) {
   //edm::ESHandle<ME0Geometry> hGeom;
   //iSetup.get<MuonGeometryRecord>().get(hGeom);
   //const ME0Geometry* ME0Geometry_ =( &*hGeom);

  LogDebug("MuonME0SegmentsValidation")<<"Info : Loading Geometry information\n";
  ibooker.setCurrentFolder("MuonME0RecHitsV/ME0SegmentsTask");

  unsigned int nregion  = 2;

  edm::LogInfo("MuonME0SegmentsValidation")<<"+++ Info : # of region : "<<nregion<<std::endl;

  LogDebug("MuonME0SegmentsValidation")<<"+++ Info : finish to get geometry information from ES.\n";

  me0_segment_chi2    = ibooker.book1D("me0_seg_ReducedChi2","#chi^{2}/ndof; #chi^{2}/ndof; # Segments",100,0,5);
  me0_segment_numRH   = ibooker.book1D("me0_seg_NumberRH","Number of fitted RecHits; # RecHits; entries",11,-0.5,10.5);
  //me0_segment_EtaRH   = ibooker.book1D("me0_specRH_globalEta","Fitted RecHits Eta Distribution; #eta; entries",200,-4.0,4.0);
  //me0_segment_PhiRH   = ibooker.book1D("me0_specRH_globalPhi","Fitted RecHits Phi Distribution; #eta; entries",18,-3.14,3.14);
  me0_segment_time    = ibooker.book1D("me0_seg_time","Segment Timing; ns; entries",40,15,25);
  me0_segment_timeErr = ibooker.book1D("me0_seg_timErr","Segment Timing Error; ns; entries",50,0,0.5);

  for( unsigned int region_num = 0 ; region_num < nregion ; region_num++ ) {
      me0_specRH_zr[region_num] = BookHistZR(ibooker,"me0_specRH_tot","Segment RecHits",region_num);
      for( unsigned int layer_num = 0 ; layer_num < 6 ; layer_num++) {
          //me0_strip_dg_zr[region_num][layer_num] = BookHistZR(ibooker,"me0_strip_dg","SimHit",region_num,layer_num);
          me0_specRH_xy[region_num][layer_num] = BookHistXY(ibooker,"me0_specRH","Segment RecHits",region_num,layer_num);
          //me0_rh_xy_Muon[region_num][layer_num] = BookHistXY(ibooker,"me0_rh","RecHit Muon",region_num,layer_num);

          std::string histo_name_DeltaX = std::string("me0_specRH_DeltaX_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string histo_name_DeltaY = std::string("me0_specRH_DeltaY_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string histo_label_DeltaX = "Segment RecHits Delta X : region"+regionLabel[region_num]+
                " layer "+layerLabel[layer_num]+" "+" ; x_{SimHit} - x_{Segment RecHits} ; entries";
          std::string histo_label_DeltaY = "Segment RecHits Delta Y : region"+regionLabel[region_num]+
                " layer "+layerLabel[layer_num]+" "+" ; y_{SimHit} - y_{Segment RecHit} ; entries";

          me0_specRH_DeltaX[region_num][layer_num] = ibooker.book1D(histo_name_DeltaX.c_str(), histo_label_DeltaX.c_str(),100,-10,10);
          me0_specRH_DeltaY[region_num][layer_num] = ibooker.book1D(histo_name_DeltaY.c_str(), histo_label_DeltaY.c_str(),100,-10,10);

          std::string histo_name_PullX = std::string("me0_specRH_PullX_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string histo_name_PullY = std::string("me0_specRH_PullY_r")+regionLabel[region_num]+"_l"+layerLabel[layer_num];
          std::string histo_label_PullX = "Segment RecHits Pull X : region"+regionLabel[region_num]+
                " layer "+layerLabel[layer_num]+" "+" ; #frac{x_{SimHit} - x_{Segment RecHit}}{#sigma_{x,RecHit}} ; entries";
          std::string histo_label_PullY = "Segment RecHits Pull Y : region"+regionLabel[region_num]+
                " layer "+layerLabel[layer_num]+" "+" ; #frac{y_{SimHit} - y_{Segment RecHit}}{#sigma_{y,RecHit}} ; entries";

          me0_specRH_PullX[region_num][layer_num] = ibooker.book1D(histo_name_PullX.c_str(), histo_label_DeltaX.c_str(),100,-10,10);
          me0_specRH_PullY[region_num][layer_num] = ibooker.book1D(histo_name_PullY.c_str(), histo_label_DeltaY.c_str(),100,-10,10);
      }
  }
}


ME0SegmentsValidation::~ME0SegmentsValidation() {
}


void ME0SegmentsValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup& iSetup)
{
 edm::ESHandle<ME0Geometry> hGeom;
 iSetup.get<MuonGeometryRecord>().get(hGeom);
 const ME0Geometry* ME0Geometry_ =( &*hGeom);

  edm::Handle<ME0SegmentCollection> ME0Segments;
  e.getByToken(InputTagToken_Segments, ME0Segments);

  if (!ME0Segments.isValid() ) {
    edm::LogError("ME0SegmentsValidation") << "Cannot get ME0RecHits/ME0Segments by Token InputTagToken";
    return ;
  }



 for (auto me0s = ME0Segments->begin(); me0s != ME0Segments->end(); me0s++) {

   // The ME0 Ensamble DetId refers to layer = 1
   ME0DetId id = me0s->me0DetId();
   //std::cout <<" Original ME0DetID "<<id<<std::endl;
   auto chamber = ME0Geometry_->chamber(id);
   //std::cout <<"Global Segment Position "<< chamber->toGlobal(me0s->localPosition())<<std::endl;
   auto segLP = me0s->localPosition();
   auto segLD = me0s->localDirection();
   //std::cout <<" Global Direction theta = "<<segLD.theta()<<" phi="<<segLD.phi()<<std::endl;
   auto me0rhs = me0s->specificRecHits();
   //std::cout <<"ME0 Ensamble Det Id "<<id<<" Number of RecHits "<<me0rhs.size()<<std::endl;

//   Int_t detId = id;
//   Float_t localX = segLP.x();
//   Float_t localY = segLP.y();
//   Float_t localZ = segLP.z();
//   Float_t dirTheta = segLD.theta();
//   Float_t dirPhi = segLD.phi();
   Short_t numberRH = me0rhs.size();
   Float_t chi2 = (Float_t) me0s->chi2();
   Float_t ndof = me0s->degreesOfFreedom();
   Double_t time = me0s->time();   
   Double_t timeErr = me0s->timeErr();   

   Float_t reducedChi2 = chi2/ndof;

   me0_segment_chi2->Fill(reducedChi2);
   me0_segment_numRH->Fill(numberRH);
   me0_segment_time->Fill(time);
   me0_segment_timeErr->Fill(timeErr);

   for (auto rh = me0rhs.begin(); rh!= me0rhs.end(); rh++)
   {

     auto me0id = rh->me0Id();
     auto rhr = ME0Geometry_->etaPartition(me0id);
     auto rhLP = rh->localPosition();
     auto erhLEP = rh->localPositionError();
     auto rhGP = rhr->toGlobal(rhLP);
     auto rhLPSegm = chamber->toLocal(rhGP);
     float xe = segLP.x()+segLD.x()*rhLPSegm.z()/segLD.z();
     float ye = segLP.y()+segLD.y()*rhLPSegm.z()/segLD.z();
     float ze = rhLPSegm.z();
     LocalPoint extrPoint(xe,ye,ze); // in segment rest frame
     auto extSegm = rhr->toLocal(chamber->toGlobal(extrPoint)); // in layer restframe

//     Int_t detId = me0id;

     Short_t region = me0id.region();
//     Short_t station = 0;
//     Short_t ring = 0;
     Short_t layer = me0id.layer();
//     Short_t chamber = me0id.chamber();
//     Short_t chamber = me0id.chamber();

     Float_t x = rhLP.x();
     Float_t xErr = erhLEP.xx();
     Float_t y = rhLP.y();
     Float_t yErr = erhLEP.yy();

     Float_t globalR = rhGP.perp();
     Float_t globalX = rhGP.x();
     Float_t globalY = rhGP.y();
     Float_t globalZ = rhGP.z();
//     Float_t globalEta = rhGP.eta();
//     Float_t globalPhi = rhGP.phi();

     Float_t xExt = extSegm.x();
     Float_t yExt = extSegm.y();

     Float_t pull_x = (x - xExt) / sqrt(xErr);
     Float_t pull_y = (y - yExt) / sqrt(yErr);

     int region_num=0 ;
     if ( region ==-1 ) region_num = 0 ;
     else if ( region==1) region_num = 1;
     int layer_num = layer-1;

     me0_specRH_xy[region_num][layer_num]->Fill(globalX,globalY);
     me0_specRH_zr[region_num]->Fill(globalZ,globalR);

     me0_specRH_DeltaX[region_num][layer_num]->Fill(x - xExt);
     me0_specRH_DeltaY[region_num][layer_num]->Fill(y - yExt);
     me0_specRH_PullX[region_num][layer_num]->Fill(pull_x);
     me0_specRH_PullY[region_num][layer_num]->Fill(pull_y);


  }
}

}
