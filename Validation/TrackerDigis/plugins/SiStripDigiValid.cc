#include "Validation/TrackerDigis/interface/SiStripDigiValid.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

SiStripDigiValid::SiStripDigiValid(const edm::ParameterSet& ps)
  : dbe_(0)
  , runStandalone ( ps.getParameter<bool>("runStandalone")  )  
  , outputFile_( ps.getUntrackedParameter<std::string>( "outputFile", "stripdigihisto.root" ) )
  , edmDetSetVector_SiStripDigi_Token_( consumes< edm::DetSetVector<SiStripDigi> >( ps.getParameter<edm::InputTag>( "src" ) ) ) {

}

SiStripDigiValid::~SiStripDigiValid(){
}


void SiStripDigiValid::beginJob(){

}

void SiStripDigiValid::bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es){
  dbe_ = edm::Service<DQMStore>().operator->();

   if ( dbe_ ) {
     ibooker.setCurrentFolder("TrackerDigisV/TrackerDigis/Strip");

     for(int i = 0 ;i<3 ; i++) {
       Char_t histo[200];
       // Z Plus Side
       sprintf(histo,"adc_tib_layer1_extmodule%d_zp",i+1);
       meAdcTIBLayer1Extzp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer1_intmodule%d_zp",i+1);
       meAdcTIBLayer1Intzp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer2_extmodule%d_zp",i+1);
       meAdcTIBLayer2Extzp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer2_intmodule%d_zp",i+1);
       meAdcTIBLayer2Intzp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer3_extmodule%d_zp",i+1);
       meAdcTIBLayer3Extzp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer3_intmodule%d_zp",i+1);
       meAdcTIBLayer3Intzp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer4_extmodule%d_zp",i+1);
       meAdcTIBLayer4Extzp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer4_intmodule%d_zp",i+1);
       meAdcTIBLayer4Intzp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tib_layer1_extmodule%d_zp",i+1);
       meStripTIBLayer1Extzp_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer1_intmodule%d_zp",i+1);
       meStripTIBLayer1Intzp_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer2_extmodule%d_zp",i+1);
       meStripTIBLayer2Extzp_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer2_intmodule%d_zp",i+1);
       meStripTIBLayer2Intzp_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer3_extmodule%d_zp",i+1);
       meStripTIBLayer3Extzp_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer3_intmodule%d_zp",i+1);
       meStripTIBLayer3Intzp_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer4_extmodule%d_zp",i+1);
       meStripTIBLayer4Extzp_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer4_intmodule%d_zp",i+1);
       meStripTIBLayer4Intzp_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       //  Z Minus Side
       sprintf(histo,"adc_tib_layer1_extmodule%d_zm",i+1);
       meAdcTIBLayer1Extzm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer1_intmodule%d_zm",i+1);
       meAdcTIBLayer1Intzm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer2_extmodule%d_zm",i+1);
       meAdcTIBLayer2Extzm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer2_intmodule%d_zm",i+1);
       meAdcTIBLayer2Intzm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer3_extmodule%d_zm",i+1);
       meAdcTIBLayer3Extzm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer3_intmodule%d_zm",i+1);
       meAdcTIBLayer3Intzm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer4_extmodule%d_zm",i+1);
       meAdcTIBLayer4Extzm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tib_layer4_intmodule%d_zm",i+1);
       meAdcTIBLayer4Intzm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tib_layer1_extmodule%d_zm",i+1);
       meStripTIBLayer1Extzm_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer1_intmodule%d_zm",i+1);
       meStripTIBLayer1Intzm_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer2_extmodule%d_zm",i+1);
       meStripTIBLayer2Extzm_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer2_intmodule%d_zm",i+1);
       meStripTIBLayer2Intzm_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer3_extmodule%d_zm",i+1);
       meStripTIBLayer3Extzm_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer3_intmodule%d_zm",i+1);
       meStripTIBLayer3Intzm_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer4_extmodule%d_zm",i+1);
       meStripTIBLayer4Extzm_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tib_layer4_intmodule%d_zm",i+1);
       meStripTIBLayer4Intzm_[i] =  ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
   }

     for(int i = 0 ;i<6 ; i++) {
       Char_t histo[200];
       // Z Plus Side
       sprintf(histo,"adc_tob_layer1_module%d_zp",i+1);
       meAdcTOBLayer1zp_[i] = ibooker.book1D(histo,"Digis ADC",10,0.,300.);
       sprintf(histo,"strip_tob_layer1_module%d_zp",i+1);
       meStripTOBLayer1zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer2_module%d_zp",i+1);
       meAdcTOBLayer2zp_[i] = ibooker.book1D(histo,"Digis ADC",10,0.,300.);
       sprintf(histo,"strip_tob_layer2_module%d_zp",i+1);
       meStripTOBLayer2zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer3_module%d_zp",i+1);
       meAdcTOBLayer3zp_[i] = ibooker.book1D(histo,"Digis ADC",10,0.,300.);
       sprintf(histo,"strip_tob_layer3_module%d_zp",i+1);
       meStripTOBLayer3zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer4_module%d_zp",i+1);
       meAdcTOBLayer4zp_[i] = ibooker.book1D(histo,"Digis ADC",10,0.,300.);
       sprintf(histo,"strip_tob_layer4_module%d_zp",i+1);
       meStripTOBLayer4zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer5_module%d_zp",i+1);
       meAdcTOBLayer5zp_[i] = ibooker.book1D(histo,"Digis ADC",10,0.,300.);
       sprintf(histo,"strip_tob_layer5_module%d_zp",i+1);
       meStripTOBLayer5zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer6_module%d_zp",i+1);
       meAdcTOBLayer6zp_[i] = ibooker.book1D(histo,"Digis ADC",10,0.,300.);
       sprintf(histo,"strip_tob_layer6_module%d_zp",i+1);
       meStripTOBLayer6zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       // Z Minus Side
       sprintf(histo,"adc_tob_layer1_module%d_zm",i+1);
       meAdcTOBLayer1zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tob_layer1_module%d_zm",i+1);
       meStripTOBLayer1zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer2_module%d_zm",i+1);
       meAdcTOBLayer2zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tob_layer2_module%d_zm",i+1);
       meStripTOBLayer2zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer3_module%d_zm",i+1);
       meAdcTOBLayer3zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tob_layer3_module%d_zm",i+1);
       meStripTOBLayer3zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer4_module%d_zm",i+1);
       meAdcTOBLayer4zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tob_layer4_module%d_zm",i+1);
       meStripTOBLayer4zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer5_module%d_zm",i+1);
       meAdcTOBLayer5zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tob_layer5_module%d_zm",i+1);
       meStripTOBLayer5zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"adc_tob_layer6_module%d_zm",i+1);
       meAdcTOBLayer6zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tob_layer6_module%d_zm",i+1);
       meStripTOBLayer6zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
     }
 
     for(int i = 0 ;i<3 ; i++) {
       Char_t histo[200];
       // Z Plus Side
       sprintf(histo,"adc_tid_wheel1_ring%d_zp",i+1);
       meAdcTIDWheel1zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tid_wheel2_ring%d_zp",i+1);
       meAdcTIDWheel2zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tid_wheel3_ring%d_zp",i+1);
       meAdcTIDWheel3zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tid_wheel1_ring%d_zp",i+1);
       meStripTIDWheel1zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.); 
       sprintf(histo,"strip_tid_wheel2_ring%d_zp",i+1);
       meStripTIDWheel2zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tid_wheel3_ring%d_zp",i+1);
       meStripTIDWheel3zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       // Z minus Side
       sprintf(histo,"adc_tid_wheel1_ring%d_zm",i+1);
       meAdcTIDWheel1zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tid_wheel2_ring%d_zm",i+1);
       meAdcTIDWheel2zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tid_wheel3_ring%d_zm",i+1);
       meAdcTIDWheel3zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tid_wheel1_ring%d_zm",i+1);
       meStripTIDWheel1zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tid_wheel2_ring%d_zm",i+1);
       meStripTIDWheel2zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tid_wheel3_ring%d_zm",i+1);
       meStripTIDWheel3zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
     }
     
     for(int i = 0 ;i<7 ; i++) {
       Char_t histo[200];
       // Z Plus Side
       sprintf(histo,"adc_tec_wheel1_ring%d_zp",i+1);
       meAdcTECWheel1zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel2_ring%d_zp",i+1);
       meAdcTECWheel2zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel3_ring%d_zp",i+1);
       meAdcTECWheel3zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tec_wheel1_ring%d_zp",i+1);
       meStripTECWheel1zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel2_ring%d_zp",i+1);
       meStripTECWheel2zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel3_ring%d_zp",i+1);
       meStripTECWheel3zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);

       // Z Minus Side
       sprintf(histo,"adc_tec_wheel1_ring%d_zm",i+1);
       meAdcTECWheel1zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel2_ring%d_zm",i+1);
       meAdcTECWheel2zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel3_ring%d_zm",i+1);
       meAdcTECWheel3zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tec_wheel1_ring%d_zm",i+1);
       meStripTECWheel1zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel2_ring%d_zm",i+1);
       meStripTECWheel2zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel3_ring%d_zm",i+1);
       meStripTECWheel3zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
     }

     for(int i = 0 ;i<6 ; i++) {
       Char_t histo[200];
       // Z Plus Side
       sprintf(histo,"adc_tec_wheel4_ring%d_zp",i+1);
       meAdcTECWheel4zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel5_ring%d_zp",i+1);
       meAdcTECWheel5zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel6_ring%d_zp",i+1);
       meAdcTECWheel6zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tec_wheel4_ring%d_zp",i+1);
       meStripTECWheel4zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel5_ring%d_zp",i+1);
       meStripTECWheel5zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel6_ring%d_zp",i+1);
       meStripTECWheel6zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);

       // Z Minus Side
       sprintf(histo,"adc_tec_wheel4_ring%d_zm",i+1);
       meAdcTECWheel4zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel5_ring%d_zm",i+1);
       meAdcTECWheel5zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel6_ring%d_zm",i+1);
       meAdcTECWheel6zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tec_wheel4_ring%d_zm",i+1);
       meStripTECWheel4zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel5_ring%d_zm",i+1);
       meStripTECWheel5zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel6_ring%d_zm",i+1);
       meStripTECWheel6zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
     }

     for(int i = 0 ;i<5 ; i++) {
       Char_t histo[200];
       // Z Plus Side
       sprintf(histo,"adc_tec_wheel7_ring%d_zp",i+1);
       meAdcTECWheel7zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel8_ring%d_zp",i+1);
       meAdcTECWheel8zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tec_wheel7_ring%d_zp",i+1);
       meStripTECWheel7zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel8_ring%d_zp",i+1);
       meStripTECWheel8zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);

       // Z Minus Side
       sprintf(histo,"adc_tec_wheel7_ring%d_zm",i+1);
       meAdcTECWheel7zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"adc_tec_wheel8_ring%d_zm",i+1);
       meAdcTECWheel8zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tec_wheel7_ring%d_zm",i+1);
       meStripTECWheel7zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
       sprintf(histo,"strip_tec_wheel8_ring%d_zm",i+1);
       meStripTECWheel8zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
     }

     for(int i = 0 ;i<4 ; i++) {
       Char_t histo[200];
       // Z Plus Side
       sprintf(histo,"adc_tec_wheel9_ring%d_zp",i+1);
       meAdcTECWheel9zp_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tec_wheel9_ring%d_zp",i+1);
       meStripTECWheel9zp_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);

       // Z Minus Side
       sprintf(histo,"adc_tec_wheel9_ring%d_zm",i+1);
       meAdcTECWheel9zm_[i] = ibooker.book1D(histo,"Digis ADC",50,0.,300.);
       sprintf(histo,"strip_tec_wheel9_ring%d_zm",i+1);
       meStripTECWheel9zm_[i] = ibooker.book1D(histo,"Digis Strip Num.",200,0.,800.);
     }

     for(int i = 0 ;i<4 ; i++) {
       Char_t histo[200];
       sprintf(histo,"ndigi_tib_layer_%d_zm",i+1);
       meNDigiTIBLayerzm_[i] = ibooker.book1D(histo, "Digi Multiplicity",100,0.,500.);
       sprintf(histo,"ndigi_tib_layer_%d_zp",i+1); 
       meNDigiTIBLayerzp_[i] = ibooker.book1D(histo, "Digi Multiplicity",100,0.,500.);
     }

     for(int i = 0 ;i<6 ; i++) {
       Char_t histo[200];
       sprintf(histo,"ndigi_tob_layer_%d_zm",i+1);
       meNDigiTOBLayerzm_[i] = ibooker.book1D(histo, "Digi Multiplicity",100,0.,500.);
       sprintf(histo,"ndigi_tob_layer_%d_zp",i+1);
       meNDigiTOBLayerzp_[i] = ibooker.book1D(histo, "Digi Multiplicity",100,0.,500.);
     }

     for(int i = 0 ;i<3 ; i++) {
       Char_t histo[200];
       sprintf(histo,"ndigi_tid_wheel_%d_zm",i+1);
       meNDigiTIDWheelzm_[i] = ibooker.book1D(histo, "Digi Multiplicity",100,0.,500.);
       sprintf(histo,"ndigi_tid_wheel_%d_zp",i+1);
       meNDigiTIDWheelzp_[i] = ibooker.book1D(histo, "Digi Multiplicity",100,0.,500.);
     }

     for(int i = 0 ;i<9 ; i++) {
       Char_t histo[200];
       sprintf(histo,"ndigi_tec_wheel_%d_zm",i+1);
       meNDigiTECWheelzm_[i] = ibooker.book1D(histo, "Digi Multiplicity",100,0.,500.);
       sprintf(histo,"ndigi_tec_wheel_%d_zp",i+1);
       meNDigiTECWheelzp_[i] = ibooker.book1D(histo, "Digi Multiplicity",100,0.,500.);
     }
   }
}

void SiStripDigiValid::endJob() {
  if ( runStandalone && outputFile_.size() != 0 && dbe_ ){ dbe_->save(outputFile_);}

}


void SiStripDigiValid::analyze(const edm::Event& e, const edm::EventSetup& c){
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  c.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();



 int ndigilayertibzp[4];
 int ndigilayertibzm[4];
 
 for( int i = 0; i< 4; i++ ) {
    ndigilayertibzp[i] = 0;
    ndigilayertibzm[i] = 0;
 }

 int ndigilayertobzp[6];
 int ndigilayertobzm[6];

 for( int i = 0; i< 6; i++ ) {
    ndigilayertobzp[i] = 0;
    ndigilayertobzm[i] = 0;
 }

 int ndigiwheeltidzp[3];
 int ndigiwheeltidzm[3];

 for( int i = 0; i< 3; i++ ) {
   ndigiwheeltidzp[i] = 0;
   ndigiwheeltidzm[i] = 0;
 }

 int ndigiwheelteczp[9];
 int ndigiwheelteczm[9];

 for( int i = 0; i< 9; i++ ) {
   ndigiwheelteczp[i] = 0;
   ndigiwheelteczm[i] = 0;
 }



 //LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
 edm::ESHandle<TrackerGeometry> tracker;
 c.get<TrackerDigiGeometryRecord>().get( tracker );

 std::string digiProducer = "siStripDigis";
 edm::Handle<edm::DetSetVector<SiStripDigi> > stripDigis;
 e.getByToken( edmDetSetVector_SiStripDigi_Token_, stripDigis );
 edm::DetSetVector<SiStripDigi>::const_iterator DSViter = stripDigis->begin();
 for( ; DSViter != stripDigis->end(); DSViter++) {
         unsigned int id = DSViter->id;
         DetId  detId(id);
         edm::DetSet<SiStripDigi>::const_iterator  begin = DSViter->data.begin();
         edm::DetSet<SiStripDigi>::const_iterator  end   = DSViter->data.end();
         edm::DetSet<SiStripDigi>::const_iterator  iter;

        if(detId.subdetId()==StripSubdetector::TIB){
             
             for ( iter = begin ; iter != end; iter++ ) { // loop digis
               if( tTopo->tibStringInfo(id)[0] == 1) {
                 ++ndigilayertibzm[tTopo->tibLayer(id)-1];
                 if( tTopo->tibLayer(id) == 1 ) { 
                    if ( tTopo->tibStringInfo(id)[1] == 1 ) {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer1Intzm_[0] -> Fill((*iter).adc()); meStripTIBLayer1Intzm_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer1Intzm_[1] -> Fill((*iter).adc()); meStripTIBLayer1Intzm_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer1Intzm_[2] -> Fill((*iter).adc()); meStripTIBLayer1Intzm_[2] ->Fill((*iter).strip()); }
                    }else {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer1Extzm_[0] -> Fill((*iter).adc()); meStripTIBLayer1Extzm_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer1Extzm_[1] -> Fill((*iter).adc()); meStripTIBLayer1Extzm_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer1Extzm_[2] -> Fill((*iter).adc()); meStripTIBLayer1Extzm_[2] ->Fill((*iter).strip()); }
                   } 
                 }
                 if( tTopo->tibLayer(id) == 2 ) {
                    if ( tTopo->tibStringInfo(id)[1] == 1 ) {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer2Intzm_[0] -> Fill((*iter).adc()); meStripTIBLayer2Intzm_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer2Intzm_[1] -> Fill((*iter).adc()); meStripTIBLayer2Intzm_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer2Intzm_[2] -> Fill((*iter).adc()); meStripTIBLayer2Intzm_[2] ->Fill((*iter).strip()); }
                    }else {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer2Extzm_[0] -> Fill((*iter).adc()); meStripTIBLayer2Extzm_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer2Extzm_[1] -> Fill((*iter).adc()); meStripTIBLayer2Extzm_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer2Extzm_[2] -> Fill((*iter).adc()); meStripTIBLayer2Extzm_[2] ->Fill((*iter).strip()); }
                   }
                 }
                 if( tTopo->tibLayer(id) == 3 ) {
                    if ( tTopo->tibStringInfo(id)[1] == 1 ) {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer3Intzm_[0] -> Fill((*iter).adc()); meStripTIBLayer3Intzm_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer3Intzm_[1] -> Fill((*iter).adc()); meStripTIBLayer3Intzm_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer3Intzm_[2] -> Fill((*iter).adc()); meStripTIBLayer3Intzm_[2] ->Fill((*iter).strip()); }
                    }else {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer3Extzm_[0] -> Fill((*iter).adc()); meStripTIBLayer3Extzm_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer3Extzm_[1] -> Fill((*iter).adc()); meStripTIBLayer3Extzm_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer3Extzm_[2] -> Fill((*iter).adc()); meStripTIBLayer3Extzm_[2] ->Fill((*iter).strip()); }
                   }
                 }
                 if( tTopo->tibLayer(id) == 4 ) {
                    if ( tTopo->tibStringInfo(id)[1] == 1 ) {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer4Intzm_[0] -> Fill((*iter).adc()); meStripTIBLayer4Intzm_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer4Intzm_[1] -> Fill((*iter).adc()); meStripTIBLayer4Intzm_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer4Intzm_[2] -> Fill((*iter).adc()); meStripTIBLayer4Intzm_[2] ->Fill((*iter).strip()); }
                    }else {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer4Extzm_[0] -> Fill((*iter).adc()); meStripTIBLayer4Extzm_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer4Extzm_[1] -> Fill((*iter).adc()); meStripTIBLayer4Extzm_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer4Extzm_[2] -> Fill((*iter).adc()); meStripTIBLayer4Extzm_[2] ->Fill((*iter).strip()); }
                   }
                 }
               }else {
                 ++ndigilayertibzp[tTopo->tibLayer(id)-1];
                 if( tTopo->tibLayer(id) == 1 ) {
                    if ( tTopo->tibStringInfo(id)[1] == 1 ) {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer1Intzp_[0] -> Fill((*iter).adc()); meStripTIBLayer1Intzp_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer1Intzp_[1] -> Fill((*iter).adc()); meStripTIBLayer1Intzp_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer1Intzp_[2] -> Fill((*iter).adc()); meStripTIBLayer1Intzp_[2] ->Fill((*iter).strip()); }
                    }else {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer1Extzp_[0] -> Fill((*iter).adc()); meStripTIBLayer1Extzp_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer1Extzp_[1] -> Fill((*iter).adc()); meStripTIBLayer1Extzp_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer1Extzp_[2] -> Fill((*iter).adc()); meStripTIBLayer1Extzp_[2] ->Fill((*iter).strip()); }
                   } 
                 }
                 if( tTopo->tibLayer(id) == 2 ) {
                    if ( tTopo->tibStringInfo(id)[1] == 1 ) {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer2Intzp_[0] -> Fill((*iter).adc()); meStripTIBLayer2Intzp_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer2Intzp_[1] -> Fill((*iter).adc()); meStripTIBLayer2Intzp_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer2Intzp_[2] -> Fill((*iter).adc()); meStripTIBLayer2Intzp_[2] ->Fill((*iter).strip()); }
                    }else {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer2Extzp_[0] -> Fill((*iter).adc()); meStripTIBLayer2Extzp_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer2Extzp_[1] -> Fill((*iter).adc()); meStripTIBLayer2Extzp_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer2Extzp_[2] -> Fill((*iter).adc()); meStripTIBLayer2Extzp_[2] ->Fill((*iter).strip()); }
                   }
                 }
                 if( tTopo->tibLayer(id) == 3 ) {
                    if ( tTopo->tibStringInfo(id)[1] == 1 ) {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer3Intzp_[0] -> Fill((*iter).adc()); meStripTIBLayer3Intzp_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer3Intzp_[1] -> Fill((*iter).adc()); meStripTIBLayer3Intzp_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer3Intzp_[2] -> Fill((*iter).adc()); meStripTIBLayer3Intzp_[2] ->Fill((*iter).strip()); }
                    }else {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer3Extzp_[0] -> Fill((*iter).adc()); meStripTIBLayer3Extzp_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer3Extzp_[1] -> Fill((*iter).adc()); meStripTIBLayer3Extzp_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer3Extzp_[2] -> Fill((*iter).adc()); meStripTIBLayer3Extzp_[2] ->Fill((*iter).strip()); }
                   }
                 }
                 if( tTopo->tibLayer(id) == 4 ) {
                    if ( tTopo->tibStringInfo(id)[1] == 1) {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer4Intzp_[0] -> Fill((*iter).adc()); meStripTIBLayer4Intzp_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer4Intzp_[1] -> Fill((*iter).adc()); meStripTIBLayer4Intzp_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer4Intzp_[2] -> Fill((*iter).adc()); meStripTIBLayer4Intzp_[2] ->Fill((*iter).strip()); }
                    }else {
                       if( tTopo->tibModule(id) == 1 ) { meAdcTIBLayer4Extzp_[0] -> Fill((*iter).adc()); meStripTIBLayer4Extzp_[0] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 2 ) { meAdcTIBLayer4Extzp_[1] -> Fill((*iter).adc()); meStripTIBLayer4Extzp_[1] ->Fill((*iter).strip()); }
                       if( tTopo->tibModule(id) == 3 ) { meAdcTIBLayer4Extzp_[2] -> Fill((*iter).adc()); meStripTIBLayer4Extzp_[2] ->Fill((*iter).strip()); }
                   }
                 }

              }
            } 
        } 
        if(detId.subdetId()==StripSubdetector::TOB){
              
             for ( iter = begin ; iter != end; iter++ ) { // loop digis
               if( tTopo->tobRodInfo(id)[0] == 1) {
                 ++ndigilayertobzm[tTopo->tobLayer(id)-1];  
                 if( tTopo->tobLayer(id) == 1 ) { 
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer1zm_[0] -> Fill((*iter).adc()); meStripTOBLayer1zm_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer1zm_[1] -> Fill((*iter).adc()); meStripTOBLayer1zm_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer1zm_[2] -> Fill((*iter).adc()); meStripTOBLayer1zm_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer1zm_[3] -> Fill((*iter).adc()); meStripTOBLayer1zm_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer1zm_[4] -> Fill((*iter).adc()); meStripTOBLayer1zm_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer1zm_[5] -> Fill((*iter).adc()); meStripTOBLayer1zm_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 2 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer2zm_[0] -> Fill((*iter).adc()); meStripTOBLayer2zm_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer2zm_[1] -> Fill((*iter).adc()); meStripTOBLayer2zm_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer2zm_[2] -> Fill((*iter).adc()); meStripTOBLayer2zm_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer2zm_[3] -> Fill((*iter).adc()); meStripTOBLayer2zm_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer2zm_[4] -> Fill((*iter).adc()); meStripTOBLayer2zm_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer2zm_[5] -> Fill((*iter).adc()); meStripTOBLayer2zm_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 3 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer3zm_[0] -> Fill((*iter).adc()); meStripTOBLayer3zm_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer3zm_[1] -> Fill((*iter).adc()); meStripTOBLayer3zm_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer3zm_[2] -> Fill((*iter).adc()); meStripTOBLayer3zm_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer3zm_[3] -> Fill((*iter).adc()); meStripTOBLayer3zm_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer3zm_[4] -> Fill((*iter).adc()); meStripTOBLayer3zm_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer3zm_[5] -> Fill((*iter).adc()); meStripTOBLayer3zm_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 4 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer4zm_[0] -> Fill((*iter).adc()); meStripTOBLayer4zm_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer4zm_[1] -> Fill((*iter).adc()); meStripTOBLayer4zm_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer4zm_[2] -> Fill((*iter).adc()); meStripTOBLayer4zm_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer4zm_[3] -> Fill((*iter).adc()); meStripTOBLayer4zm_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer4zm_[4] -> Fill((*iter).adc()); meStripTOBLayer4zm_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer4zm_[5] -> Fill((*iter).adc()); meStripTOBLayer4zm_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 5 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer5zm_[0] -> Fill((*iter).adc()); meStripTOBLayer5zm_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer5zm_[1] -> Fill((*iter).adc()); meStripTOBLayer5zm_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer5zm_[2] -> Fill((*iter).adc()); meStripTOBLayer5zm_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer5zm_[3] -> Fill((*iter).adc()); meStripTOBLayer5zm_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer5zm_[4] -> Fill((*iter).adc()); meStripTOBLayer5zm_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer5zm_[5] -> Fill((*iter).adc()); meStripTOBLayer5zm_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 6 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer6zm_[0] -> Fill((*iter).adc()); meStripTOBLayer6zm_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer6zm_[1] -> Fill((*iter).adc()); meStripTOBLayer6zm_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer6zm_[2] -> Fill((*iter).adc()); meStripTOBLayer6zm_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer6zm_[3] -> Fill((*iter).adc()); meStripTOBLayer6zm_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer6zm_[4] -> Fill((*iter).adc()); meStripTOBLayer6zm_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer6zm_[5] -> Fill((*iter).adc()); meStripTOBLayer6zm_[5] ->Fill((*iter).strip()); }
                 }

               }else {
                 ++ndigilayertobzp[tTopo->tobLayer(id)-1];
                 if( tTopo->tobLayer(id) == 1 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer1zp_[0] -> Fill((*iter).adc()); meStripTOBLayer1zp_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer1zp_[1] -> Fill((*iter).adc()); meStripTOBLayer1zp_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer1zp_[2] -> Fill((*iter).adc()); meStripTOBLayer1zp_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer1zp_[3] -> Fill((*iter).adc()); meStripTOBLayer1zp_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer1zp_[4] -> Fill((*iter).adc()); meStripTOBLayer1zp_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer1zp_[5] -> Fill((*iter).adc()); meStripTOBLayer1zp_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 2 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer2zp_[0] -> Fill((*iter).adc()); meStripTOBLayer2zp_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer2zp_[1] -> Fill((*iter).adc()); meStripTOBLayer2zp_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer2zp_[2] -> Fill((*iter).adc()); meStripTOBLayer2zp_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer2zp_[3] -> Fill((*iter).adc()); meStripTOBLayer2zp_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer2zp_[4] -> Fill((*iter).adc()); meStripTOBLayer2zp_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer2zp_[5] -> Fill((*iter).adc()); meStripTOBLayer2zp_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 3 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer3zp_[0] -> Fill((*iter).adc()); meStripTOBLayer3zp_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer3zp_[1] -> Fill((*iter).adc()); meStripTOBLayer3zp_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer3zp_[2] -> Fill((*iter).adc()); meStripTOBLayer3zp_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer3zp_[3] -> Fill((*iter).adc()); meStripTOBLayer3zp_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer3zp_[4] -> Fill((*iter).adc()); meStripTOBLayer3zp_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer3zp_[5] -> Fill((*iter).adc()); meStripTOBLayer3zp_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 4 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer4zp_[0] -> Fill((*iter).adc()); meStripTOBLayer4zp_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer4zp_[1] -> Fill((*iter).adc()); meStripTOBLayer4zp_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer4zp_[2] -> Fill((*iter).adc()); meStripTOBLayer4zp_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer4zp_[3] -> Fill((*iter).adc()); meStripTOBLayer4zp_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer4zp_[4] -> Fill((*iter).adc()); meStripTOBLayer4zp_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer4zp_[5] -> Fill((*iter).adc()); meStripTOBLayer4zp_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 5 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer5zp_[0] -> Fill((*iter).adc()); meStripTOBLayer5zp_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer5zp_[1] -> Fill((*iter).adc()); meStripTOBLayer5zp_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer5zp_[2] -> Fill((*iter).adc()); meStripTOBLayer5zp_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer5zp_[3] -> Fill((*iter).adc()); meStripTOBLayer5zp_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer5zp_[4] -> Fill((*iter).adc()); meStripTOBLayer5zp_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer5zp_[5] -> Fill((*iter).adc()); meStripTOBLayer5zp_[5] ->Fill((*iter).strip()); }
                 }
                 if( tTopo->tobLayer(id) == 6 ) {
                     if ( tTopo->tobModule(id) == 1 ) { meAdcTOBLayer6zp_[0] -> Fill((*iter).adc()); meStripTOBLayer6zp_[0] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 2 ) { meAdcTOBLayer6zp_[1] -> Fill((*iter).adc()); meStripTOBLayer6zp_[1] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 3 ) { meAdcTOBLayer6zp_[2] -> Fill((*iter).adc()); meStripTOBLayer6zp_[2] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 4 ) { meAdcTOBLayer6zp_[3] -> Fill((*iter).adc()); meStripTOBLayer6zp_[3] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 5 ) { meAdcTOBLayer6zp_[4] -> Fill((*iter).adc()); meStripTOBLayer6zp_[4] ->Fill((*iter).strip()); }
                     if ( tTopo->tobModule(id) == 6 ) { meAdcTOBLayer6zp_[5] -> Fill((*iter).adc()); meStripTOBLayer6zp_[5] ->Fill((*iter).strip()); }
                 }


               }
             }
        }
   
        if (detId.subdetId()==StripSubdetector::TID) {
              
            for ( iter = begin ; iter != end; iter++ ) {
              if( tTopo->tidSide(id) == 1){
                 ++ndigiwheeltidzm[tTopo->tidWheel(id)-1];
                if( tTopo->tidWheel(id) == 1 ) {
                   if(tTopo->tidRing(id)== 1) { meAdcTIDWheel1zm_[0] -> Fill((*iter).adc()); meStripTIDWheel1zm_[0] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 2) { meAdcTIDWheel1zm_[1] -> Fill((*iter).adc()); meStripTIDWheel1zm_[1] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 3) { meAdcTIDWheel1zm_[2] -> Fill((*iter).adc()); meStripTIDWheel1zm_[2] ->Fill((*iter).strip());}
                }
                if( tTopo->tidWheel(id) == 2 ) {
                   if(tTopo->tidRing(id)== 1) { meAdcTIDWheel2zm_[0] -> Fill((*iter).adc()); meStripTIDWheel2zm_[0] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 2) { meAdcTIDWheel2zm_[1] -> Fill((*iter).adc()); meStripTIDWheel2zm_[1] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 3) { meAdcTIDWheel2zm_[2] -> Fill((*iter).adc()); meStripTIDWheel2zm_[2] ->Fill((*iter).strip());}
                }
                if( tTopo->tidWheel(id) == 3 ) {
                   if(tTopo->tidRing(id)== 1) { meAdcTIDWheel3zm_[0] -> Fill((*iter).adc()); meStripTIDWheel3zm_[0] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 2) { meAdcTIDWheel3zm_[1] -> Fill((*iter).adc()); meStripTIDWheel3zm_[1] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 3) { meAdcTIDWheel3zm_[2] -> Fill((*iter).adc()); meStripTIDWheel3zm_[2] ->Fill((*iter).strip());}
                }

              }else{
                ++ndigiwheeltidzp[tTopo->tidWheel(id)-1];
                if( tTopo->tidWheel(id) == 1 ) { 
                   if(tTopo->tidRing(id)== 1) { meAdcTIDWheel1zp_[0] -> Fill((*iter).adc()); meStripTIDWheel1zp_[0] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 2) { meAdcTIDWheel1zp_[1] -> Fill((*iter).adc()); meStripTIDWheel1zp_[1] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 3) { meAdcTIDWheel1zp_[2] -> Fill((*iter).adc()); meStripTIDWheel1zp_[2] ->Fill((*iter).strip());}
                }  
                if( tTopo->tidWheel(id) == 2 ) {
                   if(tTopo->tidRing(id)== 1) { meAdcTIDWheel2zp_[0] -> Fill((*iter).adc()); meStripTIDWheel2zp_[0] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 2) { meAdcTIDWheel2zp_[1] -> Fill((*iter).adc()); meStripTIDWheel2zp_[1] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 3) { meAdcTIDWheel2zp_[2] -> Fill((*iter).adc()); meStripTIDWheel2zp_[2] ->Fill((*iter).strip());}
                }
                if( tTopo->tidWheel(id) == 3 ) {
                   if(tTopo->tidRing(id)== 1) { meAdcTIDWheel3zp_[0] -> Fill((*iter).adc()); meStripTIDWheel3zp_[0] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 2) { meAdcTIDWheel3zp_[1] -> Fill((*iter).adc()); meStripTIDWheel3zp_[1] ->Fill((*iter).strip());}
                   if(tTopo->tidRing(id)== 3) { meAdcTIDWheel3zp_[2] -> Fill((*iter).adc()); meStripTIDWheel3zp_[2] ->Fill((*iter).strip());}
                }
    
              } 
            }
       }
        if (detId.subdetId()==StripSubdetector::TEC) {
            
            for ( iter = begin ; iter != end; iter++ ) {
              if(tTopo->tecSide(id) == 1) {
                ++ndigiwheelteczm[tTopo->tecWheel(id)-1];
                if( tTopo->tecWheel(id) == 1 ) {
                   if ( tTopo->tecRing(id) == 1 ) { meAdcTECWheel1zm_[0] -> Fill((*iter).adc()); meStripTECWheel1zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel1zm_[1] -> Fill((*iter).adc()); meStripTECWheel1zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel1zm_[2] -> Fill((*iter).adc()); meStripTECWheel1zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel1zm_[3] -> Fill((*iter).adc()); meStripTECWheel1zm_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel1zm_[4] -> Fill((*iter).adc()); meStripTECWheel1zm_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel1zm_[5] -> Fill((*iter).adc()); meStripTECWheel1zm_[5] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel1zm_[6] -> Fill((*iter).adc()); meStripTECWheel1zm_[6] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 2 ) {
                   if ( tTopo->tecRing(id) == 1 ) { meAdcTECWheel2zm_[0] -> Fill((*iter).adc()); meStripTECWheel2zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel2zm_[1] -> Fill((*iter).adc()); meStripTECWheel2zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel2zm_[2] -> Fill((*iter).adc()); meStripTECWheel2zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel2zm_[3] -> Fill((*iter).adc()); meStripTECWheel2zm_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel2zm_[4] -> Fill((*iter).adc()); meStripTECWheel2zm_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel2zm_[5] -> Fill((*iter).adc()); meStripTECWheel2zm_[5] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel2zm_[6] -> Fill((*iter).adc()); meStripTECWheel2zm_[6] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 3 ) {
                   if ( tTopo->tecRing(id) == 1 ) { meAdcTECWheel3zm_[0] -> Fill((*iter).adc()); meStripTECWheel3zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel3zm_[1] -> Fill((*iter).adc()); meStripTECWheel3zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel3zm_[2] -> Fill((*iter).adc()); meStripTECWheel3zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel3zm_[3] -> Fill((*iter).adc()); meStripTECWheel3zm_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel3zm_[4] -> Fill((*iter).adc()); meStripTECWheel3zm_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel3zm_[5] -> Fill((*iter).adc()); meStripTECWheel3zm_[5] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel3zm_[6] -> Fill((*iter).adc()); meStripTECWheel3zm_[6] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 4 ) {
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel4zm_[0] -> Fill((*iter).adc()); meStripTECWheel4zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel4zm_[1] -> Fill((*iter).adc()); meStripTECWheel4zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel4zm_[2] -> Fill((*iter).adc()); meStripTECWheel4zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel4zm_[3] -> Fill((*iter).adc()); meStripTECWheel4zm_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel4zm_[4] -> Fill((*iter).adc()); meStripTECWheel4zm_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel4zm_[5] -> Fill((*iter).adc()); meStripTECWheel4zm_[5] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 5 ) {
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel5zm_[0] -> Fill((*iter).adc()); meStripTECWheel5zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel5zm_[1] -> Fill((*iter).adc()); meStripTECWheel5zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel5zm_[2] -> Fill((*iter).adc()); meStripTECWheel5zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel5zm_[3] -> Fill((*iter).adc()); meStripTECWheel5zm_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel5zm_[4] -> Fill((*iter).adc()); meStripTECWheel5zm_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel5zm_[5] -> Fill((*iter).adc()); meStripTECWheel5zm_[5] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 6 ) {
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel6zm_[0] -> Fill((*iter).adc()); meStripTECWheel6zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel6zm_[1] -> Fill((*iter).adc()); meStripTECWheel6zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel6zm_[2] -> Fill((*iter).adc()); meStripTECWheel6zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel6zm_[3] -> Fill((*iter).adc()); meStripTECWheel6zm_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel6zm_[4] -> Fill((*iter).adc()); meStripTECWheel6zm_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel6zm_[5] -> Fill((*iter).adc()); meStripTECWheel6zm_[5] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 7 ) {
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel7zm_[0] -> Fill((*iter).adc()); meStripTECWheel7zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel7zm_[1] -> Fill((*iter).adc()); meStripTECWheel7zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel7zm_[2] -> Fill((*iter).adc()); meStripTECWheel7zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel7zm_[3] -> Fill((*iter).adc()); meStripTECWheel7zm_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel7zm_[4] -> Fill((*iter).adc()); meStripTECWheel7zm_[4] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 8 ) {
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel8zm_[0] -> Fill((*iter).adc()); meStripTECWheel8zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel8zm_[1] -> Fill((*iter).adc()); meStripTECWheel8zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel8zm_[2] -> Fill((*iter).adc()); meStripTECWheel8zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel8zm_[3] -> Fill((*iter).adc()); meStripTECWheel8zm_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel8zm_[4] -> Fill((*iter).adc()); meStripTECWheel8zm_[4] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 9 ) {
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel9zm_[0] -> Fill((*iter).adc()); meStripTECWheel9zm_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel9zm_[1] -> Fill((*iter).adc()); meStripTECWheel9zm_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel9zm_[2] -> Fill((*iter).adc()); meStripTECWheel9zm_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel9zm_[3] -> Fill((*iter).adc()); meStripTECWheel9zm_[3] ->Fill((*iter).strip()); }
                }
              }else {
                ++ndigiwheelteczp[tTopo->tecWheel(id)-1];
                if( tTopo->tecWheel(id) == 1 ) {
                   if ( tTopo->tecRing(id) == 1 ) { meAdcTECWheel1zp_[0] -> Fill((*iter).adc()); meStripTECWheel1zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel1zp_[1] -> Fill((*iter).adc()); meStripTECWheel1zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel1zp_[2] -> Fill((*iter).adc()); meStripTECWheel1zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel1zp_[3] -> Fill((*iter).adc()); meStripTECWheel1zp_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel1zp_[4] -> Fill((*iter).adc()); meStripTECWheel1zp_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel1zp_[5] -> Fill((*iter).adc()); meStripTECWheel1zp_[5] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel1zp_[6] -> Fill((*iter).adc()); meStripTECWheel1zp_[6] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 2 ) {
                   if ( tTopo->tecRing(id) == 1 ) { meAdcTECWheel2zp_[0] -> Fill((*iter).adc()); meStripTECWheel2zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel2zp_[1] -> Fill((*iter).adc()); meStripTECWheel2zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel2zp_[2] -> Fill((*iter).adc()); meStripTECWheel2zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel2zp_[3] -> Fill((*iter).adc()); meStripTECWheel2zp_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel2zp_[4] -> Fill((*iter).adc()); meStripTECWheel2zp_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel2zp_[5] -> Fill((*iter).adc()); meStripTECWheel2zp_[5] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel2zp_[6] -> Fill((*iter).adc()); meStripTECWheel2zp_[6] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 3 ) {
                   if ( tTopo->tecRing(id) == 1 ) { meAdcTECWheel3zp_[0] -> Fill((*iter).adc()); meStripTECWheel3zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel3zp_[1] -> Fill((*iter).adc()); meStripTECWheel3zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel3zp_[2] -> Fill((*iter).adc()); meStripTECWheel3zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel3zp_[3] -> Fill((*iter).adc()); meStripTECWheel3zp_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel3zp_[4] -> Fill((*iter).adc()); meStripTECWheel3zp_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel3zp_[5] -> Fill((*iter).adc()); meStripTECWheel3zp_[5] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel3zp_[6] -> Fill((*iter).adc()); meStripTECWheel3zp_[6] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 4 ) {
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel4zp_[0] -> Fill((*iter).adc()); meStripTECWheel4zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel4zp_[1] -> Fill((*iter).adc()); meStripTECWheel4zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel4zp_[2] -> Fill((*iter).adc()); meStripTECWheel4zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel4zp_[3] -> Fill((*iter).adc()); meStripTECWheel4zp_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel4zp_[4] -> Fill((*iter).adc()); meStripTECWheel4zp_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel4zp_[5] -> Fill((*iter).adc()); meStripTECWheel4zp_[5] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 5 ) {
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel5zp_[0] -> Fill((*iter).adc()); meStripTECWheel5zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel5zp_[1] -> Fill((*iter).adc()); meStripTECWheel5zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel5zp_[2] -> Fill((*iter).adc()); meStripTECWheel5zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel5zp_[3] -> Fill((*iter).adc()); meStripTECWheel5zp_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel5zp_[4] -> Fill((*iter).adc()); meStripTECWheel5zp_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel5zp_[5] -> Fill((*iter).adc()); meStripTECWheel5zp_[5] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 6 ) {
                   if ( tTopo->tecRing(id) == 2 ) { meAdcTECWheel6zp_[0] -> Fill((*iter).adc()); meStripTECWheel6zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel6zp_[1] -> Fill((*iter).adc()); meStripTECWheel6zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel6zp_[2] -> Fill((*iter).adc()); meStripTECWheel6zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel6zp_[3] -> Fill((*iter).adc()); meStripTECWheel6zp_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel6zp_[4] -> Fill((*iter).adc()); meStripTECWheel6zp_[4] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel6zp_[5] -> Fill((*iter).adc()); meStripTECWheel6zp_[5] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 7 ) {
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel7zp_[0] -> Fill((*iter).adc()); meStripTECWheel7zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel7zp_[1] -> Fill((*iter).adc()); meStripTECWheel7zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel7zp_[2] -> Fill((*iter).adc()); meStripTECWheel7zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel7zp_[3] -> Fill((*iter).adc()); meStripTECWheel7zp_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel7zp_[4] -> Fill((*iter).adc()); meStripTECWheel7zp_[4] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 8 ) {
                   if ( tTopo->tecRing(id) == 3 ) { meAdcTECWheel8zp_[0] -> Fill((*iter).adc()); meStripTECWheel8zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel8zp_[1] -> Fill((*iter).adc()); meStripTECWheel8zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel8zp_[2] -> Fill((*iter).adc()); meStripTECWheel8zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel8zp_[3] -> Fill((*iter).adc()); meStripTECWheel8zp_[3] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel8zp_[4] -> Fill((*iter).adc()); meStripTECWheel8zp_[4] ->Fill((*iter).strip()); }
                }
                if( tTopo->tecWheel(id) == 9 ) {
                   if ( tTopo->tecRing(id) == 4 ) { meAdcTECWheel9zp_[0] -> Fill((*iter).adc()); meStripTECWheel9zp_[0] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 5 ) { meAdcTECWheel9zp_[1] -> Fill((*iter).adc()); meStripTECWheel9zp_[1] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 6 ) { meAdcTECWheel9zp_[2] -> Fill((*iter).adc()); meStripTECWheel9zp_[2] ->Fill((*iter).strip()); }
                   if ( tTopo->tecRing(id) == 7 ) { meAdcTECWheel9zp_[3] -> Fill((*iter).adc()); meStripTECWheel9zp_[3] ->Fill((*iter).strip()); }
                }
             }
           }
       }

 }
  
  for ( int i =0; i< 4; i++ ) {
     meNDigiTIBLayerzm_[i]->Fill(ndigilayertibzm[i]);
     meNDigiTIBLayerzp_[i]->Fill(ndigilayertibzp[i]);
 }
 
 for ( int i =0; i< 6; i++ ) {
     meNDigiTOBLayerzm_[i]->Fill(ndigilayertobzm[i]);
     meNDigiTOBLayerzp_[i]->Fill(ndigilayertobzp[i]);
 }

for ( int i =0; i< 3; i++ ) {
     meNDigiTIDWheelzm_[i]->Fill(ndigiwheeltidzm[i]);
     meNDigiTIDWheelzp_[i]->Fill(ndigiwheeltidzp[i]);
 }

for ( int i =0; i< 9; i++ ) {
     meNDigiTECWheelzm_[i]->Fill(ndigiwheelteczm[i]);
     meNDigiTECWheelzp_[i]->Fill(ndigiwheelteczp[i]);
 }

 

}

