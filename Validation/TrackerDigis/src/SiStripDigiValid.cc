#include "Validation/TrackerDigis/interface/SiStripDigiValid.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/LocalPoint.h"


SiStripDigiValid::SiStripDigiValid(const ParameterSet& ps):dbe_(0){

   outputFile_ = ps.getUntrackedParameter<string>("outputFile", "stripdigihisto.root");
   dbe_ = Service<DaqMonitorBEInterface>().operator->();

   meAdcTIBLayer1zm_ = dbe_->book1D("adc_tib_1_zm","Digis ADC",100,0.,300.);
   meAdcTIBLayer2zm_ = dbe_->book1D("adc_tib_2_zm","Digis ADC",100,0.,300.);
   meAdcTIBLayer3zm_ = dbe_->book1D("adc_tib_3_zm","Digis ADC",100,0.,300.);
   meAdcTIBLayer4zm_ = dbe_->book1D("adc_tib_4_zm","Digis ADC",100,0.,300.);

   meStripTIBLayer1zm_ = dbe_->book1D("strip_tib_1_zm","Digis Strip Num.",200,0.,800.);
   meStripTIBLayer2zm_ = dbe_->book1D("strip_tib_2_zm","Digis Strip Num.",200,0.,800.);
   meStripTIBLayer3zm_ = dbe_->book1D("strip_tib_3_zm","Digis Strip Num.",200,0.,800.);
   meStripTIBLayer4zm_ = dbe_->book1D("strip_tib_4_zm","Digis Strip Num.",200,0.,800.);  

   meAdcTIBLayer1zp_ = dbe_->book1D("adc_tib_1_zp","Digis ADC",100,0.,300.);
   meAdcTIBLayer2zp_ = dbe_->book1D("adc_tib_2_zp","Digis ADC",100,0.,300.);
   meAdcTIBLayer3zp_ = dbe_->book1D("adc_tib_3_zp","Digis ADC",100,0.,300.);
   meAdcTIBLayer4zp_ = dbe_->book1D("adc_tib_4_zp","Digis ADC",100,0.,300.);

   meStripTIBLayer1zp_ = dbe_->book1D("strip_tib_1_zp","Digis Strip Num.",200,0.,800.);
   meStripTIBLayer2zp_ = dbe_->book1D("strip_tib_2_zp","Digis Strip Num.",200,0.,800.);
   meStripTIBLayer3zp_ = dbe_->book1D("strip_tib_3_zp","Digis Strip Num.",200,0.,800.);
   meStripTIBLayer4zp_ = dbe_->book1D("strip_tib_4_zp","Digis Strip Num.",200,0.,800.);

   meAdcTOBLayer1zm_ = dbe_->book1D("adc_tob_1_zm","Digis ADC",100,0.,300.);
   meAdcTOBLayer2zm_ = dbe_->book1D("adc_tob_2_zm","Digis ADC",100,0.,300.);
   meAdcTOBLayer3zm_ = dbe_->book1D("adc_tob_3_zm","Digis ADC",100,0.,300.);
   meAdcTOBLayer4zm_ = dbe_->book1D("adc_tob_4_zm","Digis ADC",100,0.,300.); 
   meAdcTOBLayer5zm_ = dbe_->book1D("adc_tob_5_zm","Digis ADC",100,0.,300.);
   meAdcTOBLayer6zm_ = dbe_->book1D("adc_tob_6_zm","Digis ADC",100,0.,300.);

   meStripTOBLayer1zm_ = dbe_->book1D("strip_tob_1_zm","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer2zm_ = dbe_->book1D("strip_tob_2_zm","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer3zm_ = dbe_->book1D("strip_tob_3_zm","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer4zm_ = dbe_->book1D("strip_tob_4_zm","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer5zm_ = dbe_->book1D("strip_tob_5_zm","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer6zm_ = dbe_->book1D("strip_tob_6_zm","Digis Strip Num.",200,0.,800.);

   meAdcTOBLayer1zp_ = dbe_->book1D("adc_tob_1_zp","Digis ADC",100,0.,300.);
   meAdcTOBLayer2zp_ = dbe_->book1D("adc_tob_2_zp","Digis ADC",100,0.,300.);
   meAdcTOBLayer3zp_ = dbe_->book1D("adc_tob_3_zp","Digis ADC",100,0.,300.);
   meAdcTOBLayer4zp_ = dbe_->book1D("adc_tob_4_zp","Digis ADC",100,0.,300.);
   meAdcTOBLayer5zp_ = dbe_->book1D("adc_tob_5_zp","Digis ADC",100,0.,300.);
   meAdcTOBLayer6zp_ = dbe_->book1D("adc_tob_6_zp","Digis ADC",100,0.,300.);

   meStripTOBLayer1zp_ = dbe_->book1D("strip_tob_1_zp","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer2zp_ = dbe_->book1D("strip_tob_2_zp","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer3zp_ = dbe_->book1D("strip_tob_3_zp","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer4zp_ = dbe_->book1D("strip_tob_4_zp","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer5zp_ = dbe_->book1D("strip_tob_5_zp","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer6zp_ = dbe_->book1D("strip_tob_6_zp","Digis Strip Num.",200,0.,800.);


   meAdcTIDWheel1zp_ = dbe_->book1D("adc_tid_1_zp","Digis ADC",100,0.,300.);
   meAdcTIDWheel2zp_ = dbe_->book1D("adc_tid_2_zp","Digis ADC",100,0.,300.);
   meAdcTIDWheel3zp_ = dbe_->book1D("adc_tid_3_zp","Digis ADC",100,0.,300.);

   meAdcTIDWheel1zm_ = dbe_->book1D("adc_tid_1_zm","Digis ADC",100,0.,300.);
   meAdcTIDWheel2zm_ = dbe_->book1D("adc_tid_2_zm","Digis ADC",100,0.,300.);
   meAdcTIDWheel3zm_ = dbe_->book1D("adc_tid_3_zm","Digis ADC",100,0.,300.);
  
   meStripTIDWheel1zp_ = dbe_->book1D("strip_tid_1_zp","Digis Strip Num.",200,0.,800.);
   meStripTIDWheel2zp_ = dbe_->book1D("strip_tid_2_zp","Digis Strip Num.",200,0.,800.);
   meStripTIDWheel3zp_ = dbe_->book1D("strip_tid_3_zp","Digis Strip Num.",200,0.,800.);

   meStripTIDWheel1zm_ = dbe_->book1D("strip_tid_1_zm","Digis Strip Num.",200,0.,800.);
   meStripTIDWheel2zm_ = dbe_->book1D("strip_tid_2_zm","Digis Strip Num.",200,0.,800.);
   meStripTIDWheel3zm_ = dbe_->book1D("strip_tid_3_zm","Digis Strip Num.",200,0.,800.);

  
   meAdcTECWheel1zp_ = dbe_->book1D("adc_tec_1_zp","Digis ADC",100,0.,300.);
   meAdcTECWheel2zp_ = dbe_->book1D("adc_tec_2_zp","Digis ADC",100,0.,300.);
   meAdcTECWheel3zp_ = dbe_->book1D("adc_tec_3_zp","Digis ADC",100,0.,300.);
   meAdcTECWheel4zp_ = dbe_->book1D("adc_tec_4_zp","Digis ADC",100,0.,300.);
   meAdcTECWheel5zp_ = dbe_->book1D("adc_tec_5_zp","Digis ADC",100,0.,300.);
   meAdcTECWheel6zp_ = dbe_->book1D("adc_tec_6_zp","Digis ADC",100,0.,300.);
   meAdcTECWheel7zp_ = dbe_->book1D("adc_tec_7_zp","Digis ADC",100,0.,300.);
   meAdcTECWheel8zp_ = dbe_->book1D("adc_tec_8_zp","Digis ADC",100,0.,300.);
   meAdcTECWheel9zp_ = dbe_->book1D("adc_tec_9_zp","Digis ADC",100,0.,300.);

   meAdcTECWheel1zm_ = dbe_->book1D("adc_tec_1_zm","Digis ADC",100,0.,300.);
   meAdcTECWheel2zm_ = dbe_->book1D("adc_tec_2_zm","Digis ADC",100,0.,300.);
   meAdcTECWheel3zm_ = dbe_->book1D("adc_tec_3_zm","Digis ADC",100,0.,300.);
   meAdcTECWheel4zm_ = dbe_->book1D("adc_tec_4_zm","Digis ADC",100,0.,300.);
   meAdcTECWheel5zm_ = dbe_->book1D("adc_tec_5_zm","Digis ADC",100,0.,300.);
   meAdcTECWheel6zm_ = dbe_->book1D("adc_tec_6_zm","Digis ADC",100,0.,300.);
   meAdcTECWheel7zm_ = dbe_->book1D("adc_tec_7_zm","Digis ADC",100,0.,300.);
   meAdcTECWheel8zm_ = dbe_->book1D("adc_tec_8_zm","Digis ADC",100,0.,300.);
   meAdcTECWheel9zm_ = dbe_->book1D("adc_tec_9_zm","Digis ADC",100,0.,300.);

   meStripTECWheel1zp_ = dbe_->book1D("strip_tec_1_zp","Digis Strip Num.",200,0.,800.);
   meStripTECWheel2zp_ = dbe_->book1D("strip_tec_2_zp","Digis Strip Num.",200,0.,800.);
   meStripTECWheel3zp_ = dbe_->book1D("strip_tec_3_zp","Digis Strip Num.",200,0.,800.);
   meStripTECWheel4zp_ = dbe_->book1D("strip_tec_4_zp","Digis Strip Num.",200,0.,800.);
   meStripTECWheel5zp_ = dbe_->book1D("strip_tec_5_zp","Digis Strip Num.",200,0.,800.);
   meStripTECWheel6zp_ = dbe_->book1D("strip_tec_6_zp","Digis Strip Num.",200,0.,800.);
   meStripTECWheel7zp_ = dbe_->book1D("strip_tec_7_zp","Digis Strip Num.",200,0.,800.);
   meStripTECWheel8zp_ = dbe_->book1D("strip_tec_8_zp","Digis Strip Num.",200,0.,800.);
   meStripTECWheel9zp_ = dbe_->book1D("strip_tec_9_zp","Digis Strip Num.",200,0.,800.);

   meStripTECWheel1zm_ = dbe_->book1D("strip_tec_1_zm","Digis Strip Num.",200,0.,800.);
   meStripTECWheel2zm_ = dbe_->book1D("strip_tec_2_zm","Digis Strip Num.",200,0.,800.);
   meStripTECWheel3zm_ = dbe_->book1D("strip_tec_3_zm","Digis Strip Num.",200,0.,800.);
   meStripTECWheel4zm_ = dbe_->book1D("strip_tec_4_zm","Digis Strip Num.",200,0.,800.);
   meStripTECWheel5zm_ = dbe_->book1D("strip_tec_5_zm","Digis Strip Num.",200,0.,800.);
   meStripTECWheel6zm_ = dbe_->book1D("strip_tec_6_zm","Digis Strip Num.",200,0.,800.);
   meStripTECWheel7zm_ = dbe_->book1D("strip_tec_7_zm","Digis Strip Num.",200,0.,800.);
   meStripTECWheel8zm_ = dbe_->book1D("strip_tec_8_zm","Digis Strip Num.",200,0.,800.);
   meStripTECWheel9zm_ = dbe_->book1D("strip_tec_9_zm","Digis Strip Num.",200,0.,800.);
   
   for(int i = 0 ;i<4 ; i++) {
      Char_t histo[200];
      sprintf(histo,"ndigi_tib_layer_%d_zm",i+1);
      meNDigiTIBLayerzm_[i] = dbe_->book1D(histo, "Digi Multiplicity",200,0.,500.);
      sprintf(histo,"ndigi_tib_layer_%d_zp",i+1); 
      meNDigiTIBLayerzp_[i] = dbe_->book1D(histo, "Digi Multiplicity",200,0.,500.);
   }

   for(int i = 0 ;i<6 ; i++) {
      Char_t histo[200];
      sprintf(histo,"ndigi_tob_layer_%d_zm",i+1);
      meNDigiTOBLayerzm_[i] = dbe_->book1D(histo, "Digi Multiplicity",200,0.,500.);
      sprintf(histo,"ndigi_tob_layer_%d_zp",i+1);
      meNDigiTOBLayerzp_[i] = dbe_->book1D(histo, "Digi Multiplicity",200,0.,500.);

   }

   for(int i = 0 ;i<3 ; i++) {
      Char_t histo[200];
      sprintf(histo,"ndigi_tid_wheel_%d_zm",i+1);
      meNDigiTIDWheelzm_[i] = dbe_->book1D(histo, "Digi Multiplicity",200,0.,500.);
      sprintf(histo,"ndigi_tid_wheel_%d_zp",i+1);
      meNDigiTIDWheelzp_[i] = dbe_->book1D(histo, "Digi Multiplicity",200,0.,500.);
   }

  for(int i = 0 ;i<9 ; i++) {
      Char_t histo[200];
      sprintf(histo,"ndigi_tec_wheel_%d_zm",i+1);
      meNDigiTECWheelzm_[i] = dbe_->book1D(histo, "Digi Multiplicity",200,0.,500.);
      sprintf(histo,"ndigi_tec_wheel_%d_zp",i+1);
      meNDigiTECWheelzp_[i] = dbe_->book1D(histo, "Digi Multiplicity",200,0.,500.);

   }


}

SiStripDigiValid::~SiStripDigiValid(){

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void SiStripDigiValid::beginJob(const EventSetup& c){

}

void SiStripDigiValid::endJob() {

}


void SiStripDigiValid::analyze(const Event& e, const EventSetup& c){

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



 LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
 ESHandle<TrackerGeometry> tracker;
 c.get<TrackerDigiGeometryRecord>().get( tracker );

 std::string digiProducer = "stripdigi";
 Handle<StripDigiCollection> stripDigis;
 e.getByLabel(digiProducer, stripDigis);
 std::vector<unsigned int>  vec = stripDigis->detIDs();

 if ( vec.size() > 0 )
    LogInfo("SiStripDigiValid")<<" DetId Size = "<< vec.size();


   
 for (unsigned int i=0; i< vec.size(); i++) {
    unsigned int id = vec[i];
    if( id != 999999999){ //if is valid detector
        DetId  detId(id);
        StripDigiCollection::Range  range = stripDigis->get(id);
        std::vector<StripDigi>::const_iterator begin = range.first;
        std::vector<StripDigi>::const_iterator end = range.second;
        std::vector<StripDigi>::const_iterator iter;

        if(detId.subdetId()==StripSubdetector::TIB){
             TIBDetId tibid(id);
             for ( iter = begin ; iter != end; iter++ ) { // loop digis
               if( tibid.string()[0] == 0) {
                 ++ndigilayertibzm[tibid.layer()-1];
                 if( tibid.layer() == 1 ) { meAdcTIBLayer1zm_ -> Fill((*iter).adc()); meStripTIBLayer1zm_ ->Fill((*iter).strip()); }
                 if( tibid.layer() == 2 ) { meAdcTIBLayer2zm_ -> Fill((*iter).adc()); meStripTIBLayer2zm_ ->Fill((*iter).strip()); }
                 if( tibid.layer() == 3 ) { meAdcTIBLayer3zm_ -> Fill((*iter).adc()); meStripTIBLayer3zm_ ->Fill((*iter).strip()); } 
                 if( tibid.layer() == 4 ) { meAdcTIBLayer4zm_ -> Fill((*iter).adc()); meStripTIBLayer4zm_ ->Fill((*iter).strip()); }
               }else {
                 ++ndigilayertibzp[tibid.layer()-1];
                 if( tibid.layer() == 1 ) { meAdcTIBLayer1zp_ -> Fill((*iter).adc()); meStripTIBLayer1zp_ ->Fill((*iter).strip()); }
                 if( tibid.layer() == 2 ) { meAdcTIBLayer2zp_ -> Fill((*iter).adc()); meStripTIBLayer2zp_ ->Fill((*iter).strip()); }
                 if( tibid.layer() == 3 ) { meAdcTIBLayer3zp_ -> Fill((*iter).adc()); meStripTIBLayer3zp_ ->Fill((*iter).strip()); }
                 if( tibid.layer() == 4 ) { meAdcTIBLayer4zp_ -> Fill((*iter).adc()); meStripTIBLayer4zp_ ->Fill((*iter).strip()); }
              }
            } 
        } 
        if(detId.subdetId()==StripSubdetector::TOB){
             TOBDetId tobid(id); 
             for ( iter = begin ; iter != end; iter++ ) { // loop digis
               if( tobid.rod()[0] == 0 ) {
                 ++ndigilayertobzm[tobid.layer()-1];  
                 if( tobid.layer() == 1 ) { meAdcTOBLayer1zm_ -> Fill((*iter).adc()); meStripTOBLayer1zm_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 2 ) { meAdcTOBLayer2zm_ -> Fill((*iter).adc()); meStripTOBLayer2zm_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 3 ) { meAdcTOBLayer3zm_ -> Fill((*iter).adc()); meStripTOBLayer3zm_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 4 ) { meAdcTOBLayer4zm_ -> Fill((*iter).adc()); meStripTOBLayer4zm_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 5 ) { meAdcTOBLayer5zm_ -> Fill((*iter).adc()); meStripTOBLayer5zm_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 6 ) { meAdcTOBLayer6zm_ -> Fill((*iter).adc()); meStripTOBLayer6zm_ ->Fill((*iter).strip()); } 
               }else {
                 ++ndigilayertobzp[tobid.layer()-1];
                 if( tobid.layer() == 1 ) { meAdcTOBLayer1zp_ -> Fill((*iter).adc()); meStripTOBLayer1zp_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 2 ) { meAdcTOBLayer2zp_ -> Fill((*iter).adc()); meStripTOBLayer2zp_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 3 ) { meAdcTOBLayer3zp_ -> Fill((*iter).adc()); meStripTOBLayer3zp_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 4 ) { meAdcTOBLayer4zp_ -> Fill((*iter).adc()); meStripTOBLayer4zp_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 5 ) { meAdcTOBLayer5zp_ -> Fill((*iter).adc()); meStripTOBLayer5zp_ ->Fill((*iter).strip()); }
                 if( tobid.layer() == 6 ) { meAdcTOBLayer6zp_ -> Fill((*iter).adc()); meStripTOBLayer6zp_ ->Fill((*iter).strip()); }
               }
             }
        }
   
        if (detId.subdetId()==StripSubdetector::TID) {
            TIDDetId tidid(id);  
            for ( iter = begin ; iter != end; iter++ ) {
              if( tidid.side() == 1){
                 ++ndigiwheeltidzm[tidid.wheel()-1];
                if( tidid.wheel() == 1 ) { meAdcTIDWheel1zm_ -> Fill((*iter).adc()); meStripTIDWheel1zm_ ->Fill((*iter).strip());}
                if( tidid.wheel() == 2 ) { meAdcTIDWheel2zm_ -> Fill((*iter).adc()); meStripTIDWheel2zm_ ->Fill((*iter).strip());}
                if( tidid.wheel() == 3 ) { meAdcTIDWheel3zm_ -> Fill((*iter).adc()); meStripTIDWheel3zm_ ->Fill((*iter).strip());}
              }else{
                ++ndigiwheeltidzp[tidid.wheel()-1];
                if( tidid.wheel() == 1 ) { meAdcTIDWheel1zp_ -> Fill((*iter).adc()); meStripTIDWheel1zp_ ->Fill((*iter).strip()); }
                if( tidid.wheel() == 2 ) { meAdcTIDWheel2zp_ -> Fill((*iter).adc()); meStripTIDWheel2zp_ ->Fill((*iter).strip()); }
                if( tidid.wheel() == 3 ) { meAdcTIDWheel3zp_ -> Fill((*iter).adc()); meStripTIDWheel3zp_ ->Fill((*iter).strip()); }
              } 
            }
       }
        if (detId.subdetId()==StripSubdetector::TEC) {
            TECDetId tecid(id);
            for ( iter = begin ; iter != end; iter++ ) {
              if(tecid.side() == 1) {
                ++ndigiwheelteczm[tecid.wheel()-1];
                if( tecid.wheel() == 1 ) { meAdcTECWheel1zm_ -> Fill((*iter).adc()); meStripTECWheel1zm_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 2 ) { meAdcTECWheel2zm_ -> Fill((*iter).adc()); meStripTECWheel2zm_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 3 ) { meAdcTECWheel3zm_ -> Fill((*iter).adc()); meStripTECWheel3zm_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 4 ) { meAdcTECWheel4zm_ -> Fill((*iter).adc()); meStripTECWheel4zm_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 5 ) { meAdcTECWheel5zm_ -> Fill((*iter).adc()); meStripTECWheel5zm_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 6 ) { meAdcTECWheel6zm_ -> Fill((*iter).adc()); meStripTECWheel6zm_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 7 ) { meAdcTECWheel7zm_ -> Fill((*iter).adc()); meStripTECWheel7zm_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 8 ) { meAdcTECWheel8zm_ -> Fill((*iter).adc()); meStripTECWheel8zm_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 9 ) { meAdcTECWheel9zm_ -> Fill((*iter).adc()); meStripTECWheel9zm_ ->Fill((*iter).strip()); }     
              }else {
                ++ndigiwheelteczp[tecid.wheel()-1];
                if( tecid.wheel() == 1 ) { meAdcTECWheel1zp_ -> Fill((*iter).adc()); meStripTECWheel1zp_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 2 ) { meAdcTECWheel2zp_ -> Fill((*iter).adc()); meStripTECWheel2zp_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 3 ) { meAdcTECWheel3zp_ -> Fill((*iter).adc()); meStripTECWheel3zp_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 4 ) { meAdcTECWheel4zp_ -> Fill((*iter).adc()); meStripTECWheel4zp_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 5 ) { meAdcTECWheel5zp_ -> Fill((*iter).adc()); meStripTECWheel5zp_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 6 ) { meAdcTECWheel6zp_ -> Fill((*iter).adc()); meStripTECWheel6zp_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 7 ) { meAdcTECWheel7zp_ -> Fill((*iter).adc()); meStripTECWheel7zp_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 8 ) { meAdcTECWheel8zp_ -> Fill((*iter).adc()); meStripTECWheel8zp_ ->Fill((*iter).strip()); }
                if( tecid.wheel() == 9 ) { meAdcTECWheel9zp_ -> Fill((*iter).adc()); meStripTECWheel9zp_ ->Fill((*iter).strip()); }
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

