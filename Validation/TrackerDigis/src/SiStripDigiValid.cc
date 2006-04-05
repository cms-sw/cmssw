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
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/LocalPoint.h"


SiStripDigiValid::SiStripDigiValid(const ParameterSet& ps):dbe_(0){

   outputFile_ = ps.getUntrackedParameter<string>("outputFile", "stripdigihisto.root");
   dbe_ = Service<DaqMonitorBEInterface>().operator->();

   meAdcTIBLayer1_ = dbe_->book1D("adc_tib_1","Digis ADC",100,0.,100.);
   meAdcTIBLayer2_ = dbe_->book1D("adc_tib_2","Digis ADC",100,0.,100.);
   meAdcTIBLayer3_ = dbe_->book1D("adc_tib_3","Digis ADC",100,0.,100.);

   meStripTIBLayer1_ = dbe_->book1D("strip_tib_1","Digis Strip Num.",200,0.,800.);
   meStripTIBLayer2_ = dbe_->book1D("strip_tib_2","Digis Strip Num.",200,0.,800.);
   meStripTIBLayer3_ = dbe_->book1D("strip_tib_3","Digis Strip Num.",200,0.,800.);
  
   meAdcTOBLayer1_ = dbe_->book1D("adc_tob_1","Digis ADC",100,0.,100.);
   meAdcTOBLayer2_ = dbe_->book1D("adc_tob_2","Digis ADC",100,0.,100.);
   meAdcTOBLayer3_ = dbe_->book1D("adc_tob_3","Digis ADC",100,0.,100.);
   meAdcTOBLayer4_ = dbe_->book1D("adc_tob_4","Digis ADC",100,0.,100.); 
   meAdcTOBLayer5_ = dbe_->book1D("adc_tob_5","Digis ADC",100,0.,100.);
   meAdcTOBLayer6_ = dbe_->book1D("adc_tob_6","Digis ADC",100,0.,100.);

   meStripTOBLayer1_ = dbe_->book1D("strip_tob_1","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer2_ = dbe_->book1D("strip_tob_2","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer3_ = dbe_->book1D("strip_tob_3","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer4_ = dbe_->book1D("strip_tob_4","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer5_ = dbe_->book1D("strip_tob_5","Digis Strip Num.",200,0.,800.);
   meStripTOBLayer6_ = dbe_->book1D("strip_tob_6","Digis Strip Num.",200,0.,800.);


   meAdcTIDWheel1_ = dbe_->book1D("adc_tid_1","Digis ADC",100,0.,100.);
   meAdcTIDWheel2_ = dbe_->book1D("adc_tid_2","Digis ADC",100,0.,100.);
   meAdcTIDWheel3_ = dbe_->book1D("adc_tid_3","Digis ADC",100,0.,100.);
  
   meStripTIDWheel1_ = dbe_->book1D("strip_tid_1","Digis Strip Num.",200,0.,800.);
   meStripTIDWheel2_ = dbe_->book1D("strip_tid_2","Digis Strip Num.",200,0.,800.);
   meStripTIDWheel3_ = dbe_->book1D("strip_tid_3","Digis Strip Num.",200,0.,800.);
   
   meAdcTECWheel1_ = dbe_->book1D("adc_tec_1","Digis ADC",100,0.,100.);
   meAdcTECWheel2_ = dbe_->book1D("adc_tec_2","Digis ADC",100,0.,100.);
   meAdcTECWheel3_ = dbe_->book1D("adc_tec_3","Digis ADC",100,0.,100.);
   meAdcTECWheel4_ = dbe_->book1D("adc_tec_4","Digis ADC",100,0.,100.);
   meAdcTECWheel5_ = dbe_->book1D("adc_tec_5","Digis ADC",100,0.,100.);
   meAdcTECWheel6_ = dbe_->book1D("adc_tec_6","Digis ADC",100,0.,100.);
   meAdcTECWheel7_ = dbe_->book1D("adc_tec_7","Digis ADC",100,0.,100.);
   meAdcTECWheel8_ = dbe_->book1D("adc_tec_8","Digis ADC",100,0.,100.);
   meAdcTECWheel9_ = dbe_->book1D("adc_tec_9","Digis ADC",100,0.,100.);

   meStripTECWheel1_ = dbe_->book1D("strip_tec_1","Digis Strip Num.",200,0.,800.);
   meStripTECWheel2_ = dbe_->book1D("strip_tec_2","Digis Strip Num.",200,0.,800.);
   meStripTECWheel3_ = dbe_->book1D("strip_tec_3","Digis Strip Num.",200,0.,800.);
   meStripTECWheel4_ = dbe_->book1D("strip_tec_4","Digis Strip Num.",200,0.,800.);
   meStripTECWheel5_ = dbe_->book1D("strip_tec_5","Digis Strip Num.",200,0.,800.);
   meStripTECWheel6_ = dbe_->book1D("strip_tec_6","Digis Strip Num.",200,0.,800.);
   meStripTECWheel7_ = dbe_->book1D("strip_tec_7","Digis Strip Num.",200,0.,800.);
   meStripTECWheel8_ = dbe_->book1D("strip_tec_8","Digis Strip Num.",200,0.,800.);
   meStripTECWheel9_ = dbe_->book1D("strip_tec_9","Digis Strip Num.",200,0.,800.);
  
   meNDigiTIBLayer_ = dbe_->book2D("ndigi_tib_layer","Digi Multiplicity",5,0.,5.0,200,0.,500.);
   meNDigiTOBLayer_ = dbe_->book2D("ndigi_tob_layer","Digi Multiplicity",8,0.,8.0,200,0.,500.);
   meNDigiTIDWheel_ = dbe_->book2D("ndigi_tid_wheel","Digi Multiplicity",5,0.,5.0,200,0.,500.);
   meNDigiTECWheel_ = dbe_->book2D("ndigi_tec_wheel","Digi Multiplicity",11,0.,11.0,200,0.,500.);

}

SiStripDigiValid::~SiStripDigiValid(){

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void SiStripDigiValid::beginJob(const EventSetup& c){

}

void SiStripDigiValid::endJob() {

}


void SiStripDigiValid::analyze(const Event& e, const EventSetup& c){

int ndigitiblayer1=0;
int ndigitiblayer2=0;
int ndigitiblayer3=0;
int ndigitoblayer1=0;
int ndigitoblayer2=0;
int ndigitoblayer3=0;
int ndigitoblayer4=0;
int ndigitoblayer5=0;
int ndigitoblayer6=0;
int ndigitidwheel1=0;
int ndigitidwheel2=0;
int ndigitidwheel3=0;
int ndigitecwheel1=0;
int ndigitecwheel2=0;
int ndigitecwheel3=0;
int ndigitecwheel4=0;
int ndigitecwheel5=0;
int ndigitecwheel6=0;
int ndigitecwheel7=0;
int ndigitecwheel8=0;
int ndigitecwheel9=0;

 LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
 ESHandle<TrackingGeometry> tracker;
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
        const GeomDetUnit * stripdet=tracker->idToDet(detId);
        StripDigiCollection::Range  range = stripDigis->get(id);
        std::vector<StripDigi>::const_iterator begin = range.first;
        std::vector<StripDigi>::const_iterator end = range.second;
        std::vector<StripDigi>::const_iterator iter;

        if(detId.subdetId()==StripSubdetector::TIB){
             TIBDetId tibid(id);
             const RectangularStripTopology& Rtopol=(RectangularStripTopology&)stripdet->topology();
             for ( iter = begin ; iter != end; iter++ ) { // loop digis
                 GlobalPoint gpoint = stripdet->surface().toGlobal(LocalPoint(0,0,0));
                 LogInfo("SiStripDigiValid")<<" Strip="<<(*iter).strip()<<" Adc="<< (*iter).adc()
                                            <<" Gx="<<gpoint.x()<<" Gy="<<gpoint.y()<<" Gz="<<gpoint.z();
                 if( tibid.layer() == 1 ) { meAdcTIBLayer1_ -> Fill((*iter).adc()); meStripTIBLayer1_ ->Fill((*iter).strip()); ++ndigitiblayer1; }
                 if( tibid.layer() == 2 ) { meAdcTIBLayer2_ -> Fill((*iter).adc()); meStripTIBLayer2_ ->Fill((*iter).strip()); ++ndigitiblayer2;}
                 if( tibid.layer() == 3 ) { meAdcTIBLayer3_ -> Fill((*iter).adc()); meStripTIBLayer3_ ->Fill((*iter).strip()); ++ndigitiblayer3;} 
            } 
        } 
        if(detId.subdetId()==StripSubdetector::TOB){
             TOBDetId tobid(id); 
             const RectangularStripTopology& Rtopol=(RectangularStripTopology&)stripdet->topology();
             for ( iter = begin ; iter != end; iter++ ) { // loop digis
                 GlobalPoint gpoint = stripdet->surface().toGlobal(LocalPoint(0,0,0));
                 LogInfo("SiStripDigiValid")<<" Strip="<<(*iter).strip()<<" Adc="<< (*iter).adc()
                                            <<" Gx="<<gpoint.x()<<" Gy="<<gpoint.y()<<" Gz="<<gpoint.z();
                 if( tobid.layer() == 1 ) { meAdcTOBLayer1_ -> Fill((*iter).adc()); meStripTOBLayer1_ ->Fill((*iter).strip()); ++ndigitoblayer1;}
                 if( tobid.layer() == 2 ) { meAdcTOBLayer2_ -> Fill((*iter).adc()); meStripTOBLayer2_ ->Fill((*iter).strip()); ++ndigitoblayer2;}
                 if( tobid.layer() == 3 ) { meAdcTOBLayer3_ -> Fill((*iter).adc()); meStripTOBLayer3_ ->Fill((*iter).strip()); ++ndigitoblayer3;}
                 if( tobid.layer() == 4 ) { meAdcTOBLayer4_ -> Fill((*iter).adc()); meStripTOBLayer4_ ->Fill((*iter).strip()); ++ndigitoblayer4;}
                 if( tobid.layer() == 5 ) { meAdcTOBLayer5_ -> Fill((*iter).adc()); meStripTOBLayer5_ ->Fill((*iter).strip()); ++ndigitoblayer5;}
                 if( tobid.layer() == 6 ) { meAdcTOBLayer6_ -> Fill((*iter).adc()); meStripTOBLayer6_ ->Fill((*iter).strip()); ++ndigitoblayer6;} 
             }
        }
   
        if (detId.subdetId()==StripSubdetector::TID) {
            TIDDetId tidid(id);  
            const TrapezoidalStripTopology& Ttopol=(TrapezoidalStripTopology&)stripdet->topology();
            for ( iter = begin ; iter != end; iter++ ) {
                GlobalPoint gpoint = stripdet->surface().toGlobal(LocalPoint(0,0,0));
                LogInfo("SiStripDigiValid")<<" Strip="<<(*iter).strip()<<" Adc="<< (*iter).adc()
                                           <<" Gx="<<gpoint.x()<<" Gy="<<gpoint.y()<<" Gz="<<gpoint.z();
                if( tidid.wheel() == 1 ) { meAdcTIDWheel1_ -> Fill((*iter).adc()); meStripTIDWheel1_ ->Fill((*iter).strip()); ++ndigitidwheel1;}
                if( tidid.wheel() == 2 ) { meAdcTIDWheel2_ -> Fill((*iter).adc()); meStripTIDWheel2_ ->Fill((*iter).strip()); ++ndigitidwheel2;}
                if( tidid.wheel() == 3 ) { meAdcTIDWheel3_ -> Fill((*iter).adc()); meStripTIDWheel3_ ->Fill((*iter).strip()); ++ndigitidwheel3;}
 
            }
       }
        if (detId.subdetId()==StripSubdetector::TEC) {
            TECDetId tecid(id);
            const TrapezoidalStripTopology& Ttopol=(TrapezoidalStripTopology&)stripdet->topology();
            for ( iter = begin ; iter != end; iter++ ) {
                GlobalPoint gpoint = stripdet->surface().toGlobal(LocalPoint(0,0,0));
                LogInfo("SiStripDigiValid")<<" Strip="<<(*iter).strip()<<" Adc="<< (*iter).adc()
                                           <<" Gx="<<gpoint.x()<<" Gy="<<gpoint.y()<<" Gz="<<gpoint.z();

                if( tecid.wheel() == 1 ) { meAdcTECWheel1_ -> Fill((*iter).adc()); meStripTECWheel1_ ->Fill((*iter).strip()); ++ndigitecwheel1;}
                if( tecid.wheel() == 2 ) { meAdcTECWheel2_ -> Fill((*iter).adc()); meStripTECWheel2_ ->Fill((*iter).strip()); ++ndigitecwheel2;}
                if( tecid.wheel() == 3 ) { meAdcTECWheel3_ -> Fill((*iter).adc()); meStripTECWheel3_ ->Fill((*iter).strip()); ++ndigitecwheel3;}
                if( tecid.wheel() == 4 ) { meAdcTECWheel4_ -> Fill((*iter).adc()); meStripTECWheel4_ ->Fill((*iter).strip()); ++ndigitecwheel4;}
                if( tecid.wheel() == 5 ) { meAdcTECWheel5_ -> Fill((*iter).adc()); meStripTECWheel5_ ->Fill((*iter).strip()); ++ndigitecwheel5;}
                if( tecid.wheel() == 6 ) { meAdcTECWheel6_ -> Fill((*iter).adc()); meStripTECWheel6_ ->Fill((*iter).strip()); ++ndigitecwheel6;}
                if( tecid.wheel() == 7 ) { meAdcTECWheel7_ -> Fill((*iter).adc()); meStripTECWheel7_ ->Fill((*iter).strip()); ++ndigitecwheel7;}
                if( tecid.wheel() == 8 ) { meAdcTECWheel8_ -> Fill((*iter).adc()); meStripTECWheel8_ ->Fill((*iter).strip()); ++ndigitecwheel8;}
                if( tecid.wheel() == 9 ) { meAdcTECWheel9_ -> Fill((*iter).adc()); meStripTECWheel9_ ->Fill((*iter).strip()); ++ndigitecwheel9;}     
            }
       }

   }

 }
            meNDigiTIBLayer_->Fill(1,ndigitiblayer1);
            meNDigiTIBLayer_->Fill(2,ndigitiblayer2);
            meNDigiTIBLayer_->Fill(3,ndigitiblayer3);
            meNDigiTOBLayer_->Fill(1,ndigitoblayer1);
            meNDigiTOBLayer_->Fill(2,ndigitoblayer2);
            meNDigiTOBLayer_->Fill(3,ndigitoblayer3);
            meNDigiTOBLayer_->Fill(4,ndigitoblayer4);
            meNDigiTOBLayer_->Fill(5,ndigitoblayer5);
            meNDigiTOBLayer_->Fill(6,ndigitoblayer6);
            meNDigiTIDWheel_->Fill(1,ndigitidwheel1);
            meNDigiTIDWheel_->Fill(2,ndigitidwheel2);
            meNDigiTIDWheel_->Fill(3,ndigitidwheel3);          
            meNDigiTECWheel_->Fill(1,ndigitecwheel1);
            meNDigiTECWheel_->Fill(2,ndigitecwheel2);
            meNDigiTECWheel_->Fill(3,ndigitecwheel3);
            meNDigiTECWheel_->Fill(4,ndigitecwheel4);
            meNDigiTECWheel_->Fill(5,ndigitecwheel5);
            meNDigiTECWheel_->Fill(6,ndigitecwheel6);
            meNDigiTECWheel_->Fill(7,ndigitecwheel7);
            meNDigiTECWheel_->Fill(8,ndigitecwheel8);
            meNDigiTECWheel_->Fill(9,ndigitecwheel9);
}

