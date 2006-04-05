#include "Validation/TrackerDigis/interface/SiPixelDigiValid.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"




SiPixelDigiValid::SiPixelDigiValid(const ParameterSet& ps):dbe_(0){
  
   outputFile_ = ps.getUntrackedParameter<string>("outputFile", "pixeldigihisto.root");
   dbe_ = Service<DaqMonitorBEInterface>().operator->();
  //Blade Number
   meNDigiBlade1Zp_ = dbe_->book1D("nblade_1zp","Number of Digis",25,0.,25.);
   meNDigiBlade2Zp_ = dbe_->book1D("nblade_2zp","Number of Digis",25,0.,25.);
   meNDigiBlade1Zm_ = dbe_->book1D("nblade_1zm","Number of Digis",25,0.,25.);
   meNDigiBlade2Zm_ = dbe_->book1D("nblade_2zm","Number of Digis",25,0.,25.);
  //ADC Count
   meAdcDisk1Panel1Zp_ = dbe_->book1D("adc_disk1_panel1_zp","Digi charge",500,0.,500.);
   meAdcDisk1Panel2Zp_ = dbe_->book1D("adc_disk1_panel2_zp","Digi charge",500,0.,500.);
   meAdcDisk2Panel1Zp_ = dbe_->book1D("adc_disk2_panel1_zp","Digi charge",500,0.,500.);
   meAdcDisk2Panel2Zp_ = dbe_->book1D("adc_disk2_panel2_zp","Digi charge",500,0.,500.);

   meAdcDisk1Panel1Zm_ = dbe_->book1D("adc_disk1_panel1_zm","Digi charge",500,0.,500.);
   meAdcDisk1Panel2Zm_ = dbe_->book1D("adc_disk1_panel2_zm","Digi charge",500,0.,500.);
   meAdcDisk2Panel1Zm_ = dbe_->book1D("adc_disk2_panel1_zm","Digi charge",500,0.,500.);
   meAdcDisk2Panel2Zm_ = dbe_->book1D("adc_disk2_panel2_zm","Digi charge",500,0.,500.);
  //Col Number
   meColDisk1Panel1Zp_ = dbe_->book1D("col_disk1_panel1_zp","Digi column",500,0.,500.);
   meColDisk1Panel2Zp_ = dbe_->book1D("col_disk1_panel2_zp","Digi column",500,0.,500.);
   meColDisk2Panel1Zp_ = dbe_->book1D("col_disk2_panel1_zp","Digi column",500,0.,500.);
   meColDisk2Panel2Zp_ = dbe_->book1D("col_disk2_panel2_zp","Digi column",500,0.,500.);

   meColDisk1Panel1Zm_ = dbe_->book1D("col_disk1_panel1_zm","Digi column",500,0.,500.);
   meColDisk1Panel2Zm_ = dbe_->book1D("col_disk1_panel2_zm","Digi column",500,0.,500.);
   meColDisk2Panel1Zm_ = dbe_->book1D("col_disk2_panel1_zm","Digi column",500,0.,500.);
   meColDisk2Panel2Zm_ = dbe_->book1D("col_disk2_panel2_zm","Digi column",500,0.,500.);

  // ROW Number
   meRowDisk1Panel1Zp_ = dbe_->book1D("row_disk1_panel1_zp","Digi row",200,0.,200.);
   meRowDisk1Panel2Zp_ = dbe_->book1D("row_disk1_panel2_zp","Digi row",200,0.,200.);
   meRowDisk2Panel1Zp_ = dbe_->book1D("row_disk2_panel1_zp","Digi row",200,0.,200.);
   meRowDisk2Panel2Zp_ = dbe_->book1D("row_disk2_panel2_zp","Digi row",200,0.,200.);

   meRowDisk1Panel1Zm_ = dbe_->book1D("row_disk1_panel1_zm","Digi row",200,0.,200.);
   meRowDisk1Panel2Zm_ = dbe_->book1D("row_disk1_panel2_zm","Digi row",200,0.,200.);
   meRowDisk2Panel1Zm_ = dbe_->book1D("row_disk2_panel1_zm","Digi row",200,0.,200.);
   meRowDisk2Panel2Zm_ = dbe_->book1D("row_disk2_panel2_zm","Digi row",200,0.,200.);
  /////Barrel 
   meAdcLayer1Ladder1_ = dbe_->book1D("adc_layer1ladder1","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder2_ = dbe_->book1D("adc_layer1ladder2","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder3_ = dbe_->book1D("adc_layer1ladder3","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder4_ = dbe_->book1D("adc_layer1ladder4","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder5_ = dbe_->book1D("adc_layer1ladder5","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder6_ = dbe_->book1D("adc_layer1ladder6","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder7_ = dbe_->book1D("adc_layer1ladder7","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder8_ = dbe_->book1D("adc_layer1ladder8","Digi charge",300, 0., 300.);
   
   meAdcLayer2Ladder1_ = dbe_->book1D("adc_layer2ladder1","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder2_ = dbe_->book1D("adc_layer2ladder2","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder3_ = dbe_->book1D("adc_layer2ladder3","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder4_ = dbe_->book1D("adc_layer2ladder4","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder5_ = dbe_->book1D("adc_layer2ladder5","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder6_ = dbe_->book1D("adc_layer2ladder6","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder7_ = dbe_->book1D("adc_layer2ladder7","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder8_ = dbe_->book1D("adc_layer2ladder8","Digi charge",300, 0., 300.);

   meAdcLayer3Ladder1_ = dbe_->book1D("adc_layer3ladder1","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder2_ = dbe_->book1D("adc_layer3ladder2","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder3_ = dbe_->book1D("adc_layer3ladder3","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder4_ = dbe_->book1D("adc_layer3ladder4","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder5_ = dbe_->book1D("adc_layer3ladder5","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder6_ = dbe_->book1D("adc_layer3ladder6","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder7_ = dbe_->book1D("adc_layer3ladder7","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder8_ = dbe_->book1D("adc_layer3ladder8","Digi charge",300, 0., 300.);


}

SiPixelDigiValid::~SiPixelDigiValid(){
 
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void SiPixelDigiValid::beginJob(const EventSetup& c){

}

void SiPixelDigiValid::endJob() {

}


void SiPixelDigiValid::analyze(const Event& e, const EventSetup& c){

 LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

 //edm::ESHandle<TrackingGeometry> tracker;
 edm::ESHandle<TrackerGeometry> tracker;
 c.get<TrackerDigiGeometryRecord>().get( tracker );     

 string digiProducer = "pixdigi";
 Handle<PixelDigiCollection> pixelDigis;
 e.getByLabel(digiProducer, pixelDigis);
 vector<unsigned int>  vec = pixelDigis->detIDs();


 if ( vec.size() > 0 ) 
 LogInfo("SiPixelDigiValid") <<"DetId Size = " <<vec.size();

 for (unsigned int i=0; i< vec.size(); i++) {
       unsigned int id = vec[i];
       if( id != 999999999){ //if is valid detector
          DetId  detId(id);
          const GeomDetUnit * pixeldet=tracker->idToDet(detId);
          PixelDigiCollection::Range  range = pixelDigis->get(id);
          std::vector<PixelDigi>::const_iterator begin = range.first;
          std::vector<PixelDigi>::const_iterator end = range.second;
          std::vector<PixelDigi>::const_iterator iter;
          
          if(detId.subdetId()==PixelSubdetector::PixelBarrel ) {
             PXBDetId  bdetid(id);
             unsigned int layer  = bdetid.layer();   // Layer:1,2,3.
             unsigned int ladder = bdetid.ladder();  // Ladeer: 1-20, 32, 44. 
             unsigned int zindex = bdetid.module();  // Z-index: 1-8.
             LogInfo("SiPixelDigiValid")<<"Barrel:: Layer="<<layer<<" Ladder="<<ladder<<" zindex="<<zindex;
             for ( iter = begin ; iter != end; iter++ ) {
                if( layer == 1 ) {
                     if (zindex == 1)  meAdcLayer1Ladder1_->Fill((*iter).adc());
                     if (zindex == 2)  meAdcLayer1Ladder2_->Fill((*iter).adc());
                     if (zindex == 3)  meAdcLayer1Ladder3_->Fill((*iter).adc());
                     if (zindex == 4)  meAdcLayer1Ladder4_->Fill((*iter).adc());
                     if (zindex == 5)  meAdcLayer1Ladder5_->Fill((*iter).adc());
                     if (zindex == 6)  meAdcLayer1Ladder6_->Fill((*iter).adc());
                     if (zindex == 7)  meAdcLayer1Ladder7_->Fill((*iter).adc());
                     if (zindex == 8)  meAdcLayer1Ladder8_->Fill((*iter).adc());
                } 
                if( layer == 2 ) {
                     if (zindex == 1)  meAdcLayer2Ladder1_->Fill((*iter).adc());
                     if (zindex == 2)  meAdcLayer2Ladder2_->Fill((*iter).adc());
                     if (zindex == 3)  meAdcLayer2Ladder3_->Fill((*iter).adc());
                     if (zindex == 4)  meAdcLayer2Ladder4_->Fill((*iter).adc());
                     if (zindex == 5)  meAdcLayer2Ladder5_->Fill((*iter).adc());
                     if (zindex == 6)  meAdcLayer2Ladder6_->Fill((*iter).adc());
                     if (zindex == 7)  meAdcLayer2Ladder7_->Fill((*iter).adc());
                     if (zindex == 8)  meAdcLayer2Ladder8_->Fill((*iter).adc());
                }
                if( layer == 3 ) {
                     if (zindex == 1)  meAdcLayer3Ladder1_->Fill((*iter).adc());
                     if (zindex == 2)  meAdcLayer3Ladder2_->Fill((*iter).adc());
                     if (zindex == 3)  meAdcLayer3Ladder3_->Fill((*iter).adc());
                     if (zindex == 4)  meAdcLayer3Ladder4_->Fill((*iter).adc());
                     if (zindex == 5)  meAdcLayer3Ladder5_->Fill((*iter).adc());
                     if (zindex == 6)  meAdcLayer3Ladder6_->Fill((*iter).adc());
                     if (zindex == 7)  meAdcLayer3Ladder7_->Fill((*iter).adc());
                     if (zindex == 8)  meAdcLayer3Ladder8_->Fill((*iter).adc());
                }
 
             }   
           
          }
 
          if(detId.subdetId()==PixelSubdetector::PixelEndcap ){ //Endcap
             //const RectangularPixelTopology& Rtopol=(RectangularPixelTopology&)pixeldet->topology();
             PXFDetId  fdetid(id);
             unsigned int side  = fdetid.side();
             unsigned int disk  = fdetid.disk();
             unsigned int blade = fdetid.blade();
             unsigned int panel = fdetid.panel();
             unsigned int mod   = fdetid.module();
             LogInfo("SiPixelDigiValid")<<"EndcaP="<<side<<" Disk="<<disk<<" Blade="<<blade<<" Panel="<<panel<<" Module="<<mod;
             if(side == 1 &&  disk == 1 && panel == 1){
               meNDigiBlade1Zp_->Fill(blade);
               for ( iter = begin ; iter != end; iter++ ) {
                 LogInfo("SiPixelDigiValid") <<"Channel="<<(*iter).channel()<<" Adc="<<(*iter).adc()<<" Row="<<(*iter).row()<<" Col="<<(*iter).column();
                 meAdcDisk1Panel1Zp_->Fill( (*iter).adc()    );
                 meColDisk1Panel1Zp_->Fill( (*iter).column() );
                 meRowDisk1Panel1Zp_->Fill( (*iter).row()    );
               }
             }

             if(side == 1 &&  disk == 1 && panel == 2){
               meNDigiBlade1Zp_->Fill(blade);
               for ( iter = begin ; iter != end; iter++ ) {
                 LogInfo("SiPixelDigiValid") <<"Channel="<<(*iter).channel()<<" Adc="<<(*iter).adc()<<" Row="<<(*iter).row()<<" Col="<<(*iter).column();
                 meAdcDisk1Panel2Zp_->Fill( (*iter).adc()    );
                 meColDisk1Panel2Zp_->Fill( (*iter).column() );
                 meRowDisk1Panel2Zp_->Fill( (*iter).row()    );
               }
             }

             if(side == 1 &&  disk == 2 && panel == 1){
               meNDigiBlade2Zp_->Fill(blade);
               for ( iter = begin ; iter != end; iter++ ) {
                 LogInfo("SiPixelDigiValid") <<"Channel="<<(*iter).channel()<<" Adc="<<(*iter).adc()<<" Row="<<(*iter).row()<<" Col="<<(*iter).column();
                 meAdcDisk2Panel1Zp_->Fill( (*iter).adc()    );
                 meColDisk2Panel1Zp_->Fill( (*iter).column() );
                 meRowDisk2Panel1Zp_->Fill( (*iter).row()    );
               }
             }
            if(side == 1 &&  disk == 2 && panel == 2){
               meNDigiBlade2Zp_->Fill(blade);
               for ( iter = begin ; iter != end; iter++ ) {
                 LogInfo("SiPixelDigiValid") <<"Channel="<<(*iter).channel()<<" Adc="<<(*iter).adc()<<" Row="<<(*iter).row()<<" Col="<<(*iter).column();
                 meAdcDisk2Panel2Zp_->Fill( (*iter).adc()    );
                 meColDisk2Panel2Zp_->Fill( (*iter).column() );
                 meRowDisk2Panel2Zp_->Fill( (*iter).row()    );
               }
             }

             if(side == 2 &&  disk == 1 && panel == 1 ){
               meNDigiBlade1Zm_->Fill(blade);
               for ( iter = begin ; iter != end; iter++ ) {
                 LogInfo("SiPixelDigiValid") <<"Channel="<<(*iter).channel()<<" Adc="<<(*iter).adc()<<" Row="<<(*iter).row()<<" Col="<<(*iter).column();
                 meAdcDisk1Panel1Zm_->Fill( (*iter).adc()    );
                 meColDisk1Panel1Zm_->Fill( (*iter).column() );
                 meRowDisk1Panel1Zm_->Fill( (*iter).row()    );
               }
             }

            if(side == 2 &&  disk == 1 && panel == 2 ){
               meNDigiBlade1Zm_->Fill(blade);
               for ( iter = begin ; iter != end; iter++ ) {
                 LogInfo("SiPixelDigiValid") <<"Channel="<<(*iter).channel()<<" Adc="<<(*iter).adc()<<" Row="<<(*iter).row()<<" Col="<<(*iter).column();
                 meAdcDisk1Panel2Zm_->Fill( (*iter).adc()    );
                 meColDisk1Panel2Zm_->Fill( (*iter).column() );
                 meRowDisk1Panel2Zm_->Fill( (*iter).row()    );
               }

             }
             if(side == 2 &&  disk == 1 && panel == 1){
               meNDigiBlade2Zm_->Fill(blade);
               for ( iter = begin ; iter != end; iter++ ) {
                 LogInfo("SiPixelDigiValid") <<"Channel="<<(*iter).channel()<<" Adc="<<(*iter).adc()<<" Row="<<(*iter).row()<<" Col="<<(*iter).column();
                 meAdcDisk2Panel1Zm_->Fill( (*iter).adc()    );
                 meColDisk2Panel1Zm_->Fill( (*iter).column() );
                 meRowDisk2Panel1Zm_->Fill( (*iter).row()    );
               }
             }
             if(side == 2 &&  disk == 1 && panel == 2){
               meNDigiBlade2Zm_->Fill(blade);
               for ( iter = begin ; iter != end; iter++ ) {
                 LogInfo("SiPixelDigiValid") <<"Channel="<<(*iter).channel()<<" Adc="<<(*iter).adc()<<" Row="<<(*iter).row()<<" Col="<<(*iter).column();
                 meAdcDisk2Panel2Zm_->Fill( (*iter).adc()    );
                 meColDisk2Panel2Zm_->Fill( (*iter).column() );
                 meRowDisk2Panel2Zm_->Fill( (*iter).row()    );
               }
             }
           

          }//Endcap

       }//end if id.
    }
}

//define this as a plug-in
//DEFINE_FWK_MODULE(SiPixelDigiValid)
