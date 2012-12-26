#include "Validation/TrackerDigis/interface/SiPixelDigiValid.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
using namespace edm;

SiPixelDigiValid::SiPixelDigiValid(const ParameterSet& ps):dbe_(0){
  
   
 outputFile_ = ps.getUntrackedParameter<string>("outputFile", "pixeldigihisto.root");
 src_ =  ps.getParameter<edm::InputTag>( "src" );


   dbe_ = Service<DQMStore>().operator->();
  
 if ( dbe_ ) {
    dbe_->setCurrentFolder("TrackerDigisV/TrackerDigis/Pixel");

   meDigiMultiLayer1Ring1_ =  dbe_->book1D("digimulti_layer1ring1","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer1Ring2_ =  dbe_->book1D("digimulti_layer1ring2","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer1Ring3_ =  dbe_->book1D("digimulti_layer1ring3","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer1Ring4_ =  dbe_->book1D("digimulti_layer1ring4","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer1Ring5_ =  dbe_->book1D("digimulti_layer1ring5","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer1Ring6_ =  dbe_->book1D("digimulti_layer1ring6","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer1Ring7_ =  dbe_->book1D("digimulti_layer1ring7","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer1Ring8_ =  dbe_->book1D("digimulti_layer1ring8","Digi Multiplicity ",30, 0., 30.);

   meDigiMultiLayer2Ring1_ =  dbe_->book1D("digimulti_layer2ring1","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer2Ring2_ =  dbe_->book1D("digimulti_layer2ring2","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer2Ring3_ =  dbe_->book1D("digimulti_layer2ring3","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer2Ring4_ =  dbe_->book1D("digimulti_layer2ring4","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer2Ring5_ =  dbe_->book1D("digimulti_layer2ring5","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer2Ring6_ =  dbe_->book1D("digimulti_layer2ring6","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer2Ring7_ =  dbe_->book1D("digimulti_layer2ring7","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer2Ring8_ =  dbe_->book1D("digimulti_layer2ring8","Digi Multiplicity ",30, 0., 30.);

   meDigiMultiLayer3Ring1_ =  dbe_->book1D("digimulti_layer3ring1","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer3Ring2_ =  dbe_->book1D("digimulti_layer3ring2","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer3Ring3_ =  dbe_->book1D("digimulti_layer3ring3","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer3Ring4_ =  dbe_->book1D("digimulti_layer3ring4","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer3Ring5_ =  dbe_->book1D("digimulti_layer3ring5","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer3Ring6_ =  dbe_->book1D("digimulti_layer3ring6","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer3Ring7_ =  dbe_->book1D("digimulti_layer3ring7","Digi Multiplicity ",30, 0., 30.);
   meDigiMultiLayer3Ring8_ =  dbe_->book1D("digimulti_layer3ring8","Digi Multiplicity ",30, 0., 30.);
  
  /////Barrel 
   meAdcLayer1Ring1_ = dbe_->book1D("adc_layer1ring1","Digi charge",50, 0., 300.);
   meAdcLayer1Ring2_ = dbe_->book1D("adc_layer1ring2","Digi charge",50, 0., 300.);
   meAdcLayer1Ring3_ = dbe_->book1D("adc_layer1ring3","Digi charge",50, 0., 300.);
   meAdcLayer1Ring4_ = dbe_->book1D("adc_layer1ring4","Digi charge",50, 0., 300.);
   meAdcLayer1Ring5_ = dbe_->book1D("adc_layer1ring5","Digi charge",50, 0., 300.);
   meAdcLayer1Ring6_ = dbe_->book1D("adc_layer1ring6","Digi charge",50, 0., 300.);
   meAdcLayer1Ring7_ = dbe_->book1D("adc_layer1ring7","Digi charge",50, 0., 300.);
   meAdcLayer1Ring8_ = dbe_->book1D("adc_layer1ring8","Digi charge",50, 0., 300.);

   meRowLayer1Ring1_ = dbe_->book1D("row_layer1ring1","Digi row",50, 0., 200.);
   meRowLayer1Ring2_ = dbe_->book1D("row_layer1ring2","Digi row",50, 0., 200.);
   meRowLayer1Ring3_ = dbe_->book1D("row_layer1ring3","Digi row",50, 0., 200.);
   meRowLayer1Ring4_ = dbe_->book1D("row_layer1ring4","Digi row",50, 0., 200.);
   meRowLayer1Ring5_ = dbe_->book1D("row_layer1ring5","Digi row",50, 0., 200.);
   meRowLayer1Ring6_ = dbe_->book1D("row_layer1ring6","Digi row",50, 0., 200.);
   meRowLayer1Ring7_ = dbe_->book1D("row_layer1ring7","Digi row",50, 0., 200.);
   meRowLayer1Ring8_ = dbe_->book1D("row_layer1ring8","Digi row",50, 0., 200.);

   meColLayer1Ring1_ = dbe_->book1D("col_layer1ring1","Digi column",50, 0., 500.);
   meColLayer1Ring2_ = dbe_->book1D("col_layer1ring2","Digi column",50, 0., 500.);
   meColLayer1Ring3_ = dbe_->book1D("col_layer1ring3","Digi column",50, 0., 500.);
   meColLayer1Ring4_ = dbe_->book1D("col_layer1ring4","Digi column",50, 0., 500.);
   meColLayer1Ring5_ = dbe_->book1D("col_layer1ring5","Digi column",50, 0., 500.);
   meColLayer1Ring6_ = dbe_->book1D("col_layer1ring6","Digi column",50, 0., 500.);
   meColLayer1Ring7_ = dbe_->book1D("col_layer1ring7","Digi column",50, 0., 500.);
   meColLayer1Ring8_ = dbe_->book1D("col_layer1ring8","Digi column",50, 0., 500.);
   

   meAdcLayer2Ring1_ = dbe_->book1D("adc_layer2ring1","Digi charge",50, 0., 300.);
   meAdcLayer2Ring2_ = dbe_->book1D("adc_layer2ring2","Digi charge",50, 0., 300.);
   meAdcLayer2Ring3_ = dbe_->book1D("adc_layer2ring3","Digi charge",50, 0., 300.);
   meAdcLayer2Ring4_ = dbe_->book1D("adc_layer2ring4","Digi charge",50, 0., 300.);
   meAdcLayer2Ring5_ = dbe_->book1D("adc_layer2ring5","Digi charge",50, 0., 300.);
   meAdcLayer2Ring6_ = dbe_->book1D("adc_layer2ring6","Digi charge",50, 0., 300.);
   meAdcLayer2Ring7_ = dbe_->book1D("adc_layer2ring7","Digi charge",50, 0., 300.);
   meAdcLayer2Ring8_ = dbe_->book1D("adc_layer2ring8","Digi charge",50, 0., 300.);

   meRowLayer2Ring1_ = dbe_->book1D("row_layer2ring1","Digi row",50, 0., 200.);
   meRowLayer2Ring2_ = dbe_->book1D("row_layer2ring2","Digi row",50, 0., 200.);
   meRowLayer2Ring3_ = dbe_->book1D("row_layer2ring3","Digi row",50, 0., 200.);
   meRowLayer2Ring4_ = dbe_->book1D("row_layer2ring4","Digi row",50, 0., 200.);
   meRowLayer2Ring5_ = dbe_->book1D("row_layer2ring5","Digi row",50, 0., 200.);
   meRowLayer2Ring6_ = dbe_->book1D("row_layer2ring6","Digi row",50, 0., 200.);
   meRowLayer2Ring7_ = dbe_->book1D("row_layer2ring7","Digi row",50, 0., 200.);
   meRowLayer2Ring8_ = dbe_->book1D("row_layer2ring8","Digi row",50, 0., 200.);

   meColLayer2Ring1_ = dbe_->book1D("col_layer2ring1","Digi column",50, 0., 500.);
   meColLayer2Ring2_ = dbe_->book1D("col_layer2ring2","Digi column",50, 0., 500.);
   meColLayer2Ring3_ = dbe_->book1D("col_layer2ring3","Digi column",50, 0., 500.);
   meColLayer2Ring4_ = dbe_->book1D("col_layer2ring4","Digi column",50, 0., 500.);
   meColLayer2Ring5_ = dbe_->book1D("col_layer2ring5","Digi column",50, 0., 500.);
   meColLayer2Ring6_ = dbe_->book1D("col_layer2ring6","Digi column",50, 0., 500.);
   meColLayer2Ring7_ = dbe_->book1D("col_layer2ring7","Digi column",50, 0., 500.);
   meColLayer2Ring8_ = dbe_->book1D("col_layer2ring8","Digi column",50, 0., 500.);


   meAdcLayer3Ring1_ = dbe_->book1D("adc_layer3ring1","Digi charge",50, 0., 300.);
   meAdcLayer3Ring2_ = dbe_->book1D("adc_layer3ring2","Digi charge",50, 0., 300.);
   meAdcLayer3Ring3_ = dbe_->book1D("adc_layer3ring3","Digi charge",50, 0., 300.);
   meAdcLayer3Ring4_ = dbe_->book1D("adc_layer3ring4","Digi charge",50, 0., 300.);
   meAdcLayer3Ring5_ = dbe_->book1D("adc_layer3ring5","Digi charge",50, 0., 300.);
   meAdcLayer3Ring6_ = dbe_->book1D("adc_layer3ring6","Digi charge",50, 0., 300.);
   meAdcLayer3Ring7_ = dbe_->book1D("adc_layer3ring7","Digi charge",50, 0., 300.);
   meAdcLayer3Ring8_ = dbe_->book1D("adc_layer3ring8","Digi charge",50, 0., 300.);

   meRowLayer3Ring1_ = dbe_->book1D("row_layer3ring1","Digi row",50, 0., 200.);
   meRowLayer3Ring2_ = dbe_->book1D("row_layer3ring2","Digi row",50, 0., 200.);
   meRowLayer3Ring3_ = dbe_->book1D("row_layer3ring3","Digi row",50, 0., 200.);
   meRowLayer3Ring4_ = dbe_->book1D("row_layer3ring4","Digi row",50, 0., 200.);
   meRowLayer3Ring5_ = dbe_->book1D("row_layer3ring5","Digi row",50, 0., 200.);
   meRowLayer3Ring6_ = dbe_->book1D("row_layer3ring6","Digi row",50, 0., 200.);
   meRowLayer3Ring7_ = dbe_->book1D("row_layer3ring7","Digi row",50, 0., 200.);
   meRowLayer3Ring8_ = dbe_->book1D("row_layer3ring8","Digi row",50, 0., 200.);

   meColLayer3Ring1_ = dbe_->book1D("col_layer3ring1","Digi column",50, 0., 500.);
   meColLayer3Ring2_ = dbe_->book1D("col_layer3ring2","Digi column",50, 0., 500.);
   meColLayer3Ring3_ = dbe_->book1D("col_layer3ring3","Digi column",50, 0., 500.);
   meColLayer3Ring4_ = dbe_->book1D("col_layer3ring4","Digi column",50, 0., 500.);
   meColLayer3Ring5_ = dbe_->book1D("col_layer3ring5","Digi column",50, 0., 500.);
   meColLayer3Ring6_ = dbe_->book1D("col_layer3ring6","Digi column",50, 0., 500.);
   meColLayer3Ring7_ = dbe_->book1D("col_layer3ring7","Digi column",50, 0., 500.);
   meColLayer3Ring8_ = dbe_->book1D("col_layer3ring8","Digi column",50, 0., 500.);

   meDigiMultiLayer1Ladders_  =dbe_->bookProfile("digi_layer1_ladders","Digi Num. per ladder",22,0.0,21.0, 100, 0.0, 100);
   meDigiMultiLayer2Ladders_  =dbe_->bookProfile("digi_layer2_ladders","Digi Num. per ladder",34,0.0,32.0, 100, 0.0, 100);
   meDigiMultiLayer3Ladders_  =dbe_->bookProfile("digi_layer3_ladders","Digi Num. per ladder",46,0.0,45.0, 100, 0.0, 100);
 
  //Forward Pixel
   /* ZMinus Side 1st Disk */
   meAdcZmDisk1Panel1Plaq1_ = dbe_->book1D("adc_zm_disk1_panel1_plaq1","Digi charge",50,0.,300.);
   meAdcZmDisk1Panel1Plaq2_ = dbe_->book1D("adc_zm_disk1_panel1_plaq2","Digi charge",50,0.,300.);
   meAdcZmDisk1Panel1Plaq3_ = dbe_->book1D("adc_zm_disk1_panel1_plaq3","Digi charge",50,0.,300.);
   meAdcZmDisk1Panel1Plaq4_ = dbe_->book1D("adc_zm_disk1_panel1_plaq4","Digi charge",50,0.,300.);
   meAdcZmDisk1Panel2Plaq1_ = dbe_->book1D("adc_zm_disk1_panel2_plaq1","Digi charge",50,0.,300.);
   meAdcZmDisk1Panel2Plaq2_ = dbe_->book1D("adc_zm_disk1_panel2_plaq2","Digi charge",50,0.,300.);
   meAdcZmDisk1Panel2Plaq3_ = dbe_->book1D("adc_zm_disk1_panel2_plaq3","Digi charge",50,0.,300.);

   meRowZmDisk1Panel1Plaq1_ = dbe_->book1D("row_zm_disk1_panel1_plaq1","Digi row",50,0.,100.);
   meRowZmDisk1Panel1Plaq2_ = dbe_->book1D("row_zm_disk1_panel1_plaq2","Digi row",50,0.,200.);
   meRowZmDisk1Panel1Plaq3_ = dbe_->book1D("row_zm_disk1_panel1_plaq3","Digi row",50,0.,200.);
   meRowZmDisk1Panel1Plaq4_ = dbe_->book1D("row_zm_disk1_panel1_plaq4","Digi row",50,0.,100.);
   meRowZmDisk1Panel2Plaq1_ = dbe_->book1D("row_zm_disk1_panel2_plaq1","Digi row",50,0.,200.);
   meRowZmDisk1Panel2Plaq2_ = dbe_->book1D("row_zm_disk1_panel2_plaq2","Digi row",50,0.,200.);
   meRowZmDisk1Panel2Plaq3_ = dbe_->book1D("row_zm_disk1_panel2_plaq3","Digi row",50,0.,200.);

   meColZmDisk1Panel1Plaq1_ = dbe_->book1D("col_zm_disk1_panel1_plaq1","Digi column",50,0.,150.);
   meColZmDisk1Panel1Plaq2_ = dbe_->book1D("col_zm_disk1_panel1_plaq2","Digi column",50,0.,200.);
   meColZmDisk1Panel1Plaq3_ = dbe_->book1D("col_zm_disk1_panel1_plaq3","Digi column",50,0.,250.);
   meColZmDisk1Panel1Plaq4_ = dbe_->book1D("col_zm_disk1_panel1_plaq4","Digi column",50,0.,300.);
   meColZmDisk1Panel2Plaq1_ = dbe_->book1D("col_zm_disk1_panel2_plaq1","Digi column",50,0.,200.);
   meColZmDisk1Panel2Plaq2_ = dbe_->book1D("col_zm_disk1_panel2_plaq2","Digi column",50,0.,250.);
   meColZmDisk1Panel2Plaq3_ = dbe_->book1D("col_zm_disk1_panel2_plaq3","Digi column",50,0.,300.);
   meNdigiZmDisk1PerPanel1_ = dbe_->book1D("digi_zm_disk1_panel1","Digi Num. Panel1 Of 1st Disk In ZMinus Side ",30,0.,30.);
   meNdigiZmDisk1PerPanel2_ = dbe_->book1D("digi_zm_disk1_panel2","Digi Num. Panel2 Of 1st Disk In ZMinus Side ",30,0.,30.);

   /* ZMius Side 2nd disk */
   meAdcZmDisk2Panel1Plaq1_ = dbe_->book1D("adc_zm_disk2_panel1_plaq1","Digi charge",50,0.,300.);
   meAdcZmDisk2Panel1Plaq2_ = dbe_->book1D("adc_zm_disk2_panel1_plaq2","Digi charge",50,0.,300.);
   meAdcZmDisk2Panel1Plaq3_ = dbe_->book1D("adc_zm_disk2_panel1_plaq3","Digi charge",50,0.,300.);
   meAdcZmDisk2Panel1Plaq4_ = dbe_->book1D("adc_zm_disk2_panel1_plaq4","Digi charge",50,0.,300.);
   meAdcZmDisk2Panel2Plaq1_ = dbe_->book1D("adc_zm_disk2_panel2_plaq1","Digi charge",50,0.,300.);
   meAdcZmDisk2Panel2Plaq2_ = dbe_->book1D("adc_zm_disk2_panel2_plaq2","Digi charge",50,0.,300.);
   meAdcZmDisk2Panel2Plaq3_ = dbe_->book1D("adc_zm_disk2_panel2_plaq3","Digi charge",50,0.,300.);

   meRowZmDisk2Panel1Plaq1_ = dbe_->book1D("row_zm_disk2_panel1_plaq1","Digi row",50,0.,100.);
   meRowZmDisk2Panel1Plaq2_ = dbe_->book1D("row_zm_disk2_panel1_plaq2","Digi row",50,0.,200.);
   meRowZmDisk2Panel1Plaq3_ = dbe_->book1D("row_zm_disk2_panel1_plaq3","Digi row",50,0.,200.);
   meRowZmDisk2Panel1Plaq4_ = dbe_->book1D("row_zm_disk2_panel1_plaq4","Digi row",50,0.,100.);
   meRowZmDisk2Panel2Plaq1_ = dbe_->book1D("row_zm_disk2_panel2_plaq1","Digi row",50,0.,200.);
   meRowZmDisk2Panel2Plaq2_ = dbe_->book1D("row_zm_disk2_panel2_plaq2","Digi row",50,0.,200.);
   meRowZmDisk2Panel2Plaq3_ = dbe_->book1D("row_zm_disk2_panel2_plaq3","Digi row",50,0.,200.);

   meColZmDisk2Panel1Plaq1_ = dbe_->book1D("col_zm_disk2_panel1_plaq1","Digi Column",50,0.,150.);
   meColZmDisk2Panel1Plaq2_ = dbe_->book1D("col_zm_disk2_panel1_plaq2","Digi Column",50,0.,200.);
   meColZmDisk2Panel1Plaq3_ = dbe_->book1D("col_zm_disk2_panel1_plaq3","Digi Column",50,0.,250.);
   meColZmDisk2Panel1Plaq4_ = dbe_->book1D("col_zm_disk2_panel1_plaq4","Digi Column",50,0.,300.);
   meColZmDisk2Panel2Plaq1_ = dbe_->book1D("col_zm_disk2_panel2_plaq1","Digi Column",50,0.,200.);
   meColZmDisk2Panel2Plaq2_ = dbe_->book1D("col_zm_disk2_panel2_plaq2","Digi Column",50,0.,250.);
   meColZmDisk2Panel2Plaq3_ = dbe_->book1D("col_zm_disk2_panel2_plaq3","Digi Column",50,0.,300.);
   meNdigiZmDisk2PerPanel1_ = dbe_->book1D("digi_zm_disk2_panel1","Digi Num. Panel1 Of 2nd Disk In ZMinus Side ",30,0.,30.);
   meNdigiZmDisk2PerPanel2_ = dbe_->book1D("digi_zm_disk2_panel2","Digi Num. Panel2 Of 2nd Disk In ZMinus Side ",30,0.,30.);


   /* ZPlus Side 1st Disk */
   meAdcZpDisk1Panel1Plaq1_ = dbe_->book1D("adc_zp_disk1_panel1_plaq1","Digi charge",50,0.,300.);
   meAdcZpDisk1Panel1Plaq2_ = dbe_->book1D("adc_zp_disk1_panel1_plaq2","Digi charge",50,0.,300.);
   meAdcZpDisk1Panel1Plaq3_ = dbe_->book1D("adc_zp_disk1_panel1_plaq3","Digi charge",50,0.,300.);
   meAdcZpDisk1Panel1Plaq4_ = dbe_->book1D("adc_zp_disk1_panel1_plaq4","Digi charge",50,0.,300.);
   meAdcZpDisk1Panel2Plaq1_ = dbe_->book1D("adc_zp_disk1_panel2_plaq1","Digi charge",50,0.,300.);
   meAdcZpDisk1Panel2Plaq2_ = dbe_->book1D("adc_zp_disk1_panel2_plaq2","Digi charge",50,0.,300.);
   meAdcZpDisk1Panel2Plaq3_ = dbe_->book1D("adc_zp_disk1_panel2_plaq3","Digi charge",50,0.,300.);

   meRowZpDisk1Panel1Plaq1_ = dbe_->book1D("row_zp_disk1_panel1_plaq1","Digi row",50,0.,100.);
   meRowZpDisk1Panel1Plaq2_ = dbe_->book1D("row_zp_disk1_panel1_plaq2","Digi row",50,0.,200.);
   meRowZpDisk1Panel1Plaq3_ = dbe_->book1D("row_zp_disk1_panel1_plaq3","Digi row",50,0.,200.);
   meRowZpDisk1Panel1Plaq4_ = dbe_->book1D("row_zp_disk1_panel1_plaq4","Digi row",50,0.,100.);
   meRowZpDisk1Panel2Plaq1_ = dbe_->book1D("row_zp_disk1_panel2_plaq1","Digi row",50,0.,200.);
   meRowZpDisk1Panel2Plaq2_ = dbe_->book1D("row_zp_disk1_panel2_plaq2","Digi row",50,0.,200.);
   meRowZpDisk1Panel2Plaq3_ = dbe_->book1D("row_zp_disk1_panel2_plaq3","Digi row",50,0.,200.);

   meColZpDisk1Panel1Plaq1_ = dbe_->book1D("col_zp_disk1_panel1_plaq1","Digi Column",50,0.,150.);
   meColZpDisk1Panel1Plaq2_ = dbe_->book1D("col_zp_disk1_panel1_plaq2","Digi column",50,0.,200.);
   meColZpDisk1Panel1Plaq3_ = dbe_->book1D("col_zp_disk1_panel1_plaq3","Digi column",50,0.,250.);
   meColZpDisk1Panel1Plaq4_ = dbe_->book1D("col_zp_disk1_panel1_plaq4","Digi column",50,0.,300.);
   meColZpDisk1Panel2Plaq1_ = dbe_->book1D("col_zp_disk1_panel2_plaq1","Digi column",50,0.,200.);
   meColZpDisk1Panel2Plaq2_ = dbe_->book1D("col_zp_disk1_panel2_plaq2","Digi column",50,0.,250.);
   meColZpDisk1Panel2Plaq3_ = dbe_->book1D("col_zp_disk1_panel2_plaq3","Digi column",50,0.,300.);
   meNdigiZpDisk1PerPanel1_ = dbe_->book1D("digi_zp_disk1_panel1","Digi Num. Panel1 Of 1st Disk In ZPlus Side ",30,0.,30.);
   meNdigiZpDisk1PerPanel2_ = dbe_->book1D("digi_zp_disk1_panel2","Digi Num. Panel2 Of 1st Disk In ZPlus Side ",30,0.,30.);


   /* ZPlus Side 2nd disk */
   meAdcZpDisk2Panel1Plaq1_ = dbe_->book1D("adc_zp_disk2_panel1_plaq1","Digi charge",50,0.,300.);
   meAdcZpDisk2Panel1Plaq2_ = dbe_->book1D("adc_zp_disk2_panel1_plaq2","Digi charge",50,0.,300.);
   meAdcZpDisk2Panel1Plaq3_ = dbe_->book1D("adc_zp_disk2_panel1_plaq3","Digi charge",50,0.,300.);
   meAdcZpDisk2Panel1Plaq4_ = dbe_->book1D("adc_zp_disk2_panel1_plaq4","Digi charge",50,0.,300.);
   meAdcZpDisk2Panel2Plaq1_ = dbe_->book1D("adc_zp_disk2_panel2_plaq1","Digi charge",50,0.,300.);
   meAdcZpDisk2Panel2Plaq2_ = dbe_->book1D("adc_zp_disk2_panel2_plaq2","Digi charge",50,0.,300.);
   meAdcZpDisk2Panel2Plaq3_ = dbe_->book1D("adc_zp_disk2_panel2_plaq3","Digi charge",50,0.,300.);

   meRowZpDisk2Panel1Plaq1_ = dbe_->book1D("row_zp_disk2_panel1_plaq1","Digi row",10,0.,100.);
   meRowZpDisk2Panel1Plaq2_ = dbe_->book1D("row_zp_disk2_panel1_plaq2","Digi row",10,0.,200.);
   meRowZpDisk2Panel1Plaq3_ = dbe_->book1D("row_zp_disk2_panel1_plaq3","Digi row",10,0.,200.);
   meRowZpDisk2Panel1Plaq4_ = dbe_->book1D("row_zp_disk2_panel1_plaq4","Digi row",10,0.,100.);
   meRowZpDisk2Panel2Plaq1_ = dbe_->book1D("row_zp_disk2_panel2_plaq1","Digi row",10,0.,200.);
   meRowZpDisk2Panel2Plaq2_ = dbe_->book1D("row_zp_disk2_panel2_plaq2","Digi row",10,0.,200.);
   meRowZpDisk2Panel2Plaq3_ = dbe_->book1D("row_zp_disk2_panel2_plaq3","Digi row",10,0.,200.);

   meColZpDisk2Panel1Plaq1_ = dbe_->book1D("col_zp_disk2_panel1_plaq1","Digi column",50,0.,150.);
   meColZpDisk2Panel1Plaq2_ = dbe_->book1D("col_zp_disk2_panel1_plaq2","Digi column",50,0.,200.);
   meColZpDisk2Panel1Plaq3_ = dbe_->book1D("col_zp_disk2_panel1_plaq3","Digi column",50,0.,250.);
   meColZpDisk2Panel1Plaq4_ = dbe_->book1D("col_zp_disk2_panel1_plaq4","Digi column",50,0.,300.);
   meColZpDisk2Panel2Plaq1_ = dbe_->book1D("col_zp_disk2_panel2_plaq1","Digi column",50,0.,200.);
   meColZpDisk2Panel2Plaq2_ = dbe_->book1D("col_zp_disk2_panel2_plaq2","Digi column",50,0.,250.);
   meColZpDisk2Panel2Plaq3_ = dbe_->book1D("col_zp_disk2_panel2_plaq3","Digi column",50,0.,300.);
   meNdigiZpDisk2PerPanel1_ = dbe_->book1D("digi_zp_disk2_panel1","Digi Num. Panel1 Of 2nd Disk In ZPlus Side ",30,0.,30.);
   meNdigiZpDisk2PerPanel2_ = dbe_->book1D("digi_zp_disk2_panel2","Digi Num. Panel2 Of 2nd Disk In ZPlus Side ",30,0.,30.);

 }
 
}

SiPixelDigiValid::~SiPixelDigiValid(){
 
 // if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void SiPixelDigiValid::beginJob(){

}

void SiPixelDigiValid::endJob() {
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void SiPixelDigiValid::analyze(const Event& e, const EventSetup& c){
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  c.get<IdealGeometryRecord>().get(tTopo);



 int ndigiperRingLayer1[8];
 int ndigiperRingLayer2[8];
 int ndigiperRingLayer3[8];
 for(int i = 0; i< 8; i++ ) {
    ndigiperRingLayer1[i] = 0;
    ndigiperRingLayer2[i] = 0;
    ndigiperRingLayer3[i] = 0;
 }

int ndigiZpDisk1PerPanel1[24];
int ndigiZpDisk1PerPanel2[24];
int ndigiZpDisk2PerPanel1[24];
int ndigiZpDisk2PerPanel2[24];
int ndigiZmDisk1PerPanel1[24];
int ndigiZmDisk1PerPanel2[24];
int ndigiZmDisk2PerPanel1[24];
int ndigiZmDisk2PerPanel2[24];

for ( int i =0 ; i< 24; i++) {
   ndigiZpDisk1PerPanel1[i] = 0;
   ndigiZpDisk1PerPanel2[i] = 0;
   ndigiZpDisk2PerPanel1[i] = 0;
   ndigiZpDisk2PerPanel2[i] = 0;
   ndigiZmDisk1PerPanel1[i] = 0;
   ndigiZmDisk1PerPanel2[i] = 0;
   ndigiZmDisk2PerPanel1[i] = 0;
   ndigiZmDisk2PerPanel2[i] = 0;
}

int ndigilayer1ladders[20];
int ndigilayer2ladders[32];
int ndigilayer3ladders[44];

for ( int i =0 ; i< 20; i++) {
   ndigilayer1ladders[i]= 0;
}

for ( int i =0 ; i< 32; i++) {
   ndigilayer2ladders[i] = 0;
}

for ( int i =0 ; i< 44; i++) {
   ndigilayer3ladders[i] = 0;
}

 //LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

 edm::ESHandle<TrackerGeometry> tracker;
 c.get<TrackerDigiGeometryRecord>().get( tracker );     

 //string digiProducer = "siPixelDigis";
 edm::Handle<edm::DetSetVector<PixelDigi> > pixelDigis;
 e.getByLabel(src_, pixelDigis);

 edm::DetSetVector<PixelDigi>::const_iterator DSViter = pixelDigis->begin();
 for( ; DSViter != pixelDigis->end(); DSViter++) {
          unsigned int id = DSViter->id;
          DetId  detId(id);
          edm::DetSet<PixelDigi>::const_iterator  begin = DSViter->data.begin();
          edm::DetSet<PixelDigi>::const_iterator  end   = DSViter->data.end();
          edm::DetSet<PixelDigi>::const_iterator  iter;          

          if(detId.subdetId()==PixelSubdetector::PixelBarrel ) {
             
             unsigned int layer  = tTopo->pxbLayer(id);   // Layer:1,2,3.
             unsigned int ladder = tTopo->pxbLadder(id);  // Ladeer: 1-20, 32, 44. 
             unsigned int zindex = tTopo->pxbModule(id);  // Z-index: 1-8.
             //LogInfo("SiPixelDigiValid")<<"Barrel:: Layer="<<layer<<" Ladder="<<ladder<<" zindex="<<zindex;
             for ( iter = begin ; iter != end; iter++ ) {
                if( layer == 1 ) {
                     ++ndigilayer1ladders[ladder-1];
                     ++ndigiperRingLayer1[zindex-1];
                     if (zindex == 1) { 
                          meAdcLayer1Ring1_->Fill((*iter).adc());
                          meRowLayer1Ring1_->Fill((*iter).row());
                          meColLayer1Ring1_->Fill((*iter).column());
                     }
                     if (zindex == 2) {
                          meAdcLayer1Ring2_->Fill((*iter).adc());
                          meRowLayer1Ring2_->Fill((*iter).row());
                          meColLayer1Ring2_->Fill((*iter).column());
                     }

                     if (zindex == 3) {
                          meAdcLayer1Ring3_->Fill((*iter).adc());
                          meRowLayer1Ring3_->Fill((*iter).row());
                          meColLayer1Ring3_->Fill((*iter).column());
                     }

                     if (zindex == 4)  {
                         meAdcLayer1Ring4_->Fill((*iter).adc());
                         meRowLayer1Ring4_->Fill((*iter).row());
                         meColLayer1Ring4_->Fill((*iter).column());
                     }

                     if (zindex == 5)  {
                         meAdcLayer1Ring5_->Fill((*iter).adc());
                         meRowLayer1Ring5_->Fill((*iter).row());
                         meColLayer1Ring5_->Fill((*iter).column());
                     }

                     if (zindex == 6)  {
                         meAdcLayer1Ring6_->Fill((*iter).adc());
                         meRowLayer1Ring6_->Fill((*iter).row());
                         meColLayer1Ring6_->Fill((*iter).column());
                     }

                     if (zindex == 7)  {
                         meAdcLayer1Ring7_->Fill((*iter).adc());
                         meRowLayer1Ring7_->Fill((*iter).row());
                         meColLayer1Ring7_->Fill((*iter).column());
                     }
                     if (zindex == 8)  {
                         meAdcLayer1Ring8_->Fill((*iter).adc());
                         meRowLayer1Ring8_->Fill((*iter).row());
                         meColLayer1Ring8_->Fill((*iter).column());
                     }

                } 
                if( layer == 2 ) {
                    ++ndigilayer2ladders[ladder-1];
                    ++ndigiperRingLayer2[zindex-1];
                    if (zindex == 1) {
                          meAdcLayer2Ring1_->Fill((*iter).adc());
                          meRowLayer2Ring1_->Fill((*iter).row());
                          meColLayer2Ring1_->Fill((*iter).column());
                     }
                     if (zindex == 2) {
                          meAdcLayer2Ring2_->Fill((*iter).adc());
                          meRowLayer2Ring2_->Fill((*iter).row());
                          meColLayer2Ring2_->Fill((*iter).column());
                     }

                     if (zindex == 3) {
                          meAdcLayer2Ring3_->Fill((*iter).adc());
                          meRowLayer2Ring3_->Fill((*iter).row());
                          meColLayer2Ring3_->Fill((*iter).column());
                     }

                     if (zindex == 4)  {
                         meAdcLayer2Ring4_->Fill((*iter).adc());
                         meRowLayer2Ring4_->Fill((*iter).row());
                         meColLayer2Ring4_->Fill((*iter).column());
                     }

                     if (zindex == 5)  {
                         meAdcLayer2Ring5_->Fill((*iter).adc());
                         meRowLayer2Ring5_->Fill((*iter).row());
                         meColLayer2Ring5_->Fill((*iter).column());
                     }

                     if (zindex == 6)  {
                         meAdcLayer2Ring6_->Fill((*iter).adc());
                         meRowLayer2Ring6_->Fill((*iter).row());
                         meColLayer2Ring6_->Fill((*iter).column());
                     }

                     if (zindex == 7)  {
                         meAdcLayer2Ring7_->Fill((*iter).adc());
                         meRowLayer2Ring7_->Fill((*iter).row());
                         meColLayer2Ring7_->Fill((*iter).column());
                     }
                     if (zindex == 8)  {
                         meAdcLayer2Ring8_->Fill((*iter).adc());
                         meRowLayer2Ring8_->Fill((*iter).row());
                         meColLayer2Ring8_->Fill((*iter).column());
                     }

                }
                if( layer == 3 ) {
                    ++ndigilayer3ladders[ladder-1];
                    ++ndigiperRingLayer3[zindex-1];
                    if (zindex == 1) {
                          meAdcLayer3Ring1_->Fill((*iter).adc());
                          meRowLayer3Ring1_->Fill((*iter).row());
                          meColLayer3Ring1_->Fill((*iter).column());
                     }
                     if (zindex == 2) {
                          meAdcLayer3Ring2_->Fill((*iter).adc());
                          meRowLayer3Ring2_->Fill((*iter).row());
                          meColLayer3Ring2_->Fill((*iter).column());
                     }

                     if (zindex == 3) {
                          meAdcLayer3Ring3_->Fill((*iter).adc());
                          meRowLayer3Ring3_->Fill((*iter).row());
                          meColLayer3Ring3_->Fill((*iter).column());
                     }

                     if (zindex == 4)  {
                         meAdcLayer3Ring4_->Fill((*iter).adc());
                         meRowLayer3Ring4_->Fill((*iter).row());
                         meColLayer3Ring4_->Fill((*iter).column());
                     }

                     if (zindex == 5)  {
                         meAdcLayer3Ring5_->Fill((*iter).adc());
                         meRowLayer3Ring5_->Fill((*iter).row());
                         meColLayer3Ring5_->Fill((*iter).column());
                     }

                     if (zindex == 6)  {
                         meAdcLayer3Ring6_->Fill((*iter).adc());
                         meRowLayer3Ring6_->Fill((*iter).row());
                         meColLayer3Ring6_->Fill((*iter).column());
                     }

                     if (zindex == 7)  {
                         meAdcLayer3Ring7_->Fill((*iter).adc());
                         meRowLayer3Ring7_->Fill((*iter).row());
                         meColLayer3Ring7_->Fill((*iter).column());
                     }
                     if (zindex == 8)  {
                         meAdcLayer3Ring8_->Fill((*iter).adc());
                         meRowLayer3Ring8_->Fill((*iter).row());
                         meColLayer3Ring8_->Fill((*iter).column());
                     }
                }
 
             }   
           
          }
 
         if(detId.subdetId()==PixelSubdetector::PixelEndcap ){ //Endcap
           
           unsigned int side  = tTopo->pxfSide(id);
           unsigned int disk  = tTopo->pxfDisk(id);
           unsigned int blade = tTopo->pxfBlade(id);
           unsigned int panel = tTopo->pxfPanel(id);
           unsigned int mod   = tTopo->pxfModule(id);
           //LogInfo("SiPixelDigiValid")<<"EndcaP="<<side<<" Disk="<<disk<<" Blade="<<blade<<" Panel="<<panel<<" Module="<<mod;
           for ( iter = begin ; iter != end; iter++ ) {
             if(side == 1 && disk == 1 && panel ==1 ){
                     if ( mod == 1 ) {
                         meAdcZmDisk1Panel1Plaq1_->Fill((*iter).adc());
                         meRowZmDisk1Panel1Plaq1_->Fill((*iter).row());
                         meColZmDisk1Panel1Plaq1_->Fill((*iter).column());
                     }else if( mod == 2 ) {
                         meAdcZmDisk1Panel1Plaq2_->Fill((*iter).adc());
                         meRowZmDisk1Panel1Plaq2_->Fill((*iter).row());
                         meColZmDisk1Panel1Plaq2_->Fill((*iter).column());
                     }else if( mod == 3 ) {
                         meAdcZmDisk1Panel1Plaq3_->Fill((*iter).adc());
                         meRowZmDisk1Panel1Plaq3_->Fill((*iter).row());
                         meColZmDisk1Panel1Plaq3_->Fill((*iter).column());
                     }else if( mod == 4 ) {
                         meAdcZmDisk1Panel1Plaq4_->Fill((*iter).adc());
                         meRowZmDisk1Panel1Plaq4_->Fill((*iter).row());
                         meColZmDisk1Panel1Plaq4_->Fill((*iter).column());
                     }else {
                         //LogError("SiPixelDigiValid")<<" The number of module is Wrong";
                     } 
                     ++ndigiZmDisk1PerPanel1[blade-1];                
             }

             if(side == 1 && disk == 1 && panel ==2 ){
                     if ( mod == 1 ) {
                         meAdcZmDisk1Panel2Plaq1_->Fill((*iter).adc());
                         meRowZmDisk1Panel2Plaq1_->Fill((*iter).row());
                         meColZmDisk1Panel2Plaq1_->Fill((*iter).column());
                     }else if( mod == 2 ) {
                         meAdcZmDisk1Panel2Plaq2_->Fill((*iter).adc());
                         meRowZmDisk1Panel2Plaq2_->Fill((*iter).row());
                         meColZmDisk1Panel2Plaq2_->Fill((*iter).column());
                     }else if( mod == 3 ) {
                         meAdcZmDisk1Panel2Plaq3_->Fill((*iter).adc());
                         meRowZmDisk1Panel2Plaq3_->Fill((*iter).row());
                         meColZmDisk1Panel2Plaq3_->Fill((*iter).column());
                     }else {
                         //LogError("SiPixelDigiValid")<<" The number of module is Wrong";
                     }
                     ++ndigiZmDisk1PerPanel2[blade-1];
             }

            if(side == 1 && disk == 2 && panel ==1 ){
                     if ( mod == 1 ) {
                         meAdcZmDisk2Panel1Plaq1_->Fill((*iter).adc());
                         meRowZmDisk2Panel1Plaq1_->Fill((*iter).row());
                         meColZmDisk2Panel1Plaq1_->Fill((*iter).column());
                     }else if( mod == 2 ) {
                         meAdcZmDisk2Panel1Plaq2_->Fill((*iter).adc());
                         meRowZmDisk2Panel1Plaq2_->Fill((*iter).row());
                         meColZmDisk2Panel1Plaq2_->Fill((*iter).column());
                     }else if( mod == 3 ) {
                         meAdcZmDisk2Panel1Plaq3_->Fill((*iter).adc());
                         meRowZmDisk2Panel1Plaq3_->Fill((*iter).row());
                         meColZmDisk2Panel1Plaq3_->Fill((*iter).column());
                     }else if( mod == 4 ) {
                         meAdcZmDisk2Panel1Plaq4_->Fill((*iter).adc());
                         meRowZmDisk2Panel1Plaq4_->Fill((*iter).row());
                         meColZmDisk2Panel1Plaq4_->Fill((*iter).column());
                     }else {
                        // LogError("SiPixelDigiValid")<<" The number of module is Wrong";
                     }
                     ++ndigiZmDisk2PerPanel1[blade-1];
             }

             if(side == 1 && disk == 2 && panel ==2 ){
                     if ( mod == 1 ) {
                         meAdcZmDisk2Panel2Plaq1_->Fill((*iter).adc());
                         meRowZmDisk2Panel2Plaq1_->Fill((*iter).row());
                         meColZmDisk2Panel2Plaq1_->Fill((*iter).column());
                     }else if( mod == 2 ) {
                         meAdcZmDisk2Panel2Plaq2_->Fill((*iter).adc());
                         meRowZmDisk2Panel2Plaq2_->Fill((*iter).row());
                         meColZmDisk2Panel2Plaq2_->Fill((*iter).column());
                     }else if( mod == 3 ) {
                         meAdcZmDisk2Panel2Plaq3_->Fill((*iter).adc());
                         meRowZmDisk2Panel2Plaq3_->Fill((*iter).row());
                         meColZmDisk2Panel2Plaq3_->Fill((*iter).column());
                     }else {
                         //LogError("SiPixelDigiValid")<<" The number of module is Wrong";
                     }
                     ++ndigiZmDisk2PerPanel2[blade-1];
             }


            if(side == 2 && disk == 1 && panel ==1 ){
                     if ( mod == 1 ) {
                         meAdcZpDisk1Panel1Plaq1_->Fill((*iter).adc());
                         meRowZpDisk1Panel1Plaq1_->Fill((*iter).row());
                         meColZpDisk1Panel1Plaq1_->Fill((*iter).column());
                     }else if( mod == 2 ) {
                         meAdcZpDisk1Panel1Plaq2_->Fill((*iter).adc());
                         meRowZpDisk1Panel1Plaq2_->Fill((*iter).row());
                         meColZpDisk1Panel1Plaq2_->Fill((*iter).column());
                     }else if( mod == 3 ) {
                         meAdcZpDisk1Panel1Plaq3_->Fill((*iter).adc());
                         meRowZpDisk1Panel1Plaq3_->Fill((*iter).row());
                         meColZpDisk1Panel1Plaq3_->Fill((*iter).column());
                     }else if( mod == 4 ) {
                         meAdcZpDisk1Panel1Plaq4_->Fill((*iter).adc());
                         meRowZpDisk1Panel1Plaq4_->Fill((*iter).row());
                         meColZpDisk1Panel1Plaq4_->Fill((*iter).column());
                     }else {
                         //LogError("SiPixelDigiValid")<<" The number of module is Wrong";
                     }
                     ++ndigiZpDisk1PerPanel1[blade-1];
             }

             if(side == 2 && disk == 1 && panel ==2 ){
                     if ( mod == 1 ) {
                         meAdcZpDisk1Panel2Plaq1_->Fill((*iter).adc());
                         meRowZpDisk1Panel2Plaq1_->Fill((*iter).row());
                         meColZpDisk1Panel2Plaq1_->Fill((*iter).column());
                     }else if( mod == 2 ) {
                         meAdcZpDisk1Panel2Plaq2_->Fill((*iter).adc());
                         meRowZpDisk1Panel2Plaq2_->Fill((*iter).row());
                         meColZpDisk1Panel2Plaq2_->Fill((*iter).column());
                     }else if( mod == 3 ) {
                         meAdcZpDisk1Panel2Plaq3_->Fill((*iter).adc());
                         meRowZpDisk1Panel2Plaq3_->Fill((*iter).row());
                         meColZpDisk1Panel2Plaq3_->Fill((*iter).column());
                     }else {
                         //LogError("SiPixelDigiValid")<<" The number of module is Wrong";
                     }
                     ++ndigiZpDisk1PerPanel2[blade-1];
             }

            if(side == 2 && disk == 2 && panel ==1 ){
                     if ( mod == 1 ) {
                         meAdcZpDisk2Panel1Plaq1_->Fill((*iter).adc());
                         meRowZpDisk2Panel1Plaq1_->Fill((*iter).row());
                         meColZpDisk2Panel1Plaq1_->Fill((*iter).column());
                     }else if( mod == 2 ) {
                         meAdcZpDisk2Panel1Plaq2_->Fill((*iter).adc());
                         meRowZpDisk2Panel1Plaq2_->Fill((*iter).row());
                         meColZpDisk2Panel1Plaq2_->Fill((*iter).column());
                     }else if( mod == 3 ) {
                         meAdcZpDisk2Panel1Plaq3_->Fill((*iter).adc());
                         meRowZpDisk2Panel1Plaq3_->Fill((*iter).row());
                         meColZpDisk2Panel1Plaq3_->Fill((*iter).column());
                     }else if( mod == 4 ) {
                         meAdcZpDisk2Panel1Plaq4_->Fill((*iter).adc());
                         meRowZpDisk2Panel1Plaq4_->Fill((*iter).row());
                         meColZpDisk2Panel1Plaq4_->Fill((*iter).column());
                     }else {
                         //LogError("SiPixelDigiValid")<<" The number of module is Wrong";
                     }
                     ++ndigiZpDisk2PerPanel1[blade-1];

              }

             if(side == 2 && disk == 2 && panel ==2 ){
                     if ( mod == 1 ) {
                         meAdcZpDisk2Panel2Plaq1_->Fill((*iter).adc());
                         meRowZpDisk2Panel2Plaq1_->Fill((*iter).row());
                         meColZpDisk2Panel2Plaq1_->Fill((*iter).column());
                     }else if( mod == 2 ) {
                         meAdcZpDisk2Panel2Plaq2_->Fill((*iter).adc());
                         meRowZpDisk2Panel2Plaq2_->Fill((*iter).row());
                         meColZpDisk2Panel2Plaq2_->Fill((*iter).column());
                     }else if( mod == 3 ) {
                         meAdcZpDisk2Panel2Plaq3_->Fill((*iter).adc());
                         meRowZpDisk2Panel2Plaq3_->Fill((*iter).row());
                         meColZpDisk2Panel2Plaq3_->Fill((*iter).column());
                     }else {
                         //LogError("SiPixelDigiValid")<<" The number of module is Wrong";
                     }
                     ++ndigiZpDisk2PerPanel2[blade-1];
              }
           } //iterating the digi 

          }//Endcap 

    } // end for loop
    
    meDigiMultiLayer1Ring1_->Fill(ndigiperRingLayer1[0]);
    meDigiMultiLayer1Ring2_->Fill(ndigiperRingLayer1[1]);
    meDigiMultiLayer1Ring3_->Fill(ndigiperRingLayer1[2]);
    meDigiMultiLayer1Ring4_->Fill(ndigiperRingLayer1[3]);
    meDigiMultiLayer1Ring5_->Fill(ndigiperRingLayer1[4]);
    meDigiMultiLayer1Ring6_->Fill(ndigiperRingLayer1[5]);
    meDigiMultiLayer1Ring7_->Fill(ndigiperRingLayer1[6]);
    meDigiMultiLayer1Ring8_->Fill(ndigiperRingLayer1[7]);

    meDigiMultiLayer2Ring1_->Fill(ndigiperRingLayer2[0]);
    meDigiMultiLayer2Ring2_->Fill(ndigiperRingLayer2[1]);
    meDigiMultiLayer2Ring3_->Fill(ndigiperRingLayer2[2]);
    meDigiMultiLayer2Ring4_->Fill(ndigiperRingLayer2[3]);
    meDigiMultiLayer2Ring5_->Fill(ndigiperRingLayer2[4]);
    meDigiMultiLayer2Ring6_->Fill(ndigiperRingLayer2[5]);
    meDigiMultiLayer2Ring7_->Fill(ndigiperRingLayer2[6]);
    meDigiMultiLayer2Ring8_->Fill(ndigiperRingLayer2[7]);

    meDigiMultiLayer3Ring1_->Fill(ndigiperRingLayer3[0]);
    meDigiMultiLayer3Ring2_->Fill(ndigiperRingLayer3[1]);
    meDigiMultiLayer3Ring3_->Fill(ndigiperRingLayer3[2]);
    meDigiMultiLayer3Ring4_->Fill(ndigiperRingLayer3[3]);
    meDigiMultiLayer3Ring5_->Fill(ndigiperRingLayer3[4]);
    meDigiMultiLayer3Ring6_->Fill(ndigiperRingLayer3[5]);
    meDigiMultiLayer3Ring7_->Fill(ndigiperRingLayer3[6]);
    meDigiMultiLayer3Ring8_->Fill(ndigiperRingLayer3[7]);

    for(int i =0; i< 24; i++) {
         meNdigiZmDisk1PerPanel1_->Fill(ndigiZmDisk1PerPanel1[i]);
         meNdigiZmDisk1PerPanel2_->Fill(ndigiZmDisk1PerPanel2[i]);
         meNdigiZmDisk2PerPanel1_->Fill(ndigiZmDisk2PerPanel1[i]);
         meNdigiZmDisk2PerPanel2_->Fill(ndigiZmDisk2PerPanel2[i]);
         meNdigiZpDisk1PerPanel1_->Fill(ndigiZpDisk1PerPanel1[i]);
         meNdigiZpDisk1PerPanel2_->Fill(ndigiZpDisk1PerPanel2[i]);
         meNdigiZpDisk2PerPanel1_->Fill(ndigiZpDisk2PerPanel1[i]);
         meNdigiZpDisk2PerPanel2_->Fill(ndigiZpDisk2PerPanel2[i]);
    } 
    
   for (int i =0; i< 20; i++) {
       meDigiMultiLayer1Ladders_->Fill(i+1,ndigilayer1ladders[i]);
   }

   for (int i =0; i< 32; i++) {
       meDigiMultiLayer2Ladders_->Fill(i+1,ndigilayer2ladders[i]);
   }

   for (int i =0; i< 44; i++) {
       meDigiMultiLayer3Ladders_->Fill(i+1,ndigilayer3ladders[i]);
   }

}
