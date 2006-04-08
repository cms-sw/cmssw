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
   
  /////Barrel 
   meAdcLayer1Ladder1_ = dbe_->book1D("adc_layer1ladder1","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder2_ = dbe_->book1D("adc_layer1ladder2","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder3_ = dbe_->book1D("adc_layer1ladder3","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder4_ = dbe_->book1D("adc_layer1ladder4","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder5_ = dbe_->book1D("adc_layer1ladder5","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder6_ = dbe_->book1D("adc_layer1ladder6","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder7_ = dbe_->book1D("adc_layer1ladder7","Digi charge",300, 0., 300.);
   meAdcLayer1Ladder8_ = dbe_->book1D("adc_layer1ladder8","Digi charge",300, 0., 300.);

   meRowLayer1Ladder1_ = dbe_->book1D("row_layer1ladder1","Digi row",200, 0., 200.);
   meRowLayer1Ladder2_ = dbe_->book1D("row_layer1ladder2","Digi row",200, 0., 200.);
   meRowLayer1Ladder3_ = dbe_->book1D("row_layer1ladder3","Digi row",200, 0., 200.);
   meRowLayer1Ladder4_ = dbe_->book1D("row_layer1ladder4","Digi row",200, 0., 200.);
   meRowLayer1Ladder5_ = dbe_->book1D("row_layer1ladder5","Digi row",200, 0., 200.);
   meRowLayer1Ladder6_ = dbe_->book1D("row_layer1ladder6","Digi row",200, 0., 200.);
   meRowLayer1Ladder7_ = dbe_->book1D("row_layer1ladder7","Digi row",200, 0., 200.);
   meRowLayer1Ladder8_ = dbe_->book1D("row_layer1ladder8","Digi row",200, 0., 200.);

   meColLayer1Ladder1_ = dbe_->book1D("col_layer1ladder1","Digi column",500, 0., 500.);
   meColLayer1Ladder2_ = dbe_->book1D("col_layer1ladder2","Digi column",500, 0., 500.);
   meColLayer1Ladder3_ = dbe_->book1D("col_layer1ladder3","Digi column",500, 0., 500.);
   meColLayer1Ladder4_ = dbe_->book1D("col_layer1ladder4","Digi column",500, 0., 500.);
   meColLayer1Ladder5_ = dbe_->book1D("col_layer1ladder5","Digi column",500, 0., 500.);
   meColLayer1Ladder6_ = dbe_->book1D("col_layer1ladder6","Digi column",500, 0., 500.);
   meColLayer1Ladder7_ = dbe_->book1D("col_layer1ladder7","Digi column",500, 0., 500.);
   meColLayer1Ladder8_ = dbe_->book1D("col_layer1ladder8","Digi column",500, 0., 500.);
   
   meNdigiPerLadderL1_ = dbe_->book2D("digi_multi_layer1","Digi Num. PerLadder",21,0.,21, 100,0., 100.);

   meAdcLayer2Ladder1_ = dbe_->book1D("adc_layer2ladder1","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder2_ = dbe_->book1D("adc_layer2ladder2","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder3_ = dbe_->book1D("adc_layer2ladder3","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder4_ = dbe_->book1D("adc_layer2ladder4","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder5_ = dbe_->book1D("adc_layer2ladder5","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder6_ = dbe_->book1D("adc_layer2ladder6","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder7_ = dbe_->book1D("adc_layer2ladder7","Digi charge",300, 0., 300.);
   meAdcLayer2Ladder8_ = dbe_->book1D("adc_layer2ladder8","Digi charge",300, 0., 300.);

   meRowLayer2Ladder1_ = dbe_->book1D("row_layer2ladder1","Digi row",200, 0., 200.);
   meRowLayer2Ladder2_ = dbe_->book1D("row_layer2ladder2","Digi row",200, 0., 200.);
   meRowLayer2Ladder3_ = dbe_->book1D("row_layer2ladder3","Digi row",200, 0., 200.);
   meRowLayer2Ladder4_ = dbe_->book1D("row_layer2ladder4","Digi row",200, 0., 200.);
   meRowLayer2Ladder5_ = dbe_->book1D("row_layer2ladder5","Digi row",200, 0., 200.);
   meRowLayer2Ladder6_ = dbe_->book1D("row_layer2ladder6","Digi row",200, 0., 200.);
   meRowLayer2Ladder7_ = dbe_->book1D("row_layer2ladder7","Digi row",200, 0., 200.);
   meRowLayer2Ladder8_ = dbe_->book1D("row_layer2ladder8","Digi row",200, 0., 200.);

   meColLayer2Ladder1_ = dbe_->book1D("col_layer2ladder1","Digi column",500, 0., 500.);
   meColLayer2Ladder2_ = dbe_->book1D("col_layer2ladder2","Digi column",500, 0., 500.);
   meColLayer2Ladder3_ = dbe_->book1D("col_layer2ladder3","Digi column",500, 0., 500.);
   meColLayer2Ladder4_ = dbe_->book1D("col_layer2ladder4","Digi column",500, 0., 500.);
   meColLayer2Ladder5_ = dbe_->book1D("col_layer2ladder5","Digi column",500, 0., 500.);
   meColLayer2Ladder6_ = dbe_->book1D("col_layer2ladder6","Digi column",500, 0., 500.);
   meColLayer2Ladder7_ = dbe_->book1D("col_layer2ladder7","Digi column",500, 0., 500.);
   meColLayer2Ladder8_ = dbe_->book1D("col_layer2ladder8","Digi column",500, 0., 500.);

   meNdigiPerLadderL2_ = dbe_->book2D("digi_multi_layer2","Digi Num. PerLadder",33,0.,33, 100,0., 100.);

   meAdcLayer3Ladder1_ = dbe_->book1D("adc_layer3ladder1","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder2_ = dbe_->book1D("adc_layer3ladder2","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder3_ = dbe_->book1D("adc_layer3ladder3","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder4_ = dbe_->book1D("adc_layer3ladder4","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder5_ = dbe_->book1D("adc_layer3ladder5","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder6_ = dbe_->book1D("adc_layer3ladder6","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder7_ = dbe_->book1D("adc_layer3ladder7","Digi charge",300, 0., 300.);
   meAdcLayer3Ladder8_ = dbe_->book1D("adc_layer3ladder8","Digi charge",300, 0., 300.);

   meRowLayer3Ladder1_ = dbe_->book1D("row_layer3ladder1","Digi row",200, 0., 200.);
   meRowLayer3Ladder2_ = dbe_->book1D("row_layer3ladder2","Digi row",200, 0., 200.);
   meRowLayer3Ladder3_ = dbe_->book1D("row_layer3ladder3","Digi row",200, 0., 200.);
   meRowLayer3Ladder4_ = dbe_->book1D("row_layer3ladder4","Digi row",200, 0., 200.);
   meRowLayer3Ladder5_ = dbe_->book1D("row_layer3ladder5","Digi row",200, 0., 200.);
   meRowLayer3Ladder6_ = dbe_->book1D("row_layer3ladder6","Digi row",200, 0., 200.);
   meRowLayer3Ladder7_ = dbe_->book1D("row_layer3ladder7","Digi row",200, 0., 200.);
   meRowLayer3Ladder8_ = dbe_->book1D("row_layer3ladder8","Digi row",200, 0., 200.);

   meColLayer3Ladder1_ = dbe_->book1D("col_layer3ladder1","Digi column",500, 0., 500.);
   meColLayer3Ladder2_ = dbe_->book1D("col_layer3ladder2","Digi column",500, 0., 500.);
   meColLayer3Ladder3_ = dbe_->book1D("col_layer3ladder3","Digi column",500, 0., 500.);
   meColLayer3Ladder4_ = dbe_->book1D("col_layer3ladder4","Digi column",500, 0., 500.);
   meColLayer3Ladder5_ = dbe_->book1D("col_layer3ladder5","Digi column",500, 0., 500.);
   meColLayer3Ladder6_ = dbe_->book1D("col_layer3ladder6","Digi column",500, 0., 500.);
   meColLayer3Ladder7_ = dbe_->book1D("col_layer3ladder7","Digi column",500, 0., 500.);
   meColLayer3Ladder8_ = dbe_->book1D("col_layer3ladder8","Digi column",500, 0., 500.);

   meNdigiPerLadderL3_ = dbe_->book2D("digi_multi_layer3","Digi Num. PerLadder",45,0.,45, 100,0., 10.);

 //Forward Pixel
   /* ZMinus Side 1st Disk */
   meAdcZmDisk1Panel1Plaq1_ = dbe_->book1D("adc_zm_disk1_panel1_plaq1","Digi charge",300,0.,300.);
   meAdcZmDisk1Panel1Plaq2_ = dbe_->book1D("adc_zm_disk1_panel1_plaq2","Digi charge",300,0.,300.);
   meAdcZmDisk1Panel1Plaq2_ = dbe_->book1D("adc_zm_disk1_panel1_plaq3","Digi charge",300,0.,300.);
   meAdcZmDisk1Panel1Plaq3_ = dbe_->book1D("adc_zm_disk1_panel1_plaq4","Digi charge",300,0.,300.);
   meAdcZmDisk1Panel2Plaq1_ = dbe_->book1D("adc_zm_disk1_panel2_plaq1","Digi charge",300,0.,300.);
   meAdcZmDisk1Panel2Plaq2_ = dbe_->book1D("adc_zm_disk1_panel2_plaq2","Digi charge",300,0.,300.);
   meAdcZmDisk1Panel2Plaq3_ = dbe_->book1D("adc_zm_disk1_panel2_plaq3","Digi charge",300,0.,300.);

   meRowZmDisk1Panel1Plaq1_ = dbe_->book1D("row_zm_disk1_panel1_plaq1","Digi row",200,0.,200.);
   meRowZmDisk1Panel1Plaq2_ = dbe_->book1D("row_zm_disk1_panel1_plaq2","Digi row",200,0.,200.);
   meRowZmDisk1Panel1Plaq3_ = dbe_->book1D("row_zm_disk1_panel1_plaq3","Digi row",200,0.,200.);
   meRowZmDisk1Panel1Plaq4_ = dbe_->book1D("row_zm_disk1_panel1_plaq4","Digi row",200,0.,200.);
   meRowZmDisk1Panel2Plaq1_ = dbe_->book1D("row_zm_disk1_panel2_plaq1","Digi row",200,0.,200.);
   meRowZmDisk1Panel2Plaq2_ = dbe_->book1D("row_zm_disk1_panel2_plaq2","Digi row",200,0.,200.);
   meRowZmDisk1Panel2Plaq3_ = dbe_->book1D("row_zm_disk1_panel2_plaq3","Digi row",200,0.,200.);

   meColZmDisk1Panel1Plaq1_ = dbe_->book1D("col_zm_disk1_panel1_plaq1","Digi row",500,0.,500.);
   meColZmDisk1Panel1Plaq2_ = dbe_->book1D("col_zm_disk1_panel1_plaq2","Digi row",500,0.,500.);
   meColZmDisk1Panel1Plaq3_ = dbe_->book1D("col_zm_disk1_panel1_plaq3","Digi row",500,0.,500.);
   meColZmDisk1Panel1Plaq4_ = dbe_->book1D("col_zm_disk1_panel1_plaq4","Digi row",500,0.,500.);
   meColZmDisk1Panel2Plaq1_ = dbe_->book1D("col_zm_disk1_panel2_plaq1","Digi row",500,0.,500.);
   meColZmDisk1Panel2Plaq2_ = dbe_->book1D("col_zm_disk1_panel2_plaq2","Digi row",500,0.,500.);
   meColZmDisk1Panel2Plaq3_ = dbe_->book1D("col_zm_disk1_panel2_plaq3","Digi row",500,0.,500.);
   meNdigiZmDisk1PerPanel1_ = dbe_->book2D("digi_zm_disk1_panel1","Digi Num. Panel1 Of 1st Disk In ZMinus Side ",25,0.,25, 100,0., 10.);
   meNdigiZmDisk1PerPanel2_ = dbe_->book2D("digi_zm_disk1_panel2","Digi Num. Panel2 Of 1st Disk In ZMinus Side ",25,0.,25, 100,0., 10.);

   /* ZMius Side 2nd disk */
   meAdcZmDisk2Panel1Plaq1_ = dbe_->book1D("adc_zm_disk2_panel1_plaq1","Digi charge",300,0.,300.);
   meAdcZmDisk2Panel1Plaq2_ = dbe_->book1D("adc_zm_disk2_panel1_plaq2","Digi charge",300,0.,300.);
   meAdcZmDisk2Panel1Plaq3_ = dbe_->book1D("adc_zm_disk2_panel1_plaq3","Digi charge",300,0.,300.);
   meAdcZmDisk2Panel1Plaq4_ = dbe_->book1D("adc_zm_disk2_panel1_plaq4","Digi charge",300,0.,300.);
   meAdcZmDisk2Panel2Plaq1_ = dbe_->book1D("adc_zm_disk2_panel2_plaq1","Digi charge",300,0.,300.);
   meAdcZmDisk2Panel2Plaq2_ = dbe_->book1D("adc_zm_disk2_panel2_plaq2","Digi charge",300,0.,300.);
   meAdcZmDisk2Panel2Plaq3_ = dbe_->book1D("adc_zm_disk2_panel2_plaq3","Digi charge",300,0.,300.);

   meRowZmDisk2Panel1Plaq1_ = dbe_->book1D("row_zm_disk2_panel1_plaq1","Digi row",200,0.,200.);
   meRowZmDisk2Panel1Plaq2_ = dbe_->book1D("row_zm_disk2_panel1_plaq2","Digi row",200,0.,200.);
   meRowZmDisk2Panel1Plaq3_ = dbe_->book1D("row_zm_disk2_panel1_plaq3","Digi row",200,0.,200.);
   meRowZmDisk2Panel1Plaq4_ = dbe_->book1D("row_zm_disk2_panel1_plaq4","Digi row",200,0.,200.);
   meRowZmDisk2Panel2Plaq1_ = dbe_->book1D("row_zm_disk2_panel2_plaq1","Digi row",200,0.,200.);
   meRowZmDisk2Panel2Plaq2_ = dbe_->book1D("row_zm_disk2_panel2_plaq2","Digi row",200,0.,200.);
   meRowZmDisk2Panel2Plaq3_ = dbe_->book1D("row_zm_disk2_panel2_plaq3","Digi row",200,0.,200.);

   meColZmDisk2Panel1Plaq1_ = dbe_->book1D("col_zm_disk2_panel1_plaq1","Digi row",500,0.,500.);
   meColZmDisk2Panel1Plaq2_ = dbe_->book1D("col_zm_disk2_panel1_plaq2","Digi row",500,0.,500.);
   meColZmDisk2Panel1Plaq3_ = dbe_->book1D("col_zm_disk2_panel1_plaq3","Digi row",500,0.,500.);
   meColZmDisk2Panel1Plaq4_ = dbe_->book1D("col_zm_disk2_panel1_plaq4","Digi row",500,0.,500.);
   meColZmDisk2Panel2Plaq1_ = dbe_->book1D("col_zm_disk2_panel2_plaq1","Digi row",500,0.,500.);
   meColZmDisk2Panel2Plaq2_ = dbe_->book1D("col_zm_disk2_panel2_plaq2","Digi row",500,0.,500.);
   meColZmDisk2Panel2Plaq3_ = dbe_->book1D("col_zm_disk2_panel2_plaq3","Digi row",500,0.,500.);
   meNdigiZmDisk2PerPanel1_ = dbe_->book2D("digi_zm_disk2_panel1","Digi Num. Panel1 Of 2nd Disk In ZMinus Side ",25,0.,25, 100,0., 10.);
   meNdigiZmDisk2PerPanel2_ = dbe_->book2D("digi_zm_disk2_panel2","Digi Num. Panel2 Of 2nd Disk In ZMinus Side ",25,0.,25, 100,0., 10.);


   /* ZPlus Side 1st Disk */
   meAdcZpDisk1Panel1Plaq1_ = dbe_->book1D("adc_zp_disk1_panel1_plaq1","Digi charge",300,0.,300.);
   meAdcZpDisk1Panel1Plaq2_ = dbe_->book1D("adc_zp_disk1_panel1_plaq2","Digi charge",300,0.,300.);
   meAdcZpDisk1Panel1Plaq3_ = dbe_->book1D("adc_zp_disk1_panel1_plaq3","Digi charge",300,0.,300.);
   meAdcZpDisk1Panel1Plaq4_ = dbe_->book1D("adc_zp_disk1_panel1_plaq4","Digi charge",300,0.,300.);
   meAdcZpDisk1Panel2Plaq1_ = dbe_->book1D("adc_zp_disk1_panel2_plaq1","Digi charge",300,0.,300.);
   meAdcZpDisk1Panel2Plaq2_ = dbe_->book1D("adc_zp_disk1_panel2_plaq2","Digi charge",300,0.,300.);
   meAdcZpDisk1Panel2Plaq3_ = dbe_->book1D("adc_zp_disk1_panel2_plaq3","Digi charge",300,0.,300.);

   meRowZpDisk1Panel1Plaq1_ = dbe_->book1D("row_zp_disk1_panel1_plaq1","Digi row",200,0.,200.);
   meRowZpDisk1Panel1Plaq2_ = dbe_->book1D("row_zp_disk1_panel1_plaq2","Digi row",200,0.,200.);
   meRowZpDisk1Panel1Plaq3_ = dbe_->book1D("row_zp_disk1_panel1_plaq3","Digi row",200,0.,200.);
   meRowZpDisk1Panel1Plaq4_ = dbe_->book1D("row_zp_disk1_panel1_plaq4","Digi row",200,0.,200.);
   meRowZpDisk1Panel2Plaq1_ = dbe_->book1D("row_zp_disk1_panel2_plaq1","Digi row",200,0.,200.);
   meRowZpDisk1Panel2Plaq2_ = dbe_->book1D("row_zp_disk1_panel2_plaq2","Digi row",200,0.,200.);
   meRowZpDisk1Panel2Plaq3_ = dbe_->book1D("row_zp_disk1_panel2_plaq3","Digi row",200,0.,200.);

   meColZpDisk1Panel1Plaq1_ = dbe_->book1D("col_zp_disk1_panel1_plaq1","Digi row",500,0.,500.);
   meColZpDisk1Panel1Plaq2_ = dbe_->book1D("col_zp_disk1_panel1_plaq2","Digi row",500,0.,500.);
   meColZpDisk1Panel1Plaq3_ = dbe_->book1D("col_zp_disk1_panel1_plaq3","Digi row",500,0.,500.);
   meColZpDisk1Panel1Plaq4_ = dbe_->book1D("col_zp_disk1_panel1_plaq4","Digi row",500,0.,500.);
   meColZpDisk1Panel2Plaq1_ = dbe_->book1D("col_zp_disk1_panel2_plaq1","Digi row",500,0.,500.);
   meColZpDisk1Panel2Plaq2_ = dbe_->book1D("col_zp_disk1_panel2_plaq2","Digi row",500,0.,500.);
   meColZpDisk1Panel2Plaq3_ = dbe_->book1D("col_zp_disk1_panel2_plaq3","Digi row",500,0.,500.);
   meNdigiZpDisk1PerPanel1_ = dbe_->book2D("digi_zp_disk1_panel1","Digi Num. Panel1 Of 1st Disk In ZPlus Side ",25,0.,25, 100,0., 10.);
   meNdigiZpDisk1PerPanel2_ = dbe_->book2D("digi_zp_disk1_panel2","Digi Num. Panel2 Of 1st Disk In ZPlus Side ",25,0.,25, 100,0., 10.);


   /* ZPlus Side 2nd disk */
   meAdcZpDisk2Panel1Plaq1_ = dbe_->book1D("adc_zp_disk2_panel1_plaq1","Digi charge",300,0.,300.);
   meAdcZpDisk2Panel1Plaq2_ = dbe_->book1D("adc_zp_disk2_panel1_plaq2","Digi charge",300,0.,300.);
   meAdcZpDisk2Panel1Plaq3_ = dbe_->book1D("adc_zp_disk2_panel1_plaq3","Digi charge",300,0.,300.);
   meAdcZpDisk2Panel1Plaq4_ = dbe_->book1D("adc_zp_disk2_panel1_plaq4","Digi charge",300,0.,300.);
   meAdcZpDisk2Panel2Plaq1_ = dbe_->book1D("adc_zp_disk2_panel2_plaq1","Digi charge",300,0.,300.);
   meAdcZpDisk2Panel2Plaq2_ = dbe_->book1D("adc_zp_disk2_panel2_plaq2","Digi charge",300,0.,300.);
   meAdcZpDisk2Panel2Plaq3_ = dbe_->book1D("adc_zp_disk2_panel2_plaq3","Digi charge",300,0.,300.);

   meRowZpDisk2Panel1Plaq1_ = dbe_->book1D("row_zp_disk2_panel1_plaq1","Digi row",200,0.,200.);
   meRowZpDisk2Panel1Plaq2_ = dbe_->book1D("row_zp_disk2_panel1_plaq2","Digi row",200,0.,200.);
   meRowZpDisk2Panel1Plaq3_ = dbe_->book1D("row_zp_disk2_panel1_plaq3","Digi row",200,0.,200.);
   meRowZpDisk2Panel1Plaq4_ = dbe_->book1D("row_zp_disk2_panel1_plaq4","Digi row",200,0.,200.);
   meRowZpDisk2Panel2Plaq1_ = dbe_->book1D("row_zp_disk2_panel2_plaq1","Digi row",200,0.,200.);
   meRowZpDisk2Panel2Plaq2_ = dbe_->book1D("row_zp_disk2_panel2_plaq2","Digi row",200,0.,200.);
   meRowZpDisk2Panel2Plaq3_ = dbe_->book1D("row_zp_disk2_panel2_plaq3","Digi row",200,0.,200.);

   meColZpDisk2Panel1Plaq1_ = dbe_->book1D("col_zp_disk2_panel1_plaq1","Digi row",500,0.,500.);
   meColZpDisk2Panel1Plaq2_ = dbe_->book1D("col_zp_disk2_panel1_plaq2","Digi row",500,0.,500.);
   meColZpDisk2Panel1Plaq3_ = dbe_->book1D("col_zp_disk2_panel1_plaq3","Digi row",500,0.,500.);
   meColZpDisk2Panel1Plaq4_ = dbe_->book1D("col_zp_disk2_panel1_plaq4","Digi row",500,0.,500.);
   meColZpDisk2Panel2Plaq1_ = dbe_->book1D("col_zp_disk2_panel2_plaq1","Digi row",500,0.,500.);
   meColZpDisk2Panel2Plaq2_ = dbe_->book1D("col_zp_disk2_panel2_plaq2","Digi row",500,0.,500.);
   meColZpDisk2Panel2Plaq3_ = dbe_->book1D("col_zp_disk2_panel2_plaq3","Digi row",500,0.,500.);
   meNdigiZpDisk2PerPanel1_ = dbe_->book2D("digi_zp_disk2_panel1","Digi Num. Panel1 Of 2nd Disk In ZPlus Side ",25,0.,25, 100,0., 10.);
   meNdigiZpDisk2PerPanel2_ = dbe_->book2D("digi_zp_disk2_panel2","Digi Num. Panel2 Of 2nd Disk In ZPlus Side ",25,0.,25, 100,0., 10.);


 
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

 edm::ESHandle<TrackerGeometry> tracker;
 c.get<TrackerDigiGeometryRecord>().get( tracker );     

 string digiProducer = "pixdigi";
 Handle<PixelDigiCollection> pixelDigis;
 e.getByLabel(digiProducer, pixelDigis);
 vector<unsigned int>  vec = pixelDigis->detIDs();


 if ( vec.size() > 0 ) 
 LogInfo("SiPixelDigiValid") <<"DetId Size = " <<vec.size();

 int ndigiperladderLayer1[20];
 for(int i = 0; i< 20; i++ ) {
    ndigiperladderLayer1[i] = 0;
 }
 int ndigiperladderLayer2[32];
 for(int i = 0; i< 32; i++ ) {
    ndigiperladderLayer2[i] = 0;
 }
 int ndigiperladderLayer3[44];
 for(int i = 0; i< 44; i++ ) {
    ndigiperladderLayer3[i] = 0;
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
                   
                      ++ndigiperladderLayer1[ladder-1]; 

                     if (zindex == 1) { 
                          meAdcLayer1Ladder1_->Fill((*iter).adc());
                          meRowLayer1Ladder1_->Fill((*iter).row());
                          meColLayer1Ladder1_->Fill((*iter).column());
                     }
                     if (zindex == 2) {
                          meAdcLayer1Ladder2_->Fill((*iter).adc());
                          meRowLayer1Ladder2_->Fill((*iter).row());
                          meColLayer1Ladder2_->Fill((*iter).column());
                     }

                     if (zindex == 3) {
                          meAdcLayer1Ladder3_->Fill((*iter).adc());
                          meRowLayer1Ladder3_->Fill((*iter).row());
                          meColLayer1Ladder3_->Fill((*iter).column());
                     }

                     if (zindex == 4)  {
                         meAdcLayer1Ladder4_->Fill((*iter).adc());
                         meRowLayer1Ladder4_->Fill((*iter).row());
                         meColLayer1Ladder4_->Fill((*iter).column());
                     }

                     if (zindex == 5)  {
                         meAdcLayer1Ladder5_->Fill((*iter).adc());
                         meRowLayer1Ladder5_->Fill((*iter).row());
                         meColLayer1Ladder5_->Fill((*iter).column());
                     }

                     if (zindex == 6)  {
                         meAdcLayer1Ladder6_->Fill((*iter).adc());
                         meRowLayer1Ladder6_->Fill((*iter).row());
                         meColLayer1Ladder6_->Fill((*iter).column());
                     }

                     if (zindex == 7)  {
                         meAdcLayer1Ladder7_->Fill((*iter).adc());
                         meRowLayer1Ladder7_->Fill((*iter).row());
                         meColLayer1Ladder7_->Fill((*iter).column());
                     }
                     if (zindex == 8)  {
                         meAdcLayer1Ladder8_->Fill((*iter).adc());
                         meRowLayer1Ladder8_->Fill((*iter).row());
                         meColLayer1Ladder8_->Fill((*iter).column());
                     }

                } 
                if( layer == 2 ) {

                    ++ndigiperladderLayer2[ladder-1];

                    if (zindex == 1) {
                          meAdcLayer2Ladder1_->Fill((*iter).adc());
                          meRowLayer2Ladder1_->Fill((*iter).row());
                          meColLayer2Ladder1_->Fill((*iter).column());
                     }
                     if (zindex == 2) {
                          meAdcLayer2Ladder2_->Fill((*iter).adc());
                          meRowLayer2Ladder2_->Fill((*iter).row());
                          meColLayer2Ladder2_->Fill((*iter).column());
                     }

                     if (zindex == 3) {
                          meAdcLayer2Ladder3_->Fill((*iter).adc());
                          meRowLayer2Ladder3_->Fill((*iter).row());
                          meColLayer2Ladder3_->Fill((*iter).column());
                     }

                     if (zindex == 4)  {
                         meAdcLayer2Ladder4_->Fill((*iter).adc());
                         meRowLayer2Ladder4_->Fill((*iter).row());
                         meColLayer2Ladder4_->Fill((*iter).column());
                     }

                     if (zindex == 5)  {
                         meAdcLayer2Ladder5_->Fill((*iter).adc());
                         meRowLayer2Ladder5_->Fill((*iter).row());
                         meColLayer2Ladder5_->Fill((*iter).column());
                     }

                     if (zindex == 6)  {
                         meAdcLayer2Ladder6_->Fill((*iter).adc());
                         meRowLayer2Ladder6_->Fill((*iter).row());
                         meColLayer2Ladder6_->Fill((*iter).column());
                     }

                     if (zindex == 7)  {
                         meAdcLayer2Ladder7_->Fill((*iter).adc());
                         meRowLayer2Ladder7_->Fill((*iter).row());
                         meColLayer2Ladder7_->Fill((*iter).column());
                     }
                     if (zindex == 8)  {
                         meAdcLayer2Ladder8_->Fill((*iter).adc());
                         meRowLayer2Ladder8_->Fill((*iter).row());
                         meColLayer2Ladder8_->Fill((*iter).column());
                     }

                }
                if( layer == 3 ) {
      
                    ++ndigiperladderLayer3[ladder-1];

                    if (zindex == 1) {
                          meAdcLayer3Ladder1_->Fill((*iter).adc());
                          meRowLayer3Ladder1_->Fill((*iter).row());
                          meColLayer3Ladder1_->Fill((*iter).column());
                     }
                     if (zindex == 2) {
                          meAdcLayer3Ladder2_->Fill((*iter).adc());
                          meRowLayer3Ladder2_->Fill((*iter).row());
                          meColLayer3Ladder2_->Fill((*iter).column());
                     }

                     if (zindex == 3) {
                          meAdcLayer3Ladder3_->Fill((*iter).adc());
                          meRowLayer3Ladder3_->Fill((*iter).row());
                          meColLayer3Ladder3_->Fill((*iter).column());
                     }

                     if (zindex == 4)  {
                         meAdcLayer3Ladder4_->Fill((*iter).adc());
                         meRowLayer3Ladder4_->Fill((*iter).row());
                         meColLayer3Ladder4_->Fill((*iter).column());
                     }

                     if (zindex == 5)  {
                         meAdcLayer3Ladder5_->Fill((*iter).adc());
                         meRowLayer3Ladder5_->Fill((*iter).row());
                         meColLayer3Ladder5_->Fill((*iter).column());
                     }

                     if (zindex == 6)  {
                         meAdcLayer3Ladder6_->Fill((*iter).adc());
                         meRowLayer3Ladder6_->Fill((*iter).row());
                         meColLayer3Ladder6_->Fill((*iter).column());
                     }

                     if (zindex == 7)  {
                         meAdcLayer3Ladder7_->Fill((*iter).adc());
                         meRowLayer3Ladder7_->Fill((*iter).row());
                         meColLayer3Ladder7_->Fill((*iter).column());
                     }
                     if (zindex == 8)  {
                         meAdcLayer3Ladder8_->Fill((*iter).adc());
                         meRowLayer3Ladder8_->Fill((*iter).row());
                         meColLayer3Ladder8_->Fill((*iter).column());
                     }
                }
 
             }   
           
          }
 
////////////////////////////////////////////////////////////////
//         ForWard Pixel Digi Validation Codes                //
///////////////////////////////////////////////////////////////
        if(detId.subdetId()==PixelSubdetector::PixelEndcap ){ //Endcap
           PXFDetId  fdetid(id);
           unsigned int side  = fdetid.side();
           unsigned int disk  = fdetid.disk();
           unsigned int blade = fdetid.blade();
           unsigned int panel = fdetid.panel();
           unsigned int mod   = fdetid.module();
           LogInfo("SiPixelDigiValid")<<"EndcaP="<<side<<" Disk="<<disk<<" Blade="<<blade<<" Panel="<<panel<<" Module="<<mod;
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

       }//end if id.
    }
    
    for(int i =0; i< 24; i++) {
         meNdigiZmDisk1PerPanel1_->Fill(i,ndigiZmDisk1PerPanel1[i]);
         meNdigiZmDisk1PerPanel2_->Fill(i,ndigiZmDisk1PerPanel2[i]);
         meNdigiZmDisk2PerPanel1_->Fill(i,ndigiZmDisk2PerPanel1[i]);
         meNdigiZmDisk2PerPanel2_->Fill(i,ndigiZmDisk2PerPanel2[i]);
         meNdigiZpDisk1PerPanel1_->Fill(i,ndigiZpDisk1PerPanel1[i]);
         meNdigiZpDisk1PerPanel2_->Fill(i,ndigiZpDisk1PerPanel2[i]);
         meNdigiZpDisk2PerPanel1_->Fill(i,ndigiZpDisk2PerPanel1[i]);
         meNdigiZpDisk2PerPanel2_->Fill(i,ndigiZpDisk2PerPanel2[i]);
    } 

    for(int i = 0 ; i< 20 ; i++) {
       meNdigiPerLadderL1_->Fill(i,ndigiperladderLayer1[i]);
    } 
    for(int i = 0 ; i< 32 ; i++) {
       meNdigiPerLadderL2_->Fill(i,ndigiperladderLayer2[i]);
    }
    for(int i = 0 ; i< 44 ; i++) {
       meNdigiPerLadderL3_->Fill(i,ndigiperladderLayer3[i]);
    }


}
