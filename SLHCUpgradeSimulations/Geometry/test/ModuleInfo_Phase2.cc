// -*- C++ -*-
//
/* 
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

//
//          Original Author: Riccardo Ranieri
//                  Created: Wed May 3 10:30:00 CEST 2006
// Modified for Hybrid & LB: Fri July 31 by E. Brownson
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
//#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/Topology.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

// output
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <bitset>

//
//
// class decleration
//

class ModuleInfo_Phase2 : public edm::EDAnalyzer {
public:
  explicit ModuleInfo_Phase2(const edm::ParameterSet&);
  ~ModuleInfo_Phase2();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  // ----------member data ---------------------------
  bool fromDDD_;
  bool printDDD_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
static const double density_units = 6.24151e+18;

//
// constructors and destructor
//
ModuleInfo_Phase2::ModuleInfo_Phase2(const edm::ParameterSet& ps) {
  fromDDD_ = ps.getParameter<bool>("fromDDD");
  printDDD_ = ps.getUntrackedParameter<bool>("printDDD", true);
  //now do what ever initialization is needed
}

ModuleInfo_Phase2::~ModuleInfo_Phase2() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void ModuleInfo_Phase2::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::LogInfo("ModuleInfo_Phase2") << "begins";

  // output file
  std::ofstream Output("ModuleInfo_Phase2.log", std::ios::out);
  // TEC output as Martin Weber's
  std::ofstream TECOutput("TECLayout_CMSSW.dat", std::ios::out);
  // Numbering Scheme
  std::ofstream NumberingOutput("ModuleNumbering.dat", std::ios::out);
  // Geometry summaries
  std::ofstream GeometryOutput("GeometrySummary.log", std::ios::out);
  std::ofstream GeometryXLS("GeometryXLS.log", std::ios::out);
  //

  //
  // get the GeometricDet
  //
  edm::ESHandle<GeometricDet> rDD;
  edm::ESHandle<std::vector<GeometricDetExtra> > rDDE;
  //if (fromDDD_) {
  iSetup.get<IdealGeometryRecord>().get(rDD);
  iSetup.get<IdealGeometryRecord>().get(rDDE);
  //} else {
  //  iSetup.get<PGeometricDetRcd>().get( rDD );
  //}
  edm::LogInfo("ModuleInfo_Phase2") << " Top node is  " << rDD.product() << " " << rDD.product()->name() << std::endl;
  edm::LogInfo("ModuleInfo_Phase2") << " And Contains  Daughters: " << rDD.product()->deepComponents().size()
                                    << std::endl;
  CmsTrackerDebugNavigator nav(*rDDE.product());
  nav.dump(*rDD.product(), *rDDE.product());
  //
  //first instance tracking geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  //

  // counters
  //unsigned int MAXPXBLAY = 8;
  unsigned int pxbN = 0;
  unsigned int pxb_fullN = 0;
  unsigned int pxb_halfN = 0;
  unsigned int pxb_stackN = 0;
  //unsigned int pxb_full_strx12N = 0;
  //unsigned int pxb_half_strx12N = 0;
  //unsigned int pxb_full_strx34N = 0;
  //unsigned int pxb_half_strx34N = 0;
  unsigned int pxb_full_L[16] = {0};
  unsigned int pxb_half_L[16] = {0};
  unsigned int pxb_stack[16] = {0};
  double psi_pxb_L[16] = {0};
  double psi_pxb[16] = {0};
  double psi_pxb_strx12[16] = {0};
  double psi_pxb_strx34[16] = {0};
  double pxbR_L[16] = {0.0};
  double pxbZ_L[16] = {0.0};
  double pxbpitchx[16] = {0.0};
  double pxbpitchy[16] = {0.0};
  unsigned int pxfN = 0;
  unsigned int pxf_D_N = 0;
  unsigned int pxf_1x2N = 0;
  unsigned int pxf_1x5N = 0;
  unsigned int pxf_2x3N = 0;
  unsigned int pxf_2x4N = 0;
  unsigned int pxf_2x5N = 0;
  unsigned int pxf_D[6] = {0};
  unsigned int pxf_1x2_D[6] = {0};
  unsigned int pxf_1x5_D[6] = {0};
  unsigned int pxf_2x3_D[6] = {0};
  unsigned int pxf_2x4_D[6] = {0};
  unsigned int pxf_2x5_D[6] = {0};
  double pxfpitchx[6] = {0};
  double pxfpitchy[6] = {0};
  double psi_pxf_D[6] = {0};
  double psi_pxf[16] = {0};
  double pxfR_min_D[6] = {9999.0, 9999.0, 9999.0};
  double pxfR_max_D[6] = {0.0};
  double pxfZ_D[6] = {0.0};
  unsigned int tibN = 0;
  unsigned int tib_L12_rphiN = 0;
  unsigned int tib_L12_sterN = 0;
  unsigned int tib_L34_rphiN = 0;
  unsigned int tib_L12_rphi_L[6] = {0};
  unsigned int tib_L12_ster_L[6] = {0};
  unsigned int tib_L34_rphi_L[6] = {0};
  double tib_apv_L[6] = {0};
  double apv_tib = 0;
  double tibR_L[6] = {0.0};
  double tibZ_L[6] = {0.0};
  unsigned int tidN = 0;
  unsigned int tid_r1_rphiN = 0;
  unsigned int tid_r1_sterN = 0;
  unsigned int tid_r2_rphiN = 0;
  unsigned int tid_r2_sterN = 0;
  unsigned int tid_r3_rphiN = 0;
  unsigned int tid_r1_rphi_D[3] = {0};
  unsigned int tid_r1_ster_D[3] = {0};
  unsigned int tid_r2_rphi_D[3] = {0};
  unsigned int tid_r2_ster_D[3] = {0};
  unsigned int tid_r3_rphi_D[3] = {0};
  double tid_apv_D[3] = {0};
  double apv_tid = 0;
  double tidR_min_D[3] = {9999.0, 9999.0, 9999.0};
  double tidR_max_D[3] = {0.0};
  double tidZ_D[3] = {0.0};
  unsigned int tobN = 0;
  unsigned int tob_L12_rphiN = 0;
  unsigned int tob_L12_sterN = 0;
  unsigned int tob_L34_rphiN = 0;
  unsigned int tob_L56_rphiN = 0;
  unsigned int tob_L12_rphi_L[6] = {0};
  unsigned int tob_L12_ster_L[6] = {0};
  unsigned int tob_L34_rphi_L[6] = {0};
  unsigned int tob_L56_rphi_L[6] = {0};
  double tob_apv_L[6] = {0};
  double apv_tob = 0;
  double tobR_L[6] = {0.0};
  double tobZ_L[6] = {0.0};
  unsigned int tecN = 0;
  unsigned int tec_r1_rphiN = 0;
  unsigned int tec_r1_sterN = 0;
  unsigned int tec_r2_rphiN = 0;
  unsigned int tec_r2_sterN = 0;
  unsigned int tec_r3_rphiN = 0;
  unsigned int tec_r4_rphiN = 0;
  unsigned int tec_r5_rphiN = 0;
  unsigned int tec_r5_sterN = 0;
  unsigned int tec_r6_rphiN = 0;
  unsigned int tec_r7_rphiN = 0;
  unsigned int tec_r1_rphi_D[9] = {0};
  unsigned int tec_r1_ster_D[9] = {0};
  unsigned int tec_r2_rphi_D[9] = {0};
  unsigned int tec_r2_ster_D[9] = {0};
  unsigned int tec_r3_rphi_D[9] = {0};
  unsigned int tec_r4_rphi_D[9] = {0};
  unsigned int tec_r5_rphi_D[9] = {0};
  unsigned int tec_r5_ster_D[9] = {0};
  unsigned int tec_r6_rphi_D[9] = {0};
  unsigned int tec_r7_rphi_D[9] = {0};
  double tec_apv_D[9] = {0};
  double apv_tec = 0;
  double tecR_min_D[9] = {9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0};
  double tecR_max_D[9] = {0.0};
  double tecZ_D[9] = {0.0};
  double thepixROCRowsB[16] = {0.0};
  double thepixROCColsB[16] = {0.0};
  double thepixROCRowsD[16] = {0.0};
  double thepixROCColsD[16] = {0.0};
  //
  double volume_total = 0.0;
  double weight_total = 0.0;
  double activeSurface_total = 0.0;
  double volume_pxb = 0.0;
  double weight_pxb = 0.0;
  double activeSurface_pxb = 0.0;
  double activeSurface_pxb_L[16] = {0.0};
  double volume_pxf = 0.0;
  double weight_pxf = 0.0;
  double activeSurface_pxf = 0.0;
  double activeSurface_pxf_D[6] = {0.0};
  double volume_tib = 0.0;
  double weight_tib = 0.0;
  double activeSurface_tib = 0.0;
  double activeSurface_tib_L[4] = {0.0};
  double volume_tid = 0.0;
  double weight_tid = 0.0;
  double activeSurface_tid = 0.0;
  double activeSurface_tid_D[3] = {0.0};
  double volume_tob = 0.0;
  double weight_tob = 0.0;
  double activeSurface_tob = 0.0;
  double activeSurface_tob_L[6] = {0.0};
  double volume_tec = 0.0;
  double weight_tec = 0.0;
  double activeSurface_tec = 0.0;
  double activeSurface_tec_D[9] = {0.0};
  //
  unsigned int nlayersPXB = 0;  //  number of layers
  unsigned int nlayersTIB = 0;  //  number of layers
  unsigned int nlayersTOB = 0;  //  number of layers
  unsigned int ndisksPXF = 0;
  unsigned int ndisksTID = 0;
  unsigned int nwheelsTEC = 0;

  std::vector<const GeometricDet*> modules = (*rDD).deepComponents();
  Output << "************************ List of modules with positions ************************" << std::endl;
  // MEC: 2010-04-13: need to find corresponding GeometricDetExtra.
  std::vector<GeometricDetExtra>::const_iterator gdei(rDDE->begin()), gdeEnd(rDDE->end());
  for (unsigned int i = 0; i < modules.size(); i++) {
    unsigned int rawid = modules[i]->geographicalID().rawId();
    gdei = rDDE->begin();
    for (; gdei != gdeEnd; ++gdei) {
      if (gdei->geographicalId() == modules[i]->geographicalId())
        break;
    }

    if (gdei == gdeEnd)
      throw cms::Exception("ModuleInfo") << "THERE IS NO MATCHING DetId in the GeometricDetExtra";  //THIS never happens!

    GeometricDet::nav_type detNavType = modules[i]->navType();
    Output << std::fixed << std::setprecision(6);  // set as default 6 decimal digits
    std::bitset<32> binary_rawid(rawid);
    Output << " ******** raw Id = " << rawid << " (" << binary_rawid << ") ";

    //    if ( fromDDD_ && printDDD_ ) {
    //      Output << "\t nav type = " << detNavType;
    //    }
    //nav_type typedef changed in 3_6_2; comment out for now.  idr 10/6/10

    Output << std::endl;
    int subdetid = modules[i]->geographicalID().subdetId();
    double volume = gdei->volume() / 1000;  // mm3->cm3
    double density = gdei->density() / density_units;
    double weight = gdei->weight() / density_units / 1000.;        // [kg], hence the factor 1000;
    double thickness = modules[i]->bounds()->thickness() * 10000;  // cm-->um
    double length = (modules[i]->bounds()->length());              // already in cm
    //double width = (modules[i]->bounds()->width()); // already in cm
    double activeSurface = volume / (thickness / 10000);  // cm2 (thickness in um)
    double polarRadius = std::sqrt(modules[i]->translation().X() * modules[i]->translation().X() +
                                   modules[i]->translation().Y() * modules[i]->translation().Y());
    double positionZ = std::abs(modules[i]->translation().Z()) / 10.;  //cm
    volume_total += volume;
    weight_total += weight;
    activeSurface_total += activeSurface;

    switch (subdetid) {
        // PXB
      case 1: {
        pxbN++;
        volume_pxb += volume;
        weight_pxb += weight;
        activeSurface_pxb += activeSurface;
        std::string name = modules[i]->name();
        if (name == "PixelBarrelActiveFull" || name == "PixelBarrelActiveFull0" || name == "PixelBarrelActiveFull1" ||
            name == "PixelBarrelActiveFull2" || name == "PixelBarrelActiveFull3")
          pxb_fullN++;
        if (name == "PixelBarrelActiveHalf" || name == "PixelBarrelActiveHalf1")
          pxb_halfN++;
        if (name == "PixelBarrelActiveStack0" || name == "PixelBarrelActiveStack1" ||
            name == "PixelBarrelActiveStack2" || name == "PixelBarrelActiveStack3" ||
            name == "PixelBarrelActiveStack4" || name == "PixelBarrelActiveStack5" ||
            name == "PixelBarrelActiveStack6" || name == "PixelBarrelActiveStack7" ||
            name == "PixelBarrelActiveStack8" || name == "PixelBarrelActiveStack9")
          pxb_stackN++;
        //if(name == "PixelBarrelActiveFull2") pxb_full_strx12N++; // Outdated ?
        //if(name == "PixelBarrelActiveHalf2") pxb_half_strx12N++;
        //if(name == "PixelBarrelActiveFull3") pxb_full_strx34N++;
        //if(name == "PixelBarrelActiveHalf3") pxb_half_strx34N++;

        unsigned int theLayer = tTopo->pxbLayer(rawid);
        unsigned int theLadder = tTopo->pxbLadder(rawid);
        unsigned int theModule = tTopo->pxbModule(rawid);
        thepixROCRowsB[theLayer - 1] = modules[i]->pixROCRows();
        thepixROCColsB[theLayer - 1] = modules[i]->pixROCCols();
        {
          const DetId& detid = modules[i]->geographicalID();
          DetId detIdObject(detid);
          const GeomDetUnit* genericDet = pDD->idToDetUnit(detIdObject);
          const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
          //std::cout << "  "<<__LINE__<<" PixelGeomDetUnit "<<pixDet->surface().position().perp()<<" , "<<pixDet->surface().position().z()<<"\n";
          const PixelTopology* theTopol = &(pixDet->specificTopology());
          std::pair<float, float> pitchxy = theTopol->pitch();
          pxbpitchx[theLayer - 1] = double(int(0.5 + (10000 * pitchxy.first)));
          pxbpitchy[theLayer - 1] = double(int(0.5 + (10000 * pitchxy.second)));
          //std::cout<<"  "<<" BPix Layer "<< theLayer << " with Pitch = " << pxbpitchx[theLayer-1]<<" , "<<pxbpitchy[theLayer-1]<<"\n";
        }  // Discard some transitional variables.
        if (theLayer > nlayersPXB)
          nlayersPXB = theLayer;
        // The following sums will need to be verified...
        if (name == "PixelBarrelActiveFull" || name == "PixelBarrelActiveFull0" || name == "PixelBarrelActiveFull1" ||
            name == "PixelBarrelActiveFull2" || name == "PixelBarrelActiveFull3")
          pxb_full_L[theLayer - 1]++;
        if (name == "PixelBarrelActiveHalf" || name == "PixelBarrelActiveHalf1" || name == "PixelBarrelActiveHalf2" ||
            name == "PixelBarrelActiveHalf3")
          pxb_half_L[theLayer - 1]++;
        if (name == "PixelBarrelActiveStack0" || name == "PixelBarrelActiveStack1" ||
            name == "PixelBarrelActiveStack2" || name == "PixelBarrelActiveStack3" ||
            name == "PixelBarrelActiveStack4" || name == "PixelBarrelActiveStack5" ||
            name == "PixelBarrelActiveStack6" || name == "PixelBarrelActiveStack7" ||
            name == "PixelBarrelActiveStack8" || name == "PixelBarrelActiveStack9")
          pxb_stack[theLayer - 1]++;
        if (name == "PixelBarrelActiveFull" || name == "PixelBarrelActiveHalf" || name == "PixelBarrelActiveFull0" ||
            name == "PixelBarrelActiveFull1" || name == "PixelBarrelActiveHalf1" || name == "PixelBarrelActiveStack0" ||
            name == "PixelBarrelActiveStack1" || name == "PixelBarrelActiveStack2" ||
            name == "PixelBarrelActiveStack3" || name == "PixelBarrelActiveStack4" ||
            name == "PixelBarrelActiveStack5" || name == "PixelBarrelActiveStack6" ||
            name == "PixelBarrelActiveStack7" || name == "PixelBarrelActiveStack8" || name == "PixelBarrelActiveStack9")
          psi_pxb[theLayer - 1] += modules[i]->pixROCx() * modules[i]->pixROCy();

        if (name == "PixelBarrelActiveFull2" || name == "PixelBarrelActiveHalf2")
          psi_pxb_strx12[theLayer - 1] += modules[i]->pixROCx() * modules[i]->pixROCy();
        if (name == "PixelBarrelActiveFull3" || name == "PixelBarrelActiveHalf3")
          psi_pxb_strx34[theLayer - 1] += modules[i]->pixROCx() * modules[i]->pixROCy();

        // Make sure there are no new names we didn't know about.
        if ((name == "PixelBarrelActiveStack0" || name == "PixelBarrelActiveStack1" ||
             name == "PixelBarrelActiveStack2" || name == "PixelBarrelActiveStack3" ||
             name == "PixelBarrelActiveStack4" || name == "PixelBarrelActiveStack5" ||
             name == "PixelBarrelActiveStack6" || name == "PixelBarrelActiveStack7" ||
             name == "PixelBarrelActiveStack8" || name == "PixelBarrelActiveStack9" ||
             name == "PixelBarrelActiveFull" || name == "PixelBarrelActiveFull1" || name == "PixelBarrelActiveHalf" ||
             name == "PixelBarrelActiveHalf1" || name == "PixelBarrelActiveFull2" || name == "PixelBarrelActiveHalf2" ||
             name == "PixelBarrelActiveFull3" || name == "PixelBarrelActiveHalf3" ||
             name == "PixelBarrelActiveFull0") == 0)
          std::cout << "\nYou have added PXB layers that are not taken into account! \ti.e. " << name << "\n";
        if (16 < theLayer)
          std::cout << "\nYou need to increase the PXB array sizes!\n";
        activeSurface_pxb_L[theLayer - 1] += activeSurface;
        psi_pxb_L[theLayer - 1] += modules[i]->pixROCx() * modules[i]->pixROCy();

        if (pxbZ_L[theLayer - 1] < positionZ + length / 2)
          pxbZ_L[theLayer - 1] = positionZ + length / 2;
        pxbR_L[theLayer - 1] += polarRadius / 10;  // cm
        Output << " PXB"
               << "\t"
               << "Layer " << theLayer << " Ladder " << theLadder << "\t"
               << " module " << theModule << " " << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name() << std::endl;
        } else {
          Output << " NO DDD Hierarchy available " << std::endl;
        }
        break;
      }

        // PXF
      case 2: {
        pxfN++;
        volume_pxf += volume;
        weight_pxf += weight;
        activeSurface_pxf += activeSurface;
        std::string name = modules[i]->name();
        if (name == "PixelForwardSensor" || name == "PixelForwardSensor1" || name == "PixelForwardSensor2" ||
            name == "PixelForwardSensor3")
          pxf_D_N++;
        if (name == "PixelForwardActive1x2")
          pxf_1x2N++;
        if (name == "PixelForwardActive1x5")
          pxf_1x5N++;
        if (name == "PixelForwardActive2x3")
          pxf_2x3N++;
        if (name == "PixelForwardActive2x4")
          pxf_2x4N++;
        if (name == "PixelForwardActive2x5")
          pxf_2x5N++;

        unsigned int thePanel = tTopo->pxfPanel(rawid);
        unsigned int theDisk = tTopo->pxfDisk(rawid);
        unsigned int theBlade = tTopo->pxfBlade(rawid);
        unsigned int theModule = tTopo->pxfModule(rawid);
        thepixROCRowsD[theDisk - 1] = modules[i]->pixROCRows();
        thepixROCColsD[theDisk - 1] = modules[i]->pixROCCols();
        {
          const DetId& detid = modules[i]->geographicalID();
          DetId detIdObject(detid);
          const GeomDetUnit* genericDet = pDD->idToDetUnit(detIdObject);
          const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
          const PixelTopology* theTopol = &(pixDet->specificTopology());
          std::pair<float, float> pitchxy = theTopol->pitch();
          pxfpitchx[theDisk - 1] = double(int(0.5 + (10000 * pitchxy.first)));
          pxfpitchy[theDisk - 1] = double(int(0.5 + (10000 * pitchxy.second)));
        }  // Discard some transitional variables.
        if (theDisk > ndisksPXF)
          ndisksPXF = theDisk;
        if (name == "PixelForwardSensor" || name == "PixelForwardSensor1" || name == "PixelForwardSensor2" ||
            name == "PixelForwardSensor3")
          pxf_D[theDisk - 1]++;
        if (name == "PixelForwardActive1x2")
          pxf_1x2_D[theDisk - 1]++;
        if (name == "PixelForwardActive1x5")
          pxf_1x5_D[theDisk - 1]++;
        if (name == "PixelForwardActive2x3")
          pxf_2x3_D[theDisk - 1]++;
        if (name == "PixelForwardActive2x4")
          pxf_2x4_D[theDisk - 1]++;
        if (name == "PixelForwardActive2x5")
          pxf_2x5_D[theDisk - 1]++;
        // Make sure there are no new names we didn't know about.
        if ((name == "PixelForwardSensor" || name == "PixelForwardActive1x2" || name == "PixelForwardActive1x5" ||
             name == "PixelForwardActive2x3" || name == "PixelForwardActive2x4" || name == "PixelForwardActive2x5" ||
             name == "PixelForwardSensor1" || name == "PixelForwardSensor2" || name == "PixelForwardSensor3") == 0)
          std::cout << "\nYou have added PXF layers that are not taken into account! \ti.e. " << name << "\n";
        if (3 < theDisk)
          std::cout << "\nYou need to increase the PXF array sizes!\n";
        activeSurface_pxf_D[theDisk - 1] += activeSurface;
        psi_pxf_D[theDisk - 1] += modules[i]->pixROCx() * modules[i]->pixROCy();
        psi_pxf[theDisk - 1] += modules[i]->pixROCx() * modules[i]->pixROCy();
        pxfZ_D[theDisk - 1] += positionZ;
        polarRadius = polarRadius / 10.;
        if (pxfR_min_D[theDisk - 1] > polarRadius - length / 2)
          pxfR_min_D[theDisk - 1] = polarRadius - length / 2;
        if (pxfR_max_D[theDisk - 1] < polarRadius + length / 2)
          pxfR_max_D[theDisk - 1] = polarRadius + length / 2;
        std::string side;
        side = (tTopo->pxfSide(rawid) == 1) ? "-" : "+";
        Output << " PXF" << side << "\t"
               << "Disk " << theDisk << " Blade " << theBlade << " Panel " << thePanel << "\t"
               << " module " << theModule << "\t" << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name() << std::endl;
        } else {
          Output << " NO DDD Hierarchy available " << std::endl;
        }
        break;
      }

        // TIB
      case 3: {
        tibN++;
        volume_tib += volume;
        weight_tib += weight;
        activeSurface_tib += activeSurface;
        std::string name = modules[i]->name();
        if (name == "TIBActiveRphi0")
          tib_L12_rphiN++;
        if (name == "TIBActiveSter0")
          tib_L12_sterN++;
        if (name == "TIBActiveRphi2")
          tib_L34_rphiN++;

        unsigned int theLayer = tTopo->tibLayer(rawid);
        std::vector<unsigned int> theString = tTopo->tibStringInfo(rawid);
        unsigned int theModule = tTopo->tibModule(rawid);
        if (theLayer > nlayersTIB)
          nlayersTIB = theLayer;
        if (name == "TIBActiveRphi0")
          tib_L12_rphi_L[theLayer - 1]++;
        if (name == "TIBActiveSter0")
          tib_L12_ster_L[theLayer - 1]++;
        if (name == "TIBActiveRphi2")
          tib_L34_rphi_L[theLayer - 1]++;
        if ((name == "TIBActiveRphi0" || name == "TIBActiveSter0" || name == "TIBActiveRphi2") == 0)
          std::cout << "\nYou have added TIB layers that are not taken into account!\n\n";
        if (6 < theLayer)
          std::cout << "\nYou need to increase the TIB array sizes!\n";
        activeSurface_tib_L[theLayer - 1] += activeSurface;
        tib_apv_L[theLayer - 1] += modules[i]->siliconAPVNum();
        apv_tib += modules[i]->siliconAPVNum();
        if (tibZ_L[theLayer - 1] < positionZ + length / 2)
          tibZ_L[theLayer - 1] = positionZ + length / 2;
        tibR_L[theLayer - 1] += polarRadius / 10;  // cm
        std::string side;
        std::string part;
        side = (theString[0] == 1) ? "-" : "+";
        part = (theString[1] == 1) ? "int" : "ext";

        Output << " TIB" << side << "\t"
               << "Layer " << theLayer << " " << part << "\t"
               << "string " << theString[2] << "\t"
               << " module " << theModule << " " << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name();
        } else {
          Output << " NO DDD Hierarchy available ";
        }
        Output << " " << modules[i]->translation().X() << "   \t" << modules[i]->translation().Y() << "   \t"
               << modules[i]->translation().Z() << std::endl;
        break;
      }

        // TID
      case 4: {
        tidN++;
        volume_tid += volume;
        weight_tid += weight;
        activeSurface_tid += activeSurface;
        std::string name = modules[i]->name();
        if (name == "TIDModule0RphiActive")
          tid_r1_rphiN++;
        if (name == "TIDModule0StereoActive")
          tid_r1_sterN++;
        if (name == "TIDModule1RphiActive")
          tid_r2_rphiN++;
        if (name == "TIDModule1StereoActive")
          tid_r2_sterN++;
        if (name == "TIDModule2RphiActive")
          tid_r3_rphiN++;

        unsigned int theDisk = tTopo->tidWheel(rawid);
        unsigned int theRing = tTopo->tidRing(rawid);
        std::vector<unsigned int> theModule = tTopo->tidModuleInfo(rawid);
        if (theDisk > ndisksTID)
          ndisksTID = theDisk;
        if (name == "TIDModule0RphiActive")
          tid_r1_rphi_D[theDisk - 1]++;
        if (name == "TIDModule0StereoActive")
          tid_r1_ster_D[theDisk - 1]++;
        if (name == "TIDModule1RphiActive")
          tid_r2_rphi_D[theDisk - 1]++;
        if (name == "TIDModule1StereoActive")
          tid_r2_ster_D[theDisk - 1]++;
        if (name == "TIDModule2RphiActive")
          tid_r3_rphi_D[theDisk - 1]++;
        if ((name == "TIDModule0RphiActive" || name == "TIDModule0StereoActive" || name == "TIDModule1RphiActive" ||
             name == "TIDModule1StereoActive" || name == "TIDModule2RphiActive") == 0)
          std::cout << "\nYou have added TID layers that are not taken into account!\n\n";
        if (3 < theDisk)
          std::cout << "\nYou need to increase the TID array sizes!\n";
        activeSurface_tid_D[theDisk - 1] += activeSurface;
        tid_apv_D[theDisk - 1] += modules[i]->siliconAPVNum();
        apv_tid += modules[i]->siliconAPVNum();
        tidZ_D[theDisk - 1] += positionZ;
        polarRadius = polarRadius / 10.;
        if (tidR_min_D[theDisk - 1] > polarRadius - length / 2)
          tidR_min_D[theDisk - 1] = polarRadius - length / 2;
        if (tidR_max_D[theDisk - 1] < polarRadius + length / 2)
          tidR_max_D[theDisk - 1] = polarRadius + length / 2;
        std::string side;
        std::string part;
        side = (tTopo->tidSide(rawid) == 1) ? "-" : "+";
        part = (theModule[0] == 1) ? "back" : "front";
        Output << " TID" << side << "\t"
               << "Disk " << theDisk << " Ring " << theRing << " " << part << "\t"
               << " module " << theModule[1] << "\t" << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name();
        } else {
          Output << " NO DDD Hierarchy available ";
        }
        Output << " " << modules[i]->translation().X() << "   \t" << modules[i]->translation().Y() << "   \t"
               << modules[i]->translation().Z() << std::endl;
        break;
      }

        // TOB
      case 5: {
        tobN++;
        volume_tob += volume;
        weight_tob += weight;
        activeSurface_tob += activeSurface;
        std::string name = modules[i]->name();
        if (name == "TOBActiveRphi0")
          tob_L12_rphiN++;
        if (name == "TOBActiveSter0")
          tob_L12_sterN++;
        if (name == "TOBActiveRphi2")
          tob_L34_rphiN++;
        if (name == "TOBActiveRphi4")
          tob_L56_rphiN++;

        unsigned int theLayer = tTopo->tobLayer(rawid);
        std::vector<unsigned int> theRod = tTopo->tobRodInfo(rawid);
        unsigned int theModule = tTopo->tobModule(rawid);
        if (theLayer > nlayersTOB)
          nlayersTOB = theLayer;
        if (name == "TOBActiveRphi0")
          tob_L12_rphi_L[theLayer - 1]++;
        if (name == "TOBActiveSter0")
          tob_L12_ster_L[theLayer - 1]++;
        if (name == "TOBActiveRphi2")
          tob_L34_rphi_L[theLayer - 1]++;
        if (name == "TOBActiveRphi4")
          tob_L56_rphi_L[theLayer - 1]++;
        if ((name == "TOBActiveRphi0" || name == "TOBActiveSter0" || name == "TOBActiveRphi2" ||
             name == "TOBActiveRphi4") == 0)
          std::cout << "\nYou have added TOB layers that are not taken into account!\n\n";
        if (6 < theLayer)
          std::cout << "\nYou need to increase the TOB array sizes!\n";
        activeSurface_tob_L[theLayer - 1] += activeSurface;
        tob_apv_L[theLayer - 1] += modules[i]->siliconAPVNum();
        apv_tob += modules[i]->siliconAPVNum();
        if (tobZ_L[theLayer - 1] < positionZ + length / 2)
          tobZ_L[theLayer - 1] = positionZ + length / 2;
        tobR_L[theLayer - 1] += polarRadius / 10;  // cm
        std::string side;
        std::string part;
        side = (theRod[0] == 1) ? "-" : "+";
        Output << " TOB" << side << "\t"
               << "Layer " << theLayer << "\t"
               << "rod " << theRod[1] << " module " << theModule << "\t" << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name();
        } else {
          Output << " NO DDD Hierarchy available ";
        }
        Output << " " << modules[i]->translation().X() << "   \t" << modules[i]->translation().Y() << "   \t"
               << modules[i]->translation().Z() << std::endl;
        break;
      }

        // TEC
      case 6: {
        tecN++;
        volume_tec += volume;
        weight_tec += weight;
        activeSurface_tec += activeSurface;
        std::string name = modules[i]->name();
        if (name == "TECModule0RphiActive")
          tec_r1_rphiN++;
        if (name == "TECModule0StereoActive")
          tec_r1_sterN++;
        if (name == "TECModule1RphiActive")
          tec_r2_rphiN++;
        if (name == "TECModule1StereoActive")
          tec_r2_sterN++;
        if (name == "TECModule2RphiActive")
          tec_r3_rphiN++;
        if (name == "TECModule3RphiActive")
          tec_r4_rphiN++;
        if (name == "TECModule4RphiActive")
          tec_r5_rphiN++;
        if (name == "TECModule4StereoActive")
          tec_r5_sterN++;
        if (name == "TECModule5RphiActive")
          tec_r6_rphiN++;
        if (name == "TECModule6RphiActive")
          tec_r7_rphiN++;

        unsigned int theWheel = tTopo->tecWheel(rawid);
        unsigned int theModule = tTopo->tecModule(rawid);
        std::vector<unsigned int> thePetal = tTopo->tecPetalInfo(rawid);
        unsigned int theRing = tTopo->tecRing(rawid);
        if (theWheel > nwheelsTEC)
          nwheelsTEC = theWheel;
        if (name == "TECModule0RphiActive")
          tec_r1_rphi_D[theWheel - 1]++;
        if (name == "TECModule0StereoActive")
          tec_r1_ster_D[theWheel - 1]++;
        if (name == "TECModule1RphiActive")
          tec_r2_rphi_D[theWheel - 1]++;
        if (name == "TECModule1StereoActive")
          tec_r2_ster_D[theWheel - 1]++;
        if (name == "TECModule2RphiActive")
          tec_r3_rphi_D[theWheel - 1]++;
        if (name == "TECModule3RphiActive")
          tec_r4_rphi_D[theWheel - 1]++;
        if (name == "TECModule4RphiActive")
          tec_r5_rphi_D[theWheel - 1]++;
        if (name == "TECModule4StereoActive")
          tec_r5_ster_D[theWheel - 1]++;
        if (name == "TECModule5RphiActive")
          tec_r6_rphi_D[theWheel - 1]++;
        if (name == "TECModule6RphiActive")
          tec_r7_rphi_D[theWheel - 1]++;
        if ((name == "TECModule0RphiActive" || name == "TECModule0StereoActive" || name == "TECModule1RphiActive" ||
             name == "TECModule1StereoActive" || name == "TECModule2RphiActive" || name == "TECModule3RphiActive" ||
             name == "TECModule4RphiActive" || name == "TECModule4StereoActive" || name == "TECModule5RphiActive" ||
             name == "TECModule6RphiActive") == 0)
          std::cout << "\nYou have added TOB layers that are not taken into account!,\t" << name << "\n";
        if (9 < theWheel)
          std::cout << "\nYou need to increase the TEC array sizes!\n";
        activeSurface_tec_D[theWheel - 1] += activeSurface;
        tec_apv_D[theWheel - 1] += modules[i]->siliconAPVNum();
        apv_tec += modules[i]->siliconAPVNum();
        tecZ_D[theWheel - 1] += positionZ;
        polarRadius = polarRadius / 10.;
        if (tecR_min_D[theWheel - 1] > polarRadius - length / 2)
          tecR_min_D[theWheel - 1] = polarRadius - length / 2;
        if (tecR_max_D[theWheel - 1] < polarRadius + length / 2)
          tecR_max_D[theWheel - 1] = polarRadius + length / 2;
        std::string side;
        std::string petal;
        side = (tTopo->tecSide(rawid) == 1) ? "-" : "+";
        petal = (thePetal[0] == 1) ? "back" : "front";
        Output << " TEC" << side << "\t"
               << "Wheel " << theWheel << " Petal " << thePetal[1] << " " << petal << " Ring " << theRing << "\t"
               << "\t"
               << " module " << theModule << "\t" << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name();
        } else {
          Output << " NO DDD Hierarchy available ";
        }
        Output << " " << modules[i]->translation().X() << "   \t" << modules[i]->translation().Y() << "   \t"
               << modules[i]->translation().Z() << std::endl;

        // TEC output as Martin Weber's
        int out_side = (tTopo->tecSide(rawid) == 1) ? -1 : 1;
        unsigned int out_disk = tTopo->tecWheel(rawid);
        unsigned int out_sector = thePetal[1];
        int out_petal = (thePetal[0] == 1) ? 1 : -1;
        // swap sector numbers for TEC-
        if (out_side == -1) {
          // fine for back petals, substract 1 for front petals
          if (out_petal == -1) {
            out_sector = (out_sector + 6) % 8 + 1;
          }
        }
        unsigned int out_ring = tTopo->tecRing(rawid);
        int out_sensor = 0;
        if (name == "TECModule0RphiActive")
          out_sensor = -1;
        if (name == "TECModule0StereoActive")
          out_sensor = 1;
        if (name == "TECModule1RphiActive")
          out_sensor = -1;
        if (name == "TECModule1StereoActive")
          out_sensor = 1;
        if (name == "TECModule2RphiActive")
          out_sensor = -1;
        if (name == "TECModule3RphiActive")
          out_sensor = -1;
        if (name == "TECModule4RphiActive")
          out_sensor = -1;
        if (name == "TECModule4StereoActive")
          out_sensor = 1;
        if (name == "TECModule5RphiActive")
          out_sensor = -1;
        if (name == "TECModule6RphiActive")
          out_sensor = -1;
        unsigned int out_module;
        if (out_ring == 1 || out_ring == 2 || out_ring == 5) {
          // rings with stereo modules
          // create number odd by default
          out_module = 2 * (tTopo->tecModule(rawid) - 1) + 1;
          if (out_sensor == 1) {
            // in even rings, stereo modules are the even ones
            if (out_ring == 2)
              out_module += 1;
          } else
              // in odd rings, stereo modules are the odd ones
              if (out_ring != 2)
            out_module += 1;
        } else {
          out_module = tTopo->tecModule(rawid);
        }
        double out_x = modules[i]->translation().X();
        double out_y = modules[i]->translation().Y();
        double out_z = modules[i]->translation().Z();
        double out_r = sqrt(modules[i]->translation().X() * modules[i]->translation().X() +
                            modules[i]->translation().Y() * modules[i]->translation().Y());
        double out_phi_rad = atan2(modules[i]->translation().Y(), modules[i]->translation().X());
        TECOutput << out_side << " " << out_disk << " " << out_sector << " " << out_petal << " " << out_ring << " "
                  << out_module << " " << out_sensor << " " << out_x << " " << out_y << " " << out_z << " " << out_r
                  << " " << out_phi_rad << std::endl;
        //
        break;
      }
      default:
        Output << " WARNING no Silicon Strip detector, I got a " << rawid << std::endl;
        ;
    }

    // Local axes from Reco
    const GeomDet* geomdet = pDD->idToDet(modules[i]->geographicalID());
    // Global Coordinates (i,j,k)
    LocalVector xLocal(1, 0, 0);
    LocalVector yLocal(0, 1, 0);
    LocalVector zLocal(0, 0, 1);
    // Versor components
    GlobalVector xGlobal = (geomdet->surface()).toGlobal(xLocal);
    GlobalVector yGlobal = (geomdet->surface()).toGlobal(yLocal);
    GlobalVector zGlobal = (geomdet->surface()).toGlobal(zLocal);
    //

    // Output: set as default 4 decimal digits (0.1 um or 0.1 deg/rad)
    // active area center
    Output << "\t"
           << "volume " << std::fixed << std::setprecision(3) << volume << " cm3 \t"
           << "density " << std::fixed << std::setprecision(3) << density << " g/cm3 \t"
           << "weight " << std::fixed << std::setprecision(6) << weight << " kg \t"
           << "thickness " << std::fixed << std::setprecision(0) << thickness << " um \t"
           << " active area " << std::fixed << std::setprecision(2) << activeSurface << " cm2" << std::endl;
    Output << "\tActive Area Center" << std::endl;
    Output << "\t O = (" << std::fixed << std::setprecision(4) << modules[i]->translation().X() << "," << std::fixed
           << std::setprecision(4) << modules[i]->translation().Y() << "," << std::fixed << std::setprecision(4)
           << modules[i]->translation().Z() << ")" << std::endl;
    //
    //double polarRadius = std::sqrt(modules[i]->translation().X()*modules[i]->translation().X()+modules[i]->translation().Y()*modules[i]->translation().Y());
    double phiDeg = atan2(modules[i]->translation().Y(), modules[i]->translation().X()) * 360. / 6.283185307;
    double phiRad = atan2(modules[i]->translation().Y(), modules[i]->translation().X());
    //
    Output << "\t\t polar radius " << std::fixed << std::setprecision(4) << polarRadius << "\t"
           << "phi [deg] " << std::fixed << std::setprecision(4) << phiDeg << "\t"
           << "phi [rad] " << std::fixed << std::setprecision(4) << phiRad << std::endl;
    // active area versors (rotation matrix)
    DD3Vector x, y, z;
    modules[i]->rotation().GetComponents(x, y, z);
    Output << "\tActive Area Rotation Matrix" << std::endl;
    Output << "\t z = n = (" << std::fixed << std::setprecision(4) << z.X() << "," << std::fixed << std::setprecision(4)
           << z.Y() << "," << std::fixed << std::setprecision(4) << z.Z() << ")" << std::endl
           << "\t [Rec] = (" << std::fixed << std::setprecision(4) << zGlobal.x() << "," << std::fixed
           << std::setprecision(4) << zGlobal.y() << "," << std::fixed << std::setprecision(4) << zGlobal.z() << ")"
           << std::endl
           << "\t x = t = (" << std::fixed << std::setprecision(4) << x.X() << "," << std::fixed << std::setprecision(4)
           << x.Y() << "," << std::fixed << std::setprecision(4) << x.Z() << ")" << std::endl
           << "\t [Rec] = (" << std::fixed << std::setprecision(4) << xGlobal.x() << "," << std::fixed
           << std::setprecision(4) << xGlobal.y() << "," << std::fixed << std::setprecision(4) << xGlobal.z() << ")"
           << std::endl
           << "\t y = k = (" << std::fixed << std::setprecision(4) << y.X() << "," << std::fixed << std::setprecision(4)
           << y.Y() << "," << std::fixed << std::setprecision(4) << y.Z() << ")" << std::endl
           << "\t [Rec] = (" << std::fixed << std::setprecision(4) << yGlobal.x() << "," << std::fixed
           << std::setprecision(4) << yGlobal.y() << "," << std::fixed << std::setprecision(4) << yGlobal.z() << ")"
           << std::endl;

    // NumberingScheme
    NumberingOutput << rawid;

    //    if ( fromDDD_ && printDDD_ ) {
    //      NumberingOutput << " " << detNavType;
    //    }
    //nav_type typedef changed in 3_6_2; comment out for now.  idr 10/6/10

    NumberingOutput << " " << std::fixed << std::setprecision(4) << modules[i]->translation().X() << " " << std::fixed
                    << std::setprecision(4) << modules[i]->translation().Y() << " " << std::fixed
                    << std::setprecision(4) << modules[i]->translation().Z() << " " << std::endl;
    //
  }

  // params
  // Pixel
  unsigned int chan_per_psiB[16] = {0}, chan_per_psiD[16] = {0};
  double chan_pxb = 0.0;
  double chan_strx12 = 0.0;
  double chan_strx34 = 0.0;
  double chan_pxf = 0.0;
  unsigned int psi_pxbN = 0, psi_pxb_strx12N = 0, psi_pxb_strx34N = 0, psi_pxfN = 0;
  for (int i = 0; i < 16; i++) {
    chan_per_psiB[i] = (unsigned int)(thepixROCRowsB[i] * thepixROCColsB[i]);
    chan_per_psiD[i] = (unsigned int)(thepixROCRowsD[i] * thepixROCColsD[i]);
    chan_pxb += psi_pxb[i] * chan_per_psiB[i];
    chan_strx12 += psi_pxb_strx12[i] * chan_per_psiB[i];
    chan_strx34 += psi_pxb_strx34[i] * chan_per_psiB[i];
    chan_pxf += psi_pxf[i] * chan_per_psiD[i];
    psi_pxbN += (unsigned int)psi_pxb[i];
    psi_pxb_strx12N += (unsigned int)psi_pxb_strx12[i];
    psi_pxb_strx34N += (unsigned int)psi_pxb_strx34[i];
    psi_pxfN += (unsigned int)psi_pxf[i];
  }

  // Strip
  unsigned int chan_per_apv = 128;
  double chan_tib = apv_tib * chan_per_apv;
  double chan_tid = apv_tid * chan_per_apv;
  double chan_tob = apv_tob * chan_per_apv;
  double chan_tec = apv_tec * chan_per_apv;
  double psi_tot = psi_pxbN + psi_pxb_strx12N + psi_pxb_strx34N + psi_pxfN;
  double apv_tot = apv_tib + apv_tid + apv_tob + apv_tec;
  double chan_pixel = chan_pxb + chan_strx12 + chan_strx34 + chan_pxf;
  double chan_strip = chan_tib + chan_tid + chan_tob + chan_tec;
  double chan_tot = chan_pixel + chan_strip;
  //

  // summary
  Output << "---------------------" << std::endl;
  Output << " Counters " << std::endl;
  Output << "---------------------" << std::endl;
  Output << " Total number of PXB layers   = " << nlayersPXB << std::endl;
  Output << " PXB Total   = " << pxbN << std::endl;
  Output << "   Inner: Full = " << pxb_fullN << std::endl;
  Output << "   Inner: Half = " << pxb_halfN << std::endl;
  Output << "        Stacks = " << pxb_stackN << std::endl;
  //Output << "   Strx12: Full = " << pxb_full_strx12N << std::endl;
  //Output << "   Strx12: Half = " << pxb_half_strx12N << std::endl;
  //Output << "   Strx34: Full = " << pxb_full_strx34N << std::endl;
  //Output << "   Strx34: Half = " << pxb_half_strx34N << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_pxb << " kg" << std::endl;
  Output << "     Volume  = " << volume_pxb << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_pxb << " cm2" << std::endl;
  Output << "        NEED TO VERIFY THE NEXT 6 LINES!!!!!!!!!!!!!!!!! " << std::endl;
  Output << "        PSI46s Inner  = " << (int)psi_pxbN << std::endl;
  Output << "        PSI46s Strx12  = " << (int)psi_pxb_strx12N << std::endl;
  Output << "        PSI46s Strx34  = " << (int)psi_pxb_strx34N << std::endl;
  Output << "        channels Inner = " << (int)chan_pxb << std::endl;
  Output << "        channels Strx12 = " << (int)chan_strx12 << std::endl;
  Output << "        channels Strx34 = " << (int)chan_strx34 << std::endl;
  Output << " PXF    = " << pxfN << std::endl;
  Output << "   PH1 = " << pxf_D_N << std::endl;
  Output << "   1x2 = " << pxf_1x2N << std::endl;
  Output << "   1x5 = " << pxf_1x5N << std::endl;
  Output << "   2x3 = " << pxf_2x3N << std::endl;
  Output << "   2x4 = " << pxf_2x4N << std::endl;
  Output << "   2x5 = " << pxf_2x5N << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_pxf << " kg" << std::endl;
  Output << "     Volume  = " << volume_pxf << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_pxf << " cm2" << std::endl;
  Output << "        PSI46s   = " << (int)psi_pxfN << std::endl;
  Output << "        channels = " << (int)chan_pxf << std::endl;
  Output << " TIB    = " << tibN << std::endl;
  Output << "   L12 rphi   = " << tib_L12_rphiN << std::endl;
  Output << "   L12 stereo = " << tib_L12_sterN << std::endl;
  Output << "   L34        = " << tib_L34_rphiN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_tib << " kg" << std::endl;
  Output << "     Volume  = " << volume_tib << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_tib << " cm2" << std::endl;
  Output << "        APV25s   = " << (int)apv_tib << std::endl;
  Output << "        channels = " << (int)chan_tib << std::endl;
  Output << " TID    = " << tidN << std::endl;
  Output << "   r1 rphi    = " << tid_r1_rphiN << std::endl;
  Output << "   r1 stereo  = " << tid_r1_sterN << std::endl;
  Output << "   r2 rphi    = " << tid_r2_rphiN << std::endl;
  Output << "   r2 stereo  = " << tid_r2_sterN << std::endl;
  Output << "   r3 rphi    = " << tid_r3_rphiN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_tid << " kg" << std::endl;
  Output << "     Volume  = " << volume_tid << " cm3" << std::endl;
  ;
  Output << "     Surface = " << activeSurface_tid << " cm2" << std::endl;
  Output << "        APV25s   = " << (int)apv_tid << std::endl;
  Output << "        channels = " << (int)chan_tid << std::endl;
  Output << " TOB    = " << tobN << std::endl;
  Output << "   L12 rphi   = " << tob_L12_rphiN << std::endl;
  Output << "   L12 stereo = " << tob_L12_sterN << std::endl;
  Output << "   L34        = " << tob_L34_rphiN << std::endl;
  Output << "   L56        = " << tob_L56_rphiN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_tob << " kg" << std::endl;
  Output << "     Volume  = " << volume_tob << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_tob << " cm2" << std::endl;
  Output << "        APV25s   = " << (int)apv_tob << std::endl;
  Output << "        channels = " << (int)chan_tob << std::endl;
  Output << " TEC    = " << tecN << std::endl;
  Output << "   r1 rphi    = " << tec_r1_rphiN << std::endl;
  Output << "   r1 stereo  = " << tec_r1_sterN << std::endl;
  Output << "   r2 rphi    = " << tec_r2_rphiN << std::endl;
  Output << "   r2 stereo  = " << tec_r2_sterN << std::endl;
  Output << "   r3 rphi    = " << tec_r3_rphiN << std::endl;
  Output << "   r4 rphi    = " << tec_r4_rphiN << std::endl;
  Output << "   r5 rphi    = " << tec_r5_rphiN << std::endl;
  Output << "   r5 stereo  = " << tec_r5_sterN << std::endl;
  Output << "   r6 rphi    = " << tec_r6_rphiN << std::endl;
  Output << "   r7 rphi    = " << tec_r7_rphiN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_tec << " kg" << std::endl;
  Output << "     Volume  = " << volume_tec << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_tec << " cm2" << std::endl;
  Output << "        APV25s   = " << (int)apv_tec << std::endl;
  Output << "        channels = " << (int)chan_tec << std::endl;
  Output << "---------------------" << std::endl;
  Output << " Total Weight      = " << weight_total << " kg" << std::endl;
  Output << " Total Volume      = " << volume_total << " cm3" << std::endl;
  Output << " Total Active Area = " << activeSurface_total << " cm2" << std::endl;
  Output << "        PSI46s   = " << (int)psi_tot << std::endl;
  Output << "        APV25s   = " << (int)apv_tot << std::endl;
  Output << "        pixel channels = " << (int)chan_pixel << std::endl;
  Output << "        strip channels = " << (int)chan_strip << std::endl;
  Output << "        total channels = " << (int)chan_tot << std::endl;
  //
  for (unsigned int i = 0; i < nlayersPXB; i++) {
    GeometryOutput << "   PXB Layer no. " << i + 1 << std::endl;
    GeometryOutput << "        Mean radius of layer no. " << i + 1 << ": "
                   << pxbR_L[i] / (pxb_full_L[i] + pxb_half_L[i] + pxb_stack[i]) << " [cm]" << std::endl;
    GeometryOutput << "        Maximum length in Z of layer no. " << i + 1 << ": " << pxbZ_L[i] << " [cm]" << std::endl;
    GeometryOutput << "        Number of Full module in PXB layer no. " << i + 1 << ": " << pxb_full_L[i] << std::endl;
    GeometryOutput << "        Number of Half module in PXB layer no. " << i + 1 << ": " << pxb_half_L[i] << std::endl;
    GeometryOutput << "        Number of stack module in PXB layer no. " << i + 1 << ": " << pxb_stack[i] << std::endl;
    GeometryOutput << "        Active Silicon surface in PXB layer no. " << i + 1 << ": " << activeSurface_pxb_L[i]
                   << " [cm^2]" << std::endl;
    GeometryOutput << "        Number of PSI46s in PXB layer no. " << i + 1 << ": " << psi_pxb_L[i] << std::endl;
    GeometryOutput << "        Number of pixel channels in PXB layer no. " << i + 1 << ": "
                   << (int)psi_pxb_L[i] * chan_per_psiB[i] << std::endl;
    GeometryOutput << "        Pitch X & Y (microns) of PXB layer no. " << i + 1 << ": " << pxbpitchx[i] << " & "
                   << pxbpitchy[i] << std::endl;
    GeometryOutput << std::endl;
    GeometryXLS << "PXB" << i + 1 << " " << pxbR_L[i] / (pxb_full_L[i] + pxb_half_L[i] + pxb_stack[i]) << " " << 0
                << " " << pxbZ_L[i] << " " << activeSurface_pxb_L[i] << " " << psi_pxb_L[i] << " "
                << (int)psi_pxb_L[i] * chan_per_psiB[i] << " " << pxb_full_L[i] + pxb_half_L[i] + pxb_stack[i] << " "
                << pxb_full_L[i] << " " << pxb_half_L[i] << " " << pxb_stack[i] << std::endl;
  }
  for (unsigned int i = 0; i < nlayersTIB; i++) {
    GeometryOutput << "   TIB Layer no. " << i + 1 << std::endl;
    GeometryOutput << "        Meam radius of layer no. " << i + 1 << ": "
                   << tibR_L[i] / (tib_L12_rphi_L[i] + tib_L12_ster_L[i] + tib_L34_rphi_L[i]) << " [cm]" << std::endl;
    GeometryOutput << "        Maximum length in Z of layer no. " << i + 1 << ": " << tibZ_L[i] << " [cm]" << std::endl;
    if (tib_L12_rphi_L[i] != 0)
      GeometryOutput << "        Number of IB1 rphi minimodules in TIB layer no. " << i + 1 << ": " << tib_L12_rphi_L[i]
                     << std::endl;
    if (tib_L12_ster_L[i] != 0)
      GeometryOutput << "        Number of IB1 stereo minimodules in TIB layer no. " << i + 1 << ": "
                     << tib_L12_ster_L[i] << std::endl;
    if (tib_L34_rphi_L[i] != 0)
      GeometryOutput << "        Number of IB2 rphi minimodules in TIB layer no. " << i + 1 << ": " << tib_L34_rphi_L[i]
                     << std::endl;
    GeometryOutput << "        Active Silicon surface in TIB layer no. " << i + 1 << ": " << activeSurface_tib_L[i]
                   << std::endl;
    GeometryOutput << "        Number of APV25s in TIB layer no. " << i + 1 << ": " << tib_apv_L[i] << std::endl;
    GeometryOutput << "        Number of strip channels in TIB layer no. " << i + 1 << ": "
                   << (int)tib_apv_L[i] * chan_per_apv << std::endl;
    GeometryOutput << std::endl;
    GeometryXLS << "TIB" << i + 1 << " " << tibR_L[i] / (tib_L12_rphi_L[i] + tib_L12_ster_L[i] + tib_L34_rphi_L[i])
                << " " << 0 << " " << tibZ_L[i] << " " << activeSurface_tib_L[i] << " " << tib_apv_L[i] << " "
                << (int)tib_apv_L[i] * chan_per_apv << " " << tib_L12_rphi_L[i] + tib_L12_ster_L[i] + tib_L34_rphi_L[i]
                << " " << tib_L12_rphi_L[i] << " " << tib_L12_ster_L[i] << " " << tib_L34_rphi_L[i] << std::endl;
  }
  for (unsigned int i = 0; i < nlayersTOB; i++) {
    GeometryOutput << "   TOB Layer no. " << i + 1 << std::endl;
    GeometryOutput << "        Meam radius of layer no. " << i + 1 << ": "
                   << tobR_L[i] / (tob_L12_rphi_L[i] + tob_L12_ster_L[i] + tob_L34_rphi_L[i] + tob_L56_rphi_L[i])
                   << " [cm]" << std::endl;
    GeometryOutput << "        Maximum length in Z of layer no. " << i + 1 << ": " << tobZ_L[i] << " [cm]" << std::endl;
    if (tob_L12_rphi_L[i] != 0)
      GeometryOutput << "        Number of OB1 rphi minimodules in TOB layer no. " << i + 1 << ": " << tob_L12_rphi_L[i]
                     << std::endl;
    if (tob_L12_ster_L[i] != 0)
      GeometryOutput << "        Number of OB1 stereo minimodules in TOB layer no. " << i + 1 << ": "
                     << tob_L12_ster_L[i] << std::endl;
    if (tob_L34_rphi_L[i] != 0)
      GeometryOutput << "        Number of OB1 rphi minimodules in TOB layer no. " << i + 1 << ": " << tob_L34_rphi_L[i]
                     << std::endl;
    if (tob_L56_rphi_L[i] != 0)
      GeometryOutput << "        Number of OB2 rphi minimodules in TOB layer no. " << i + 1 << ": " << tob_L56_rphi_L[i]
                     << std::endl;
    GeometryOutput << "        Active Silicon surface in TOB layer no. " << i + 1 << ": " << activeSurface_tob_L[i]
                   << std::endl;
    GeometryOutput << "        Number of APV25s in TOB layer no. " << i + 1 << ": " << tob_apv_L[i] << std::endl;
    GeometryOutput << "        Number of strip channels in TOB layer no. " << i + 1 << ": "
                   << (int)tob_apv_L[i] * chan_per_apv << std::endl;
    GeometryOutput << std::endl;
    GeometryXLS << "TOB" << i + 1 << " "
                << tobR_L[i] / (tob_L12_rphi_L[i] + tob_L12_ster_L[i] + tob_L34_rphi_L[i] + tob_L56_rphi_L[i]) << " "
                << 0 << " " << tobZ_L[i] << " " << activeSurface_tob_L[i] << " " << tob_apv_L[i] << " "
                << (int)tob_apv_L[i] * chan_per_apv << " "
                << tob_L12_rphi_L[i] + tob_L12_ster_L[i] + tob_L34_rphi_L[i] + tob_L56_rphi_L[i] << " "
                << tob_L12_rphi_L[i] << " " << tob_L12_ster_L[i] << " " << tob_L34_rphi_L[i] << " " << tob_L56_rphi_L[i]
                << std::endl;
  }
  for (unsigned int i = 0; i < ndisksPXF; i++) {
    GeometryOutput << "   PXF Disk no. " << i + 1 << " (numbers are the total for both sides)" << std::endl;
    GeometryOutput << "        Minimum radius of disk no. " << i + 1 << ": " << pxfR_min_D[i] << " [cm]" << std::endl;
    GeometryOutput << "        Maximum radius of disk no. " << i + 1 << ": " << pxfR_max_D[i] << " [cm]" << std::endl;
    GeometryOutput << "        Position in Z of disk no. " << i + 1 << ": "
                   << pxfZ_D[i] / (pxf_D[i] + pxf_1x2_D[i] + pxf_1x5_D[i] + pxf_2x3_D[i] + pxf_2x4_D[i] + pxf_2x5_D[i])
                   << " [cm]" << std::endl;
    GeometryOutput << "        Number of 1x2 modules in PXF disk no. " << i + 1 << ": " << pxf_1x2_D[i] << std::endl;
    GeometryOutput << "        Number of 1x5 modules in PXF disk no. " << i + 1 << ": " << pxf_1x5_D[i] << std::endl;
    GeometryOutput << "        Number of 2x3 modules in PXF disk no. " << i + 1 << ": " << pxf_2x3_D[i] << std::endl;
    GeometryOutput << "        Number of 2x4 modules in PXF disk no. " << i + 1 << ": " << pxf_2x4_D[i] << std::endl;
    GeometryOutput << "        Number of 2x5 modules in PXF disk no. " << i + 1 << ": " << pxf_2x5_D[i] << std::endl;
    GeometryOutput << "        Number of 2x8 modules in PXF disk no. " << i + 1 << ": " << pxf_D[i] << std::endl;
    GeometryOutput << "        Active Silicon surface in PXF disk no. " << i + 1 << ": " << activeSurface_pxf_D[i]
                   << " [cm^2]" << std::endl;
    GeometryOutput << "        Number of PSI46s in PXF disk no. " << i + 1 << ": " << psi_pxf_D[i] << std::endl;
    GeometryOutput << "        Number of pixel channels in PXF disk no. " << i + 1 << ": "
                   << (int)psi_pxf_D[i] * chan_per_psiD[i] << std::endl;
    GeometryOutput << "        Pitch X & Y (microns) of PXF disk no. " << i + 1 << ": " << pxfpitchx[i] << " & "
                   << pxfpitchy[i] << std::endl;
    GeometryOutput << std::endl;
    GeometryXLS << "PXF" << i + 1 << " " << pxfR_min_D[i] << " " << pxfR_max_D[i] << " "
                << pxfZ_D[i] / (pxf_D[i] + pxf_1x2_D[i] + pxf_1x5_D[i] + pxf_2x3_D[i] + pxf_2x4_D[i] + pxf_2x5_D[i])
                << " " << activeSurface_pxf_D[i] << " " << psi_pxf_D[i] << " " << (int)psi_pxf_D[i] * chan_per_psiD[i]
                << " " << pxf_D[i] + pxf_1x2_D[i] + pxf_1x5_D[i] + pxf_2x3_D[i] + pxf_2x4_D[i] + pxf_2x5_D[i] << " "
                << pxf_D[i] << " " << pxf_1x2_D[i] << " " << pxf_1x5_D[i] << " " << pxf_2x3_D[i] << " " << pxf_2x4_D[i]
                << " " << pxf_2x5_D[i] << std::endl;
  }
  for (unsigned int i = 0; i < ndisksTID; i++) {
    GeometryOutput << "   TID Disk no. " << i + 1 << " (numbers are the total for both sides)" << std::endl;
    GeometryOutput << "        Minimum radius of disk no. " << i + 1 << ": " << tidR_min_D[i] << " [cm]" << std::endl;
    GeometryOutput << "        Maximum radius of disk no. " << i + 1 << ": " << tidR_max_D[i] << " [cm]" << std::endl;
    int tot = tid_r1_rphi_D[i] + tid_r1_ster_D[i] + tid_r2_rphi_D[i] + tid_r2_ster_D[i] + tid_r3_rphi_D[i];
    GeometryOutput << "        Position in Z of disk no. " << i + 1 << ": " << tidZ_D[i] / tot << " [cm]" << std::endl;
    GeometryOutput << "        Number of r1_rphi modules in TID disk no. " << i + 1 << ": " << tid_r1_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r1_ster modules in TID disk no. " << i + 1 << ": " << tid_r1_ster_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r2_rphi modules in TID disk no. " << i + 1 << ": " << tid_r2_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r2_ster modules in TID disk no. " << i + 1 << ": " << tid_r2_ster_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r3_rphi modules in TID disk no. " << i + 1 << ": " << tid_r3_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Active Silicon surface in TID disk no. " << i + 1 << ": " << activeSurface_tid_D[i]
                   << " [cm^2]" << std::endl;
    GeometryOutput << "        Number of APV25s in TID disk no. " << i + 1 << ": " << tid_apv_D[i] << std::endl;
    GeometryOutput << "        Number of strip channels in TID disk no. " << i + 1 << ": "
                   << (int)tid_apv_D[i] * chan_per_apv << std::endl;
    GeometryOutput << std::endl;
    GeometryXLS << "TID" << i + 1 << " " << tidR_min_D[i] << " " << tidR_max_D[i] << " " << tidZ_D[i] / tot << " "
                << activeSurface_tid_D[i] << " " << tid_apv_D[i] << " " << (int)tid_apv_D[i] * chan_per_apv << " "
                << tot << " " << tid_r1_rphi_D[i] << " " << tid_r1_ster_D[i] << " " << tid_r2_rphi_D[i] << " "
                << tid_r2_ster_D[i] << " " << tid_r3_rphi_D[i] << std::endl;
  }
  for (unsigned int i = 0; i < nwheelsTEC; i++) {
    GeometryOutput << "   TEC Disk no. " << i + 1 << " (numbers are the total for both sides)" << std::endl;
    GeometryOutput << "        Minimum radius of wheel no. " << i + 1 << ": " << tecR_min_D[i] << " [cm]" << std::endl;
    GeometryOutput << "        Maximum radius of wheel no. " << i + 1 << ": " << tecR_max_D[i] << " [cm]" << std::endl;
    int tot = tec_r1_rphi_D[i] + tec_r1_ster_D[i] + tec_r2_rphi_D[i] + tec_r2_ster_D[i] + tec_r3_rphi_D[i] +
              tec_r4_rphi_D[i] + tec_r5_rphi_D[i] + tec_r5_ster_D[i] + tec_r6_rphi_D[i] + tec_r7_rphi_D[i];
    GeometryOutput << "        Position in Z of wheel no. " << i + 1 << ": " << tecZ_D[i] / tot << " [cm]" << std::endl;
    GeometryOutput << "        Number of r1_rphi modules in TEC wheel no. " << i + 1 << ": " << tec_r1_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r1_ster modules in TEC wheel no. " << i + 1 << ": " << tec_r1_ster_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r2_rphi modules in TEC wheel no. " << i + 1 << ": " << tec_r2_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r2_ster modules in TEC wheel no. " << i + 1 << ": " << tec_r2_ster_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r3_rphi modules in TEC wheel no. " << i + 1 << ": " << tec_r3_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r4_rphi modules in TEC wheel no. " << i + 1 << ": " << tec_r4_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r5_rphi modules in TEC wheel no. " << i + 1 << ": " << tec_r5_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r5_ster modules in TEC wheel no. " << i + 1 << ": " << tec_r5_ster_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r6_rphi modules in TEC wheel no. " << i + 1 << ": " << tec_r6_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Number of r7_rphi modules in TEC wheel no. " << i + 1 << ": " << tec_r7_rphi_D[i]
                   << std::endl;
    GeometryOutput << "        Active Silicon surface in TEC wheel no. " << i + 1 << ": " << activeSurface_tec_D[i]
                   << " [cm^2]" << std::endl;
    GeometryOutput << "        Number of APV25s in TEC wheel no. " << i + 1 << ": " << tec_apv_D[i] << std::endl;
    GeometryOutput << "        Number of strip channels in TEC wheel no. " << i + 1 << ": "
                   << (int)tec_apv_D[i] * chan_per_apv << std::endl;
    GeometryOutput << std::endl;
    GeometryXLS << "TEC" << i + 1 << " " << tecR_min_D[i] << " " << tecR_max_D[i] << " " << tecZ_D[i] / tot << " "
                << activeSurface_tec_D[i] << " " << tec_apv_D[i] << " " << (int)tec_apv_D[i] * chan_per_apv << " "
                << tot << " " << tec_r1_rphi_D[i] << " " << tec_r1_ster_D[i] << " " << tec_r2_rphi_D[i] << " "
                << tec_r2_ster_D[i] << " " << tec_r3_rphi_D[i] << " " << tec_r4_rphi_D[i] << " " << tec_r5_rphi_D[i]
                << " " << tec_r5_ster_D[i] << " " << tec_r6_rphi_D[i] << " " << tec_r7_rphi_D[i] << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ModuleInfo_Phase2);
