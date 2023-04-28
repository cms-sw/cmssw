#include "SimG4Core/Geometry/interface/CMSG4CheckOverlap.h"
#include "SimG4Core/Geometry/interface/CMSG4RegionReporter.h"
#include "SimG4Core/Geometry/interface/CustomUIsession.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4GDMLParser.hh"
#include "G4GeomTestVolume.hh"
#include "G4RunManagerKernel.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4GeometryManager.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4Element.hh"
#include "G4ElementTable.hh"
#include "G4Material.hh"
#include "G4MaterialTable.hh"
#include "G4ProductionCutsTable.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4SystemOfUnits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4UnitsTable.hh"
#include "G4ios.hh"
#include "globals.hh"

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

CMSG4CheckOverlap::CMSG4CheckOverlap(const edm::ParameterSet& p,
                                     std::string& regFile,
                                     CustomUIsession* session,
                                     G4VPhysicalVolume* world) {
  bool mat = p.getParameter<bool>("MaterialFlag");
  const std::string& ss = p.getParameter<std::string>("OutputBaseName");
  if (ss.empty()) {
    edm::LogWarning("SimG4CoreGeometry")
        << "CMSG4CheckOverlap: OutputFileBaseName is not provided - no check is performed";
    return;
  }
  if (mat) {
    const std::string sss = "Materials_" + ss + ".txt";
    std::ofstream fout(sss.c_str(), std::ios::out);
    if (fout.fail()) {
      edm::LogWarning("SimG4CoreGeometry")
          << "CMSG4CheckOverlap: file <" << sss << "> is not opened - no report provided";
    } else {
      edm::LogVerbatim("SimG4CoreGeometry") << "CMSG4CheckOverlap: output file <" << sss << "> is opened";
      makeReportForMaterials(fout);
      fout.close();
    }
  }

  bool reg = p.getParameter<bool>("RegionFlag");
  if (reg) {
    const std::string qqq = (regFile.empty()) ? ss : regFile;
    const std::string sss = "Regions_" + qqq + ".txt";
    CMSG4RegionReporter rrep;
    rrep.ReportRegions(sss);
  }

  bool geom = p.getParameter<bool>("GeomFlag");
  if (geom) {
    const std::string sss = "Geometry_" + ss + ".txt";
    std::ofstream fout(sss.c_str(), std::ios::out);
    if (fout.fail()) {
      edm::LogWarning("SimG4CoreGeometry")
          << "CMSG4CheckOverlap: file <" << sss << "> is not opened - no report provided";
    } else {
      edm::LogVerbatim("SimG4CoreGeometry") << "CMSG4CheckOverlap: output file <" << sss << "> is opened";
      session->sendToFile(&fout);
      makeReportForGeometry(fout, world);
      session->stopSendToFile();
      fout.close();
    }
  }

  bool gdmlFlag = p.getParameter<bool>("gdmlFlag");
  if (gdmlFlag) {
    std::string PVname = p.getParameter<std::string>("PVname");
    if(PVname.empty() || PVname == "world" || PVname == "World") { 
      G4GDMLParser gdml = nullptr;
      gdml.SetRegionExport(true);
      gdml.SetEnergyCutsExport(true);
      gdml.SetSDExport(true);
      gdml.Write(ss + ".gdml", world, true);
    }
  }

  bool oFlag = p.getParameter<bool>("OverlapFlag");
  if (oFlag) {
    const std::string sss = "Overlaps_" + ss + ".txt";
    std::ofstream fout(sss.c_str(), std::ios::out);
    if (fout.fail()) {
      edm::LogWarning("SimG4CoreGeometry")
          << "CMSG4CheckOverlap: file <" << sss << "> is not opened - no report provided";
    } else {
      edm::LogVerbatim("SimG4CoreGeometry") << "CMSG4CheckOverlap: output file <" << ss << "> is opened";
      session->sendToFile(&fout);
      makeReportForOverlaps(fout, p, world);
      session->stopSendToFile();
      fout.close();
    }
  }
}

void CMSG4CheckOverlap::makeReportForMaterials(std::ofstream& fout) {
  int nelm = G4Element::GetNumberOfElements();
  int nmat = G4Material::GetNumberOfMaterials();
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  int ncouples = theCoupleTable->GetTableSize();
  fout << "====================================================================="
       << "\n";
  fout << "NumberOfElements  " << nelm << "\n";
  fout << "NumberOfMaterials " << nmat << "\n";
  fout << "NumberOfCouples   " << ncouples << "\n";
  fout << "====================================================================="
       << "\n";
  fout << "ElementsDump:"
       << "\n";
  G4ElementTable* elmtab = G4Element::GetElementTable();
  fout << *elmtab;
  fout << "====================================================================="
       << "\n";
  G4MaterialTable* mattab = G4Material::GetMaterialTable();
  fout << "MaterialsDump:"
       << "\n";
  //fout << *mattab << "\n";
  for (int i = 0; i < nmat; ++i) {
    fout << "### Material " << i << "   " << ((*mattab)[i])->GetNumberOfElements() << " elements\n";
    fout << (*mattab)[i] << "\n";
  }
  fout << "====================================================================="
       << "\n";
  fout << "MaterialsCutsCoupleDump:"
       << "\n";
  const std::vector<G4double>* gcut = theCoupleTable->GetEnergyCutsVector(0);
  const std::vector<G4double>* ecut = theCoupleTable->GetEnergyCutsVector(1);
  const std::vector<G4double>* pcut = theCoupleTable->GetEnergyCutsVector(2);
  const std::vector<G4double>* icut = theCoupleTable->GetEnergyCutsVector(3);
  for (int i = 0; i < ncouples; ++i) {
    const G4MaterialCutsCouple* couple = theCoupleTable->GetMaterialCutsCouple(i);
    const G4ProductionCuts* aCut = couple->GetProductionCuts();
    fout << "Index : " << i << "  used in the geometry : ";
    if (couple->IsUsed()) {
      fout << "Yes\n";
    } else {
      fout << "No \n";
    }
    fout << " Material : " << couple->GetMaterial()->GetName() << "\n";
    fout << " Range cuts        : "
         << " gamma  " << G4BestUnit(aCut->GetProductionCut(0), "Length") << "    e-  "
         << G4BestUnit(aCut->GetProductionCut(1), "Length") << "    e+  "
         << G4BestUnit(aCut->GetProductionCut(2), "Length") << " proton "
         << G4BestUnit(aCut->GetProductionCut(3), "Length") << "\n";
    fout << " Energy thresholds : ";
    fout << " gamma  " << G4BestUnit((*gcut)[i], "Energy") << "    e-  " << G4BestUnit((*ecut)[i], "Energy")
         << "    e+  " << G4BestUnit((*pcut)[i], "Energy") << " proton " << G4BestUnit((*icut)[i], "Energy") << "\n";
  }
  fout << "======================================================================"
       << "\n";
}

void CMSG4CheckOverlap::makeReportForGeometry(std::ofstream& fout, G4VPhysicalVolume* world) {
  const G4RegionStore* regs = G4RegionStore::GetInstance();
  const G4PhysicalVolumeStore* pvs = G4PhysicalVolumeStore::GetInstance();
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  int numPV = pvs->size();
  int numLV = lvs->size();
  int nreg = regs->size();
  fout << "====================================================================="
       << "\n";
  fout << "NumberOfRegions         " << nreg << "\n";
  fout << "NumberOfLogicalVolumes  " << numLV << "\n";
  fout << "NumberOfPhysicalVolumes " << numPV << "\n";
  fout << "====================================================================="
       << "\n";
  G4GeometryManager::GetInstance()->CloseGeometry(true, true, world);
  fout << "====================================================================="
       << "\n";
}

void CMSG4CheckOverlap::makeReportForOverlaps(std::ofstream& fout, const edm::ParameterSet& p, G4VPhysicalVolume* world) {
  std::vector<std::string> nodeNames = p.getParameter<std::vector<std::string>>("NodeNames");
  std::string PVname = p.getParameter<std::string>("PVname");
  std::string LVname = p.getParameter<std::string>("LVname");
  double tolerance = p.getParameter<double>("Tolerance") * CLHEP::mm;
  int nPoints = p.getParameter<int>("Resolution");
  bool verbose = p.getParameter<bool>("Verbose");
  bool regionFlag = p.getParameter<bool>("RegionFlag");
  bool gdmlFlag = p.getParameter<bool>("gdmlFlag");
  int nPrints = p.getParameter<int>("ErrorThreshold");

  const G4RegionStore* regStore = G4RegionStore::GetInstance();

  G4LogicalVolume* lv;
  const G4PhysicalVolumeStore* pvs = G4PhysicalVolumeStore::GetInstance();
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  unsigned int numPV = pvs->size();
  unsigned int numLV = lvs->size();
  unsigned int nn = nodeNames.size();

  std::vector<G4String> savedgdml;

  fout << "====================================================================="
       << "\n";
  fout << "CMSG4OverlapCheck is initialised with " << nodeNames.size() << " nodes; "
       << " nPoints= " << nPoints << "; tolerance= " << tolerance / mm << " mm; verbose: " << verbose << "\n"
       << "               RegionFlag: " << regionFlag << "  PVname: " << PVname << "  LVname: " << LVname << "\n"
       << "               Nlv= " << numLV << "   Npv= " << numPV << "\n";
  fout << "====================================================================="
       << "\n";

  G4GDMLParser* gdml = nullptr;
  if (gdmlFlag) {
    gdml = new G4GDMLParser(); 
    gdml->SetRegionExport(true);
    gdml->SetEnergyCutsExport(true);
    gdml->SetSDExport(true);
  }
  if (0 < nn) {
    for (unsigned int ii = 0; ii < nn; ++ii) {
      if (nodeNames[ii].empty() || "world" == nodeNames[ii] || "World" == nodeNames[ii]) {
        nodeNames[ii] = world->GetName();
        fout << "### Check overlaps for World " << "\n";
        G4GeomTestVolume test(world, tolerance, nPoints, verbose);
        test.SetErrorsThreshold(nPrints);
        test.TestOverlapInTree();
      } else if (regionFlag) {
        fout << "---------------------------------------------------------------" << "\n";
        fout << "### Check overlaps for G4Region Node[" << ii << "] : " << nodeNames[ii] << "\n";
        G4Region* reg = regStore->GetRegion((G4String)nodeNames[ii]);
        if (!reg) {
          fout << "### NO G4Region found - EXIT" << "\n";
          return;
        }
        std::vector<G4LogicalVolume*>::iterator rootLVItr = reg->GetRootLogicalVolumeIterator();
        unsigned int numRootLV = reg->GetNumberOfRootVolumes();
        fout << "      " << numRootLV << " Root Logical Volumes in this region"
             << "\n";

        for (unsigned int iLV = 0; iLV < numRootLV; ++iLV, ++rootLVItr) {
          // Cover each root logical volume in this region
          lv = *rootLVItr;
          fout << "### Check overlaps for G4LogicalVolume " << lv->GetName() << "\n";
          for (unsigned int i = 0; i < numPV; ++i) {
            if (((*pvs)[i])->GetLogicalVolume() == lv) {
              G4String pvname = ((*pvs)[i])->GetName();
              G4bool isNew = true;
              for (unsigned int k = 0; k < savedgdml.size(); ++k) {
                if (pvname == savedgdml[k]) {
                  isNew = false;
                  break;
                }
              }
              if (!isNew) {
                fout << "### Check overlaps for PhysVolume  " << pvname 
                     << " is skipted because was already done" << "\n";
                continue;
              }
              savedgdml.push_back(pvname);
              fout << "### Check overlaps for PhysVolume  " << pvname << "\n";
              // gdml dump only for 1 volume
              if (gdmlFlag) {
                gdml->Write(pvname + ".gdml", (*pvs)[i], true);
              }
              G4GeomTestVolume test(((*pvs)[i]), tolerance, nPoints, verbose);
              test.SetErrorsThreshold(nPrints);
              test.TestOverlapInTree();
            }
          }
        }
      } else {
        fout << "### Check overlaps for PhysVolume Node[" << ii << "] : " << nodeNames[ii] << "\n";
        G4VPhysicalVolume* pv = pvs->GetVolume((G4String)nodeNames[ii]);
        G4GeomTestVolume test(pv, tolerance, nPoints, verbose);
        test.SetErrorsThreshold(nPrints);
        test.TestOverlapInTree();
      }
    }
  }
  if (!PVname.empty()) {
    fout << "----------- List of PhysVolumes by name -----------------" << "\n";
    for (unsigned int i = 0; i < numPV; ++i) {
      if (PVname == (*pvs)[i]->GetName()) {
        fout << " ##### PhysVolume " << PVname << " [" << ((*pvs)[i])->GetCopyNo()
             << "]  LV: " << ((*pvs)[i])->GetLogicalVolume()->GetName()
             << " Mother LV: " << ((*pvs)[i])->GetMotherLogical()->GetName()
             << " Region: " << ((*pvs)[i])->GetLogicalVolume()->GetRegion()->GetName() << "\n";
        fout << "       Translation: " << ((*pvs)[i])->GetObjectTranslation() << "\n";
        fout << "       Rotation:    " << ((*pvs)[i])->GetObjectRotationValue() << "\n";
	if (gdmlFlag) {
          gdml->Write(PVname + ".gdml", (*pvs)[i], true);
        }
      }
    }
  }
  if (!LVname.empty()) {
    fout << "---------- List of Logical Volumes by name ------------------"
         << "\n";
    for (unsigned int i = 0; i < numLV; ++i) {
      if (LVname == ((*lvs)[i])->GetName()) {
        G4int np = ((*lvs)[i])->GetNoDaughters();
        fout << " ##### LogVolume " << LVname << "  " << np << " daughters"
             << " Region: " << ((*lvs)[i])->GetRegion()->GetName() << "\n";
        fout << *(((*lvs)[i])->GetSolid()) << "\n";
        for (G4int j = 0; j < np; ++j) {
          G4VPhysicalVolume* pv = ((*lvs)[i])->GetDaughter(j);
          if (pv) {
            fout << "   PV: " << pv->GetName() << " [" << pv->GetCopyNo() << "]"
                 << " type: " << pv->VolumeType() << "  multiplicity: " << pv->GetMultiplicity()
                 << " LV: " << pv->GetLogicalVolume()->GetName() << "\n";
            fout << "       Translation: " << pv->GetObjectTranslation() << "\n";
            fout << "       Rotation:    " << pv->GetObjectRotationValue() << "\n";
            fout << *(pv->GetLogicalVolume()->GetSolid()) << "\n";
          }
        }
      }
    }
  }
  fout << "---------------- End of overlap checks ---------------------"
       << "\n";
  delete gdml;
}
