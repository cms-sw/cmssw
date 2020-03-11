#include "SimG4Core/PrintGeomInfo/interface/PrintGeomInfoAction.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4Run.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Material.hh"
#include "G4Track.hh"
#include "G4VisAttributes.hh"
#include "G4UserLimits.hh"
#include "G4TransportationManager.hh"

#include <set>
#include <map>

PrintGeomInfoAction::PrintGeomInfoAction(const edm::ParameterSet &p) {
  _dumpSummary = p.getUntrackedParameter<bool>("DumpSummary", true);
  _dumpLVTree = p.getUntrackedParameter<bool>("DumpLVTree", true);
  _dumpMaterial = p.getUntrackedParameter<bool>("DumpMaterial", false);
  _dumpLVList = p.getUntrackedParameter<bool>("DumpLVList", false);
  _dumpLV = p.getUntrackedParameter<bool>("DumpLV", false);
  _dumpSolid = p.getUntrackedParameter<bool>("DumpSolid", false);
  _dumpAtts = p.getUntrackedParameter<bool>("DumpAttributes", false);
  _dumpPV = p.getUntrackedParameter<bool>("DumpPV", false);
  _dumpRotation = p.getUntrackedParameter<bool>("DumpRotation", false);
  _dumpReplica = p.getUntrackedParameter<bool>("DumpReplica", false);
  _dumpTouch = p.getUntrackedParameter<bool>("DumpTouch", false);
  _dumpSense = p.getUntrackedParameter<bool>("DumpSense", false);
  name = p.getUntrackedParameter<std::string>("Name", "*");
  nchar = name.find("*");
  name.assign(name, 0, nchar);
  names = p.getUntrackedParameter<std::vector<std::string> >("Names");
  G4cout << "PrintGeomInfoAction:: initialised with verbosity levels:"
         << " Summary   " << _dumpSummary << " LVTree   " << _dumpLVTree << " LVList    " << _dumpLVList << " Material "
         << _dumpMaterial << "\n                                                        "
         << " LV        " << _dumpLV << " Solid    " << _dumpSolid << " Attribs   " << _dumpAtts
         << "\n                                                        "
         << " PV        " << _dumpPV << " Rotation " << _dumpRotation << " Replica   " << _dumpReplica
         << "\n                                                        "
         << " Touchable " << _dumpTouch << " for names (0-" << nchar << ") = " << name
         << "\n                                                        "
         << " Sensitive " << _dumpSense << " for " << names.size() << " namess";
  for (unsigned int i = 0; i < names.size(); i++)
    G4cout << " " << names[i];
  G4cout << G4endl;
}

PrintGeomInfoAction::~PrintGeomInfoAction() {}

void PrintGeomInfoAction::update(const BeginOfJob *job) {
  if (_dumpSense) {
    edm::ESTransientHandle<DDCompactView> pDD;
    (*job)()->get<IdealGeometryRecord>().get(pDD);

    G4cout << "PrintGeomInfoAction::Get Printout of Sensitive Volumes "
           << "for " << names.size() << " Readout Units" << G4endl;
    for (unsigned int i = 0; i < names.size(); i++) {
      std::string attribute = "ReadOutName";
      std::string sd = names[i];
      DDSpecificsMatchesValueFilter filter{DDValue(attribute, sd, 0)};
      DDFilteredView fv(*pDD, filter);
      G4cout << "PrintGeomInfoAction:: Get Filtered view for " << attribute << " = " << sd << G4endl;
      bool dodet = fv.firstChild();

      std::string spaces = spacesFromLeafDepth(1);

      while (dodet) {
        const DDLogicalPart &log = fv.logicalPart();
        std::string lvname = log.name().name();
        DDTranslation tran = fv.translation();
        std::vector<int> copy = fv.copyNumbers();

        unsigned int leafDepth = copy.size();
        G4cout << leafDepth << spaces << "### VOLUME = " << lvname << " Copy No";
        for (int k = leafDepth - 1; k >= 0; k--)
          G4cout << " " << copy[k];
        G4cout << " Centre at " << tran << " (r = " << tran.Rho() << ", phi = " << tran.phi() / CLHEP::deg << ")"
               << G4endl;
        dodet = fv.next();
      }
    }
  }
}

void PrintGeomInfoAction::update(const BeginOfRun *run) {
  theTopPV = getTopPV();

  if (_dumpSummary)
    dumpSummary(G4cout);
  if (_dumpLVTree)
    dumpG4LVTree(G4cout);

  //---------- Dump list of objects of each class with detail of parameters
  if (_dumpMaterial)
    dumpMaterialList(G4cout);
  if (_dumpLVList)
    dumpG4LVList(G4cout);

  //---------- Dump LV and PV information
  if (_dumpLV || _dumpPV || _dumpTouch)
    dumpHierarchyTreePVLV(G4cout);
}

void PrintGeomInfoAction::dumpSummary(std::ostream &out) {
  //---------- Dump number of objects of each class
  out << " @@@@@@@@@@@@@@@@@@ Dumping G4 geometry objects Summary " << G4endl;
  if (theTopPV == nullptr) {
    out << " No volume created " << G4endl;
    return;
  }
  out << " @@@ Geometry built inside world volume: " << theTopPV->GetName() << G4endl;
  // Get number of solids (< # LV if several LV share a solid)
  const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  std::set<G4VSolid *> theSolids;
  for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++)
    theSolids.insert((*lvcite)->GetSolid());
  out << " Number of G4VSolid's: " << theSolids.size() << G4endl;
  out << " Number of G4LogicalVolume's: " << lvs->size() << G4endl;
  const G4PhysicalVolumeStore *pvs = G4PhysicalVolumeStore::GetInstance();
  out << " Number of G4VPhysicalVolume's: " << pvs->size() << G4endl;
  out << " Number of Touchable's: " << countNoTouchables() << G4endl;
  const G4MaterialTable *matTab = G4Material::GetMaterialTable();
  out << " Number of G4Material's: " << matTab->size() << G4endl;
}

void PrintGeomInfoAction::dumpG4LVList(std::ostream &out) {
  out << " @@@@@@@@@@@@@@@@ DUMPING G4LogicalVolume's List  " << G4endl;
  const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++)
    out << "LV:" << (*lvcite)->GetName() << "\tMaterial: " << (*lvcite)->GetMaterial()->GetName() << G4endl;
}

void PrintGeomInfoAction::dumpG4LVTree(std::ostream &out) {
  out << " @@@@@@@@@@@@@@@@ DUMPING G4LogicalVolume's Tree  " << G4endl;
  G4LogicalVolume *lv = getTopLV();
  dumpG4LVLeaf(lv, 0, 1, out);
}

void PrintGeomInfoAction::dumpMaterialList(std::ostream &out) {
  out << " @@@@@@@@@@@@@@@@ DUMPING G4Material List ";
  const G4MaterialTable *matTab = G4Material::GetMaterialTable();
  out << " with " << matTab->size() << " materials " << G4endl;
  std::vector<G4Material *>::const_iterator matite;
  for (matite = matTab->begin(); matite != matTab->end(); matite++)
    out << "Material: " << (*matite) << G4endl;
}

void PrintGeomInfoAction::dumpG4LVLeaf(G4LogicalVolume *lv,
                                       unsigned int leafDepth,
                                       unsigned int count,
                                       std::ostream &out) {
  for (unsigned int ii = 0; ii < leafDepth; ii++)
    out << "  ";
  out << " LV:(" << leafDepth << ") " << lv->GetName() << " (" << count << ")" << G4endl;
  //--- If a volume is placed n types as daughter of this LV, it should only be counted once
  std::map<G4LogicalVolume *, unsigned int> lvCount;
  std::map<G4LogicalVolume *, unsigned int>::const_iterator cite;
  int siz = lv->GetNoDaughters();
  for (int ii = 0; ii < siz; ii++) {
    cite = lvCount.find(lv->GetDaughter(ii)->GetLogicalVolume());
    if (cite != lvCount.end())
      lvCount[cite->first] = (cite->second) + 1;
    else
      lvCount.insert(std::pair<G4LogicalVolume *, unsigned int>(lv->GetDaughter(ii)->GetLogicalVolume(), 1));
  }
  for (cite = lvCount.begin(); cite != lvCount.end(); cite++)
    dumpG4LVLeaf((cite->first), leafDepth + 1, (cite->second), out);
}

int PrintGeomInfoAction::countNoTouchables() {
  int nTouch = 0;
  G4LogicalVolume *lv = getTopLV();
  add1touchable(lv, nTouch);
  return nTouch;
}

void PrintGeomInfoAction::add1touchable(G4LogicalVolume *lv, int &nTouch) {
  int siz = lv->GetNoDaughters();
  for (int ii = 0; ii < siz; ii++)
    add1touchable(lv->GetDaughter(ii)->GetLogicalVolume(), ++nTouch);
}

void PrintGeomInfoAction::dumpHierarchyTreePVLV(std::ostream &out) {
  //dumps in the following order:
  //    1) a LV with details
  //    2) list of PVs daughters of this LV with details
  //    3) list of LVs daughters of this LV and for each go to 1)

  //----- Get top PV
  G4LogicalVolume *topLV = getTopLV();

  //----- Dump this leaf (it will recursively dump all the tree)
  dumpHierarchyLeafPVLV(topLV, 0, out);
  dumpPV(theTopPV, 0, out);

  //----- Dump the touchables (it will recursively dump all the tree)
  if (_dumpTouch)
    dumpTouch(theTopPV, 0, out);
}

void PrintGeomInfoAction::dumpHierarchyLeafPVLV(G4LogicalVolume *lv, unsigned int leafDepth, std::ostream &out) {
  //----- Dump this LV
  dumpLV(lv, leafDepth, out);

  //----- Get LV daughters from list of PV daughters
  mmlvpv lvpvDaughters;
  std::set<G4LogicalVolume *> lvDaughters;
  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume *pvD = lv->GetDaughter(NoDaughters);
    lvpvDaughters.insert(mmlvpv::value_type(pvD->GetLogicalVolume(), pvD));
    lvDaughters.insert(pvD->GetLogicalVolume());
  }

  std::set<G4LogicalVolume *>::const_iterator scite;
  mmlvpv::const_iterator mmcite;

  //----- Dump daughters PV and LV
  for (scite = lvDaughters.begin(); scite != lvDaughters.end(); scite++) {
    std::pair<mmlvpv::iterator, mmlvpv::iterator> mmER = lvpvDaughters.equal_range(*scite);
    //----- Dump daughters PV of this LV
    for (mmcite = mmER.first; mmcite != mmER.second; mmcite++)
      dumpPV((*mmcite).second, leafDepth + 1, out);
    //----- Dump daughters LV
    dumpHierarchyLeafPVLV(*scite, leafDepth + 1, out);
  }
}

void PrintGeomInfoAction::dumpLV(G4LogicalVolume *lv, unsigned int leafDepth, std::ostream &out) {
  std::string spaces = spacesFromLeafDepth(leafDepth);

  //----- dump name
  if (_dumpLV) {
    out << leafDepth << spaces << "$$$ VOLUME = " << lv->GetName() << "  Solid: " << lv->GetSolid()->GetName()
        << "  MATERIAL: " << lv->GetMaterial()->GetName() << G4endl;
    if (_dumpSolid)
      dumpSolid(lv->GetSolid(), leafDepth, out);  //----- dump solid

    //----- dump LV info
    //--- material
    if (_dumpAtts) {
      //--- Visualisation attributes
      const G4VisAttributes *fVA = lv->GetVisAttributes();
      if (fVA != nullptr) {
        out << spaces << "  VISUALISATION ATTRIBUTES: " << G4endl;
        out << spaces << "    IsVisible " << fVA->IsVisible() << G4endl;
        out << spaces << "    IsDaughtersInvisible " << fVA->IsDaughtersInvisible() << G4endl;
        out << spaces << "    Colour " << fVA->GetColour() << G4endl;
        out << spaces << "    LineStyle " << fVA->GetLineStyle() << G4endl;
        out << spaces << "    LineWidth " << fVA->GetLineWidth() << G4endl;
        out << spaces << "    IsForceDrawingStyle " << fVA->IsForceDrawingStyle() << G4endl;
        out << spaces << "    ForcedDrawingStyle " << fVA->GetForcedDrawingStyle() << G4endl;
      }

      //--- User Limits
      G4UserLimits *fUL = lv->GetUserLimits();
      G4Track dummy;
      if (fUL != nullptr) {
        out << spaces << "    MaxAllowedStep " << fUL->GetMaxAllowedStep(dummy) << G4endl;
        out << spaces << "    UserMaxTrackLength " << fUL->GetUserMaxTrackLength(dummy) << G4endl;
        out << spaces << "    UserMaxTime " << fUL->GetUserMaxTime(dummy) << G4endl;
        out << spaces << "    UserMinEkine " << fUL->GetUserMinEkine(dummy) << G4endl;
        out << spaces << "    UserMinRange " << fUL->GetUserMinRange(dummy) << G4endl;
      }

      //--- other LV info
      if (lv->GetSensitiveDetector())
        out << spaces << "  IS SENSITIVE DETECTOR " << G4endl;
      if (lv->GetFieldManager())
        out << spaces << "  FIELD ON " << G4endl;

      // Pointer (possibly NULL) to optimisation info objects.
      out << spaces << "        Quality for optimisation, average number of voxels to be spent per content "
          << lv->GetSmartless() << G4endl;

      // Pointer (possibly NULL) to G4FastSimulationManager object.
      if (lv->GetFastSimulationManager())
        out << spaces << "     Logical Volume is an envelope for a FastSimulationManager " << G4endl;
      out << spaces << "     Weight used in the event biasing technique = " << lv->GetBiasWeight() << G4endl;
    }
  }
}

void PrintGeomInfoAction::dumpPV(G4VPhysicalVolume *pv, unsigned int leafDepth, std::ostream &out) {
  std::string spaces = spacesFromLeafDepth(leafDepth);

  //----- PV info
  if (_dumpPV) {
    std::string mother = "World";
    if (pv->GetMotherLogical())
      mother = pv->GetMotherLogical()->GetName();
    out << leafDepth << spaces << "### VOLUME = " << pv->GetName() << " Copy No " << pv->GetCopyNo() << " in " << mother
        << " at " << pv->GetTranslation();
  }
  if (!pv->IsReplicated()) {
    if (_dumpPV) {
      if (pv->GetRotation() == nullptr)
        out << " with no rotation" << G4endl;
      else if (!_dumpRotation)
        out << " with rotation" << G4endl;  //just rotation name
      else
        out << " with rotation " << *(pv->GetRotation()) << G4endl;
    }
  } else {
    if (_dumpReplica) {
      out << spaces << "    It is replica: " << G4endl;
      EAxis axis;
      int nReplicas;
      double width;
      double offset;
      bool consuming;
      pv->GetReplicationData(axis, nReplicas, width, offset, consuming);
      out << spaces << "     axis " << axis << G4endl << spaces << "     nReplicas " << nReplicas << G4endl;
      if (pv->GetParameterisation() != nullptr)
        out << spaces << "    It is parameterisation " << G4endl;
      else
        out << spaces << "     width " << width << G4endl << spaces << "     offset " << offset << G4endl << spaces
            << "     consuming" << consuming << G4endl;
      if (pv->GetParameterisation() != nullptr)
        out << spaces << "    It is parameterisation " << G4endl;
    }
  }
}

void PrintGeomInfoAction::dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, std::ostream &out) {
  std::string spaces = spacesFromLeafDepth(leafDepth);
  if (leafDepth == 0)
    fHistory.SetFirstEntry(pv);
  else
    fHistory.NewLevel(pv, kNormal, pv->GetCopyNo());

  G4ThreeVector globalpoint = fHistory.GetTopTransform().Inverse().TransformPoint(G4ThreeVector(0, 0, 0));
  G4LogicalVolume *lv = pv->GetLogicalVolume();

  std::string mother = "World";
  if (pv->GetMotherLogical())
    mother = pv->GetMotherLogical()->GetName();
  std::string lvname = lv->GetName();
  lvname.assign(lvname, 0, nchar);
  if (lvname == name)
    out << leafDepth << spaces << "### VOLUME = " << lv->GetName() << " Copy No " << pv->GetCopyNo() << " in " << mother
        << " global position of centre " << globalpoint << " (r = " << globalpoint.perp()
        << ", phi = " << globalpoint.phi() / CLHEP::deg << ")" << G4endl;

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume *pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated())
      dumpTouch(pvD, leafDepth + 1, out);
  }

  if (leafDepth > 0)
    fHistory.BackLevel();
}

std::string PrintGeomInfoAction::spacesFromLeafDepth(unsigned int leafDepth) {
  std::string spaces;
  unsigned int ii;
  for (ii = 0; ii < leafDepth; ii++) {
    spaces += "  ";
  }
  return spaces;
}

void PrintGeomInfoAction::dumpSolid(G4VSolid *sol, unsigned int leafDepth, std::ostream &out) {
  std::string spaces = spacesFromLeafDepth(leafDepth);
  out << spaces << *(sol) << G4endl;
}

G4VPhysicalVolume *PrintGeomInfoAction::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

G4LogicalVolume *PrintGeomInfoAction::getTopLV() { return theTopPV->GetLogicalVolume(); }
