#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include "DD4hep/DD4hepUnits.h"
#include "DD4hep/Filter.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Material.hh"
#include "G4NavigationHistory.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4Run.hh"
#include "G4Track.hh"
#include "G4TransportationManager.hh"
#include "G4UserLimits.hh"
#include "G4VisAttributes.hh"
#include "G4VPhysicalVolume.hh"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using angle_units::operators::convertRadToDeg;

typedef std::multimap<G4LogicalVolume *, G4VPhysicalVolume *, std::less<G4LogicalVolume *> > mmlvpv;

class PrintGeomInfoAction : public SimWatcher, public Observer<const BeginOfRun *> {
public:
  PrintGeomInfoAction(edm::ParameterSet const &p);
  ~PrintGeomInfoAction() override = default;

  void registerConsumes(edm::ConsumesCollector) override;
  void beginRun(edm::EventSetup const &) override;

private:
  void update(const BeginOfRun *run) override;
  void dumpSummary(std::ostream &out = G4cout);
  void dumpG4LVList(std::ostream &out = G4cout);
  void dumpG4LVTree(std::ostream &out = G4cout);
  void dumpG4Region(std::ostream &out = G4cout);
  void dumpMaterialList(std::ostream &out = G4cout);
  void dumpG4LVLeaf(G4LogicalVolume *lv, unsigned int leafDepth, unsigned int count, std::ostream &out = G4cout);
  int countNoTouchables();
  void add1touchable(G4LogicalVolume *lv, int &nTouch);
  void dumpHierarchyTreePVLV(std::ostream &out = G4cout);
  void dumpHierarchyLeafPVLV(G4LogicalVolume *lv, unsigned int leafDepth, std::ostream &out = G4cout);
  void dumpLV(G4LogicalVolume *lv, unsigned int leafDepth, std::ostream &out = G4cout);
  void dumpPV(G4VPhysicalVolume *pv, unsigned int leafDepth, std::ostream &out = G4cout);
  void dumpSolid(G4VSolid *sol, unsigned int leafDepth, std::ostream &out = G4cout);
  void dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, std::ostream &out = G4cout);
  void dumpInFile();
  void getTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, unsigned int copym, std::vector<std::string> &touches);
  std::string spacesFromLeafDepth(unsigned int leafDepth);
  G4VPhysicalVolume *getTopPV();
  G4LogicalVolume *getTopLV();

private:
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> dddToken_;

  bool dumpSummary_, dumpLVTree_, dumpLVList_, dumpMaterial_;
  bool dumpLV_, dumpSolid_, dumpAtts_, dumpPV_;
  bool dumpRotation_, dumpReplica_, dumpTouch_;
  bool dumpSense_, dumpParams_, dumpRegion_, dd4hep_;
  std::string name_;
  int nchar_;
  std::string fileMat_, fileSolid_, fileLV_, filePV_, fileTouch_, fileRegion_;
  bool fileDetail_;
  std::vector<std::string> names_;
  G4VPhysicalVolume *theTopPV_;
  G4NavigationHistory fHistory_;
};

PrintGeomInfoAction::PrintGeomInfoAction(const edm::ParameterSet &p) {
  dumpSummary_ = p.getUntrackedParameter<bool>("DumpSummary", true);
  dumpLVTree_ = p.getUntrackedParameter<bool>("DumpLVTree", true);
  dumpLVList_ = p.getUntrackedParameter<bool>("DumpLVList", false);
  dumpMaterial_ = p.getUntrackedParameter<bool>("DumpMaterial", false);
  dumpLV_ = p.getUntrackedParameter<bool>("DumpLV", false);
  dumpSolid_ = p.getUntrackedParameter<bool>("DumpSolid", false);
  dumpAtts_ = p.getUntrackedParameter<bool>("DumpAttributes", false);
  dumpPV_ = p.getUntrackedParameter<bool>("DumpPV", false);
  dumpRotation_ = p.getUntrackedParameter<bool>("DumpRotation", false);
  dumpReplica_ = p.getUntrackedParameter<bool>("DumpReplica", false);
  dumpTouch_ = p.getUntrackedParameter<bool>("DumpTouch", false);
  dumpSense_ = p.getUntrackedParameter<bool>("DumpSense", false);
  dumpParams_ = p.getUntrackedParameter<bool>("DumpParams", false);
  dumpRegion_ = p.getUntrackedParameter<bool>("DumpRegion", false);
  dd4hep_ = p.getUntrackedParameter<bool>("DD4hep", false);
  name_ = p.getUntrackedParameter<std::string>("Name", "*");
  names_ = p.getUntrackedParameter<std::vector<std::string> >("Names");
  fileMat_ = p.getUntrackedParameter<std::string>("MaterialFileName", "");
  fileSolid_ = p.getUntrackedParameter<std::string>("SolidFileName", "");
  fileLV_ = p.getUntrackedParameter<std::string>("LVFileName", "");
  filePV_ = p.getUntrackedParameter<std::string>("PVFileName", "");
  fileTouch_ = p.getUntrackedParameter<std::string>("TouchFileName", "");
  fileRegion_ = p.getUntrackedParameter<std::string>("RegionFileName", "");
  fileDetail_ = p.getUntrackedParameter<bool>("FileDetail", false);
  nchar_ = name_.find('*');
  name_.assign(name_, 0, nchar_);
  G4cout << "PrintGeomInfoAction:: initialised for dd4hep " << dd4hep_ << " with verbosity levels:"
         << " Summary   " << dumpSummary_ << " LVTree   " << dumpLVTree_ << " LVList    " << dumpLVList_ << " Material "
         << dumpMaterial_ << G4endl << "                                                        "
         << " LV        " << dumpLV_ << " Solid    " << dumpSolid_ << " Attribs   " << dumpAtts_ << G4endl
         << "                                                        "
         << " PV        " << dumpPV_ << " Rotation " << dumpRotation_ << " Replica   " << dumpReplica_ << G4endl
         << "                                                        "
         << " Touchable " << dumpTouch_ << " Rgion " << dumpRegion_ << " for names (0-" << nchar_ << ") = " << name_
         << G4endl << "                                                        "
         << " Sensitive " << dumpSense_ << " Files " << fileMat_ << ":" << fileSolid_ << ":" << fileLV_ << ":"
         << filePV_ << ":" << fileTouch_ << " FileDetail " << fileDetail_ << " fileRegion " << fileRegion_ << G4endl
         << "                                                        for " << names_.size() << " names:";
  for (unsigned int i = 0; i < names_.size(); i++)
    G4cout << " " << names_[i];
  G4cout << G4endl;
}

void PrintGeomInfoAction::registerConsumes(edm::ConsumesCollector cc) {
  if (dd4hep_) {
    dd4hepToken_ = cc.esConsumes<cms::DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
    G4cout << "PrintGeomInfoAction::Initialize ESGetToken for cms::DDCompactView" << G4endl;
  } else {
    dddToken_ = cc.esConsumes<DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
    G4cout << "PrintGeomInfoAction::Initialize ESGetToken for DDCompactView" << G4endl;
  }
}

void PrintGeomInfoAction::beginRun(edm::EventSetup const &es) {
  if (dumpSense_) {
    if (dd4hep_) {
      const cms::DDCompactView *pDD = &es.getData(dd4hepToken_);

      G4cout << "PrintGeomInfoAction::Get Printout of Sensitive Volumes "
             << "for " << names_.size() << " Readout Units" << G4endl;
      for (unsigned int i = 0; i < names_.size(); i++) {
        std::string sd = names_[i];
        const cms::DDFilter filter("ReadOutName", sd);
        cms::DDFilteredView fv(*pDD, filter);
        G4cout << "PrintGeomInfoAction:: Get Filtered view for ReadOutName = " << sd << G4endl;
        G4cout << "Lengths are in mm, angles in degrees" << G4endl;

        std::string spaces = spacesFromLeafDepth(1);

        while (fv.firstChild()) {
          auto tran = fv.translation() / dd4hep::mm;
          std::vector<int> copy = fv.copyNos();
          auto lvname = fv.name();
          unsigned int leafDepth = copy.size();
          G4cout << leafDepth << spaces << "### VOLUME = " << lvname << " Copy No";
          for (unsigned int k = 0; k < leafDepth; ++k)
            G4cout << " " << copy[k];
          if (dumpParams_) {
            G4cout << " parameters";
            for (double val : fv.parameters()) {
              if (std::abs(val) < 1.0) {
                G4cout << std::setprecision(5);
              } else
                G4cout << std::setprecision(6);
              G4cout << " " << val;
            }
            G4cout << G4endl;
          } else {
            G4cout << " Centre at " << tran << " (r = " << tran.Rho() << ", phi = " << convertRadToDeg(tran.phi())
                   << ")" << G4endl;
          }
        }
      }
    } else {
      const DDCompactView *pDD = &es.getData(dddToken_);

      G4cout << "PrintGeomInfoAction::Get Printout of Sensitive Volumes "
             << "for " << names_.size() << " Readout Units" << G4endl;
      for (unsigned int i = 0; i < names_.size(); i++) {
        std::string attribute = "ReadOutName";
        std::string sd = names_[i];
        DDSpecificsMatchesValueFilter filter{DDValue(attribute, sd, 0)};
        DDFilteredView fv(*pDD, filter);
        G4cout << "PrintGeomInfoAction:: Get Filtered view for " << attribute << " = " << sd << G4endl;
        G4cout << "Lengths are in mm, angles in degrees" << G4endl;
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
          G4cout << " Centre at " << tran << " (r = " << tran.Rho() << ", phi = " << convertRadToDeg(tran.phi()) << ")"
                 << G4endl;
          dodet = fv.next();
        }
      }
    }
  }
}

void PrintGeomInfoAction::update(const BeginOfRun *run) {
  //Now take action
  theTopPV_ = getTopPV();

  if (dumpSummary_)
    dumpSummary(G4cout);
  if (dumpLVTree_)
    dumpG4LVTree(G4cout);

  //---------- Dump list of objects of each class with detail of parameters
  if (dumpMaterial_)
    dumpMaterialList(G4cout);
  if (dumpLVList_)
    dumpG4LVList(G4cout);

  //---------- Dump LV and PV information
  if (dumpLV_ || dumpPV_ || dumpTouch_)
    dumpHierarchyTreePVLV(G4cout);

  //---------- Dump Region information
  if (dumpRegion_)
    dumpG4Region(G4cout);

  dumpInFile();
}

void PrintGeomInfoAction::dumpSummary(std::ostream &out) {
  //---------- Dump number of objects of each class
  out << " @@@@@@@@@@@@@@@@@@ Dumping G4 geometry objects Summary " << G4endl;
  if (theTopPV_ == nullptr) {
    out << " No volume created " << G4endl;
    return;
  }
  out << " @@@ Geometry built inside world volume: " << theTopPV_->GetName() << G4endl;
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
  const G4RegionStore *regs = G4RegionStore::GetInstance();
  out << " Number of G4Region's: " << regs->size() << G4endl;
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

void PrintGeomInfoAction::dumpG4Region(std::ostream &out) {
  out << " @@@@@@@@@@@@@@@@ DUMPING G4Region Tree  " << G4endl;
  const G4RegionStore *regs = G4RegionStore::GetInstance();
  std::vector<G4Region *>::const_iterator regite;
  for (regite = regs->begin(); regite != regs->end(); regite++)
    out << "Region: " << (*regite)->GetName() << " with " << (*regite)->GetNumberOfMaterials() << " materials and "
        << (*regite)->GetNumberOfRootVolumes() << " root volumes" << G4endl;
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
  dumpPV(theTopPV_, 0, out);

  //----- Dump the touchables (it will recursively dump all the tree)
  if (dumpTouch_)
    dumpTouch(theTopPV_, 0, out);
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
  if (dumpLV_) {
    out << leafDepth << spaces << "$$$ VOLUME = " << lv->GetName() << "  Solid: " << lv->GetSolid()->GetName()
        << "  MATERIAL: " << lv->GetMaterial()->GetName() << G4endl;
    if (dumpSolid_)
      dumpSolid(lv->GetSolid(), leafDepth, out);  //----- dump solid

    //----- dump LV info
    //--- material
    if (dumpAtts_) {
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
  if (dumpPV_) {
    std::string mother = "World";
    if (pv->GetMotherLogical())
      mother = pv->GetMotherLogical()->GetName();
    out << leafDepth << spaces << "### VOLUME = " << pv->GetName() << " Copy No " << pv->GetCopyNo() << " in " << mother
        << " at " << pv->GetTranslation();
  }
  if (!pv->IsReplicated()) {
    if (dumpPV_) {
      if (pv->GetRotation() == nullptr)
        out << " with no rotation" << G4endl;
      else if (!dumpRotation_)
        out << " with rotation" << G4endl;  //just rotation name
      else
        out << " with rotation " << *(pv->GetRotation()) << G4endl;
    }
  } else {
    if (dumpReplica_) {
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

void PrintGeomInfoAction::dumpSolid(G4VSolid *sol, unsigned int leafDepth, std::ostream &out) {
  std::string spaces = spacesFromLeafDepth(leafDepth);
  out << spaces << *(sol) << G4endl;
}

void PrintGeomInfoAction::dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, std::ostream &out) {
  std::string spaces = spacesFromLeafDepth(leafDepth);
  if (leafDepth == 0)
    fHistory_.SetFirstEntry(pv);
  else
    fHistory_.NewLevel(pv, kNormal, pv->GetCopyNo());

  G4ThreeVector globalpoint = fHistory_.GetTopTransform().Inverse().TransformPoint(G4ThreeVector(0, 0, 0));
  G4LogicalVolume *lv = pv->GetLogicalVolume();

  std::string mother = "World";
  if (pv->GetMotherLogical())
    mother = pv->GetMotherLogical()->GetName();
  std::string lvname = lv->GetName();
  lvname.assign(lvname, 0, nchar_);
  if (lvname == name_)
    out << leafDepth << spaces << "### VOLUME = " << lv->GetName() << " Copy No " << pv->GetCopyNo() << " in " << mother
        << " global position of centre " << globalpoint << " (r = " << globalpoint.perp()
        << ", phi = " << convertRadToDeg(globalpoint.phi()) << ")" << G4endl;

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume *pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated())
      dumpTouch(pvD, leafDepth + 1, out);
  }

  if (leafDepth > 0)
    fHistory_.BackLevel();
}

void PrintGeomInfoAction::dumpInFile() {
  //---------- Dump number objects of each class in a file
  if (theTopPV_ != nullptr) {
    if (!fileMat_.empty()) {
      const G4MaterialTable *matTab = G4Material::GetMaterialTable();
      std::ofstream fout(fileMat_.c_str());
      for (std::vector<G4Material *>::const_iterator matite = matTab->begin(); matite != matTab->end(); matite++) {
        if (!fileDetail_)
          fout << (*matite)->GetName() << G4endl;
        else
          fout << (*matite)->GetName() << " " << (*matite)->GetRadlen() << " " << (*matite)->GetNuclearInterLength()
               << G4endl;
      }
      fout.close();
    }
    const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
    if (!fileSolid_.empty()) {
      std::ofstream fout(fileSolid_.c_str());
      for (std::vector<G4LogicalVolume *>::const_iterator lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++)
        if (!fileDetail_)
          fout << (*lvcite)->GetSolid()->GetName() << G4endl;
        else
          fout << (*lvcite)->GetSolid()->GetName() << "  " << (*lvcite)->GetSolid()->GetCubicVolume() << G4endl;
      fout.close();
    }
    if (!fileLV_.empty()) {
      std::ofstream fout(fileLV_.c_str());
      for (std::vector<G4LogicalVolume *>::const_iterator lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++)
        if (!fileDetail_)
          fout << (*lvcite)->GetName() << G4endl;
        else
          fout << (*lvcite)->GetName() << "  " << (*lvcite)->GetMass(false, false) << G4endl;
      fout.close();
    }
    if (!filePV_.empty()) {
      const G4PhysicalVolumeStore *pvs = G4PhysicalVolumeStore::GetInstance();
      std::ofstream fout(filePV_.c_str());
      for (std::vector<G4VPhysicalVolume *>::const_iterator pvcite = pvs->begin(); pvcite != pvs->end(); pvcite++) {
        if (fileDetail_) {
          if (dd4hep_)
            fout << (*pvcite)->GetName() << " " << (*pvcite)->GetTranslation().x() << " "
                 << (*pvcite)->GetTranslation().y() << " " << (*pvcite)->GetTranslation().z() << G4endl;
          else
            fout << (*pvcite)->GetName() << " " << (*pvcite)->GetCopyNo() << " " << (*pvcite)->GetTranslation().x()
                 << " " << (*pvcite)->GetTranslation().y() << " " << (*pvcite)->GetTranslation().z() << G4endl;
        } else {
          if (dd4hep_)
            fout << (*pvcite)->GetName() << G4endl;
          else
            fout << (*pvcite)->GetName() << " " << (*pvcite)->GetCopyNo() << G4endl;
        }
      }
      fout.close();
    }
    if (!fileTouch_.empty()) {
      std::ofstream fout(fileTouch_.c_str());
      std::vector<std::string> touches;
      getTouch(theTopPV_, 0, 1, touches);
      std::sort(touches.begin(), touches.end());
      for (const auto &touch : touches)
        fout << touch << G4endl;
      fout.close();
    }
    if (!fileRegion_.empty()) {
      const G4RegionStore *regs = G4RegionStore::GetInstance();
      std::ofstream fout(fileRegion_.c_str());
      for (std::vector<G4Region *>::const_iterator regite = regs->begin(); regite != regs->end(); regite++) {
        if (!fileDetail_)
          fout << (*regite)->GetName() << G4endl;
        else
          fout << (*regite)->GetName() << " " << (*regite)->GetNumberOfMaterials() << " "
               << (*regite)->GetNumberOfRootVolumes() << G4endl;
      }
      fout.close();
    }
  }
}

void PrintGeomInfoAction::getTouch(G4VPhysicalVolume *pv,
                                   unsigned int leafDepth,
                                   unsigned int copym,
                                   std::vector<std::string> &touches) {
  if (leafDepth == 0)
    fHistory_.SetFirstEntry(pv);
  else
    fHistory_.NewLevel(pv, kNormal, pv->GetCopyNo());

  std::string mother = "World";
  if (pv->GetMotherLogical())
    mother = static_cast<std::string>(dd4hep::dd::noNamespace(pv->GetMotherLogical()->GetName()));

  G4LogicalVolume *lv = pv->GetLogicalVolume();
  std::string lvname = static_cast<std::string>(dd4hep::dd::noNamespace(lv->GetName()));
  unsigned int copy = static_cast<unsigned int>(pv->GetCopyNo());

  std::string name = lvname + ":" + std::to_string(copy) + "_" + mother + ":" + std::to_string(copym);
  touches.emplace_back(name);

  int NoDaughters = lv->GetNoDaughters();
  while ((NoDaughters--) > 0) {
    G4VPhysicalVolume *pvD = lv->GetDaughter(NoDaughters);
    if (!pvD->IsReplicated())
      getTouch(pvD, leafDepth + 1, copy, touches);
  }

  if (leafDepth > 0)
    fHistory_.BackLevel();
}

std::string PrintGeomInfoAction::spacesFromLeafDepth(unsigned int leafDepth) {
  std::string spaces;
  unsigned int ii;
  for (ii = 0; ii < leafDepth; ii++) {
    spaces += "  ";
  }
  return spaces;
}

G4VPhysicalVolume *PrintGeomInfoAction::getTopPV() {
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}

G4LogicalVolume *PrintGeomInfoAction::getTopLV() { return theTopPV_->GetLogicalVolume(); }

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(PrintGeomInfoAction);
