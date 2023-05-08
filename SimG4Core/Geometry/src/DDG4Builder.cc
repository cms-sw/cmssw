#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Geometry/interface/DDG4Builder.h"
#include "SimG4Core/Geometry/interface/DDG4SensitiveConverter.h"
#include "SimG4Core/Geometry/interface/DDG4SolidConverter.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"

#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4PVPlacement.hh"
#include "G4ReflectionFactory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VSolid.hh"

#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"

#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDG4Builder::DDG4Builder(const DDCompactView *cpv, G4LogicalVolumeToDDLogicalPartMap &lvmap, bool check)
    : solidConverter_(new DDG4SolidConverter), compactView_(cpv), map_(lvmap), check_(check) {
  theVectorOfDDG4Dispatchables_ = new DDG4DispContainer();
}

DDG4Builder::~DDG4Builder() { delete solidConverter_; }

G4LogicalVolume *DDG4Builder::convertLV(const DDLogicalPart &part) {
  edm::LogVerbatim("SimG4CoreGeometry") << "DDG4Builder::convertLV(): DDLogicalPart = " << part;
  G4LogicalVolume *result = logs_[part];
  if (!result) {
    G4VSolid *g4s = convertSolid(part.solid());
    G4Material *g4m = convertMaterial(part.material());
    result = new G4LogicalVolume(g4s, g4m, part.name().name());
    map_.insert(result, part);
    DDG4Dispatchable *disp = new DDG4Dispatchable(&part, result);
    theVectorOfDDG4Dispatchables_->push_back(disp);
    edm::LogVerbatim("SimG4CoreGeometry")
        << "DDG4Builder::convertLV(): new G4LogicalVolume " << part.name().name()
        << "\nDDG4Builder: newEvent: dd=" << part.ddname() << " g4=" << result->GetName();
    logs_[part] = result;  // DDD -> GEANT4
  }
  return result;
}

G4VSolid *DDG4Builder::convertSolid(const DDSolid &solid) {
  G4VSolid *result = sols_[solid];
  if (!result) {
    result = solidConverter_->convert(solid);
    sols_[solid] = result;
  }
  return result;
}

G4Material *DDG4Builder::convertMaterial(const DDMaterial &material) {
  edm::LogVerbatim("SimG4CoreGeometry") << "DDDetConstr::ConvertMaterial: material=" << material;
  G4Material *result = nullptr;
  if (material) {
    // only if it's a valid DDD-material
    if ((result = mats_[material])) {
      edm::LogVerbatim("SimG4CoreGeometry") << "  is already converted";
      return result;
    }
  } else {
    // only if it's NOT a valid DDD-material
    throw cms::Exception("SimG4CoreGeometry",
                         " material is not valid from the Detector Description: " + material.toString());
  }
  int c = 0;
  if ((c = material.noOfConstituents())) {
    // it's a composite material
    edm::LogVerbatim("SimG4CoreGeometry")
        << "  creating a G4-composite material. c=" << c << " d=" << material.density() / CLHEP::g * CLHEP::mole;
    result = new G4Material(material.name().name(), material.density(), c);
    for (int i = 0; i < c; ++i) {
      // recursive building of constituents
      edm::LogVerbatim("SimG4CoreGeometry")
          << "  adding the composite=" << material.name() << " fm=" << material.constituent(i).second;
      result->AddMaterial(convertMaterial(material.constituent(i).first),
                          material.constituent(i).second);  // fractionmass
    }
  } else {
    // it's an elementary material
    edm::LogVerbatim("SimG4CoreGeometry") << "  building an elementary material"
                                          << " z=" << material.z() << " a=" << material.a() / CLHEP::g * CLHEP::mole
                                          << " d=" << material.density() / CLHEP::g * CLHEP::cm3;
    result = new G4Material(material.name().name(), material.z(), material.a(), material.density());
  }
  mats_[material] = result;
  return result;
}

G4LogicalVolume *DDG4Builder::BuildGeometry(SensitiveDetectorCatalog &catalog) {
  G4ReflectionFactory *refFact = G4ReflectionFactory::Instance();
  refFact->SetScalePrecision(100. * refFact->GetScalePrecision());

  using Graph = DDCompactView::Graph;
  const auto &gra = compactView_->graph();
  using adjl_iterator = Graph::const_adj_iterator;
  adjl_iterator git = gra.begin();
  adjl_iterator gend = gra.end();

  for (; git != gend; ++git) {
    const DDLogicalPart &ddLP = gra.nodeData(git);
    if (!(ddLP.isDefined().second)) {
      throw cms::Exception("SimG4CoreGeometry",
                           " DDG4Builder::BuildGeometry() has encountered an "
                           "undefined DDLogicalPart named " +
                               ddLP.toString());
    }
    G4LogicalVolume *g4LV = convertLV(ddLP);
    if (!git->empty()) {
      // ask for children of ddLP
      Graph::edge_list::const_iterator cit = git->begin();
      Graph::edge_list::const_iterator cend = git->end();
      for (; cit != cend; ++cit) {
        // fetch specific data
        const DDLogicalPart &ddcurLP = gra.nodeData(cit->first);
        if (!ddcurLP.isDefined().second) {
          std::string err = " DDG4Builder::BuildGeometry() in processing \"children\" has ";
          err += "encountered an undefined DDLogicalPart named " + ddcurLP.toString() + " is a child of " +
                 ddLP.toString();
          throw cms::Exception("SimG4CoreGeometry", err);
        }
        int offset = getInt("CopyNoOffset", ddcurLP);
        int tag = getInt("CopyNoTag", ddcurLP);
        DDRotationMatrix rm(gra.edgeData(cit->second)->rot());
        DD3Vector x, y, z;
        rm.GetComponents(x, y, z);
        if ((x.Cross(y)).Dot(z) < 0)
          edm::LogVerbatim("SimG4CoreGeometry")
              << "DDG4Builder: Reflection: " << gra.edgeData(cit->second)->ddrot()
              << ">>Placement d=" << gra.nodeData(cit->first).ddname() << " m=" << ddLP.ddname()
              << " cp=" << gra.edgeData(cit->second)->copyno() << " r=" << gra.edgeData(cit->second)->ddrot().ddname();
        G4ThreeVector tempTran(gra.edgeData(cit->second)->trans().X(),
                               gra.edgeData(cit->second)->trans().Y(),
                               gra.edgeData(cit->second)->trans().Z());
        G4Translate3D transl = tempTran;
        CLHEP::HepRep3x3 temp(x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z());  // matrix
        CLHEP::HepRotation hr(temp);
        edm::LogVerbatim("SimG4CoreGeometry")
            << "Position " << gra.nodeData(cit->first).name().name() << ":"
            << gra.edgeData(cit->second)->copyno() + offset + tag << " in " << g4LV->GetName() << " at " << tempTran
            << " with rotation matrix (" << x.X() << ", " << x.Y() << ", " << x.Z() << ", " << y.X() << ", " << y.Y()
            << ", " << y.Z() << ", " << z.X() << ", " << z.Y() << ", " << z.Z() << ")";

        // G3 convention of defining rot-matrices ...
        G4Transform3D trfrm = transl * G4Rotate3D(hr.inverse());  //.inverse();

        refFact->Place(trfrm,  // transformation containing a possible reflection
                       gra.nodeData(cit->first).name().name(),
                       convertLV(gra.nodeData(cit->first)),                 // daugther
                       g4LV,                                                // mother
                       false,                                               // 'ONLY'
                       gra.edgeData(cit->second)->copyno() + offset + tag,  // copy number
                       check_);
      }  // iterate over children
    }    // if (children)
  }      // iterate over graph nodes

  // Looking for in the G4ReflectionFactory secretly created reflected
  // G4LogicalVolumes
  std::map<DDLogicalPart, G4LogicalVolume *>::const_iterator ddg4_it = logs_.begin();
  for (; ddg4_it != logs_.end(); ++ddg4_it) {
    G4LogicalVolume *reflLogicalVolume = refFact->GetReflectedLV(ddg4_it->second);
    if (reflLogicalVolume) {
      DDLogicalPart ddlv = ddg4_it->first;
      map_.insert(reflLogicalVolume, ddlv);
      DDG4Dispatchable *disp = new DDG4Dispatchable(&(ddg4_it->first), reflLogicalVolume);
      theVectorOfDDG4Dispatchables_->push_back(disp);
      edm::LogVerbatim("SimG4CoreGeometry")
          << "DDG4Builder: dd=" << ddlv.ddname() << " g4=" << reflLogicalVolume->GetName();
    }
  }

  G4LogicalVolume *world = logs_[compactView_->root()];

  //
  //  needed for building sensitive detectors
  //
  DDG4SensitiveConverter conv;
  conv.upDate(*theVectorOfDDG4Dispatchables_, catalog);

  return world;
}

int DDG4Builder::getInt(const std::string &ss, const DDLogicalPart &part) {
  DDValue val(ss);
  std::vector<const DDsvalues_type *> result = part.specifics();
  bool foundIt = false;
  for (auto stype : result) {
    foundIt = DDfetch(stype, val);
    if (foundIt)
      break;
  }
  if (foundIt) {
    std::vector<double> temp = val.doubles();
    if (temp.size() != 1) {
      throw cms::Exception("SimG4CoreGeometry",
                           " DDG4Builder::getInt() Problem with Region tags - "
                           "one and only one allowed: " +
                               ss);
    }
    return int(temp[0]);
  } else
    return 0;
}

double DDG4Builder::getDouble(const std::string &ss, const DDLogicalPart &part) {
  DDValue val(ss);
  std::vector<const DDsvalues_type *> result = part.specifics();
  bool foundIt = false;
  for (auto stype : result) {
    foundIt = DDfetch(stype, val);
    if (foundIt)
      break;
  }
  if (foundIt) {
    std::vector<std::string> temp = val.strings();
    if (temp.size() != 1) {
      throw cms::Exception("SimG4CoreGeometry",
                           " DDG4Builder::getDouble() Problem with Region tags "
                           "- one and only one allowed: " +
                               ss);
    }
    double v;
    std::string unit;
    std::istringstream is(temp[0]);
    is >> v >> unit;
    v = v * G4UnitDefinition::GetValueOf(unit.substr(1, unit.size()));
    return v;
  } else
    return 0;
}
