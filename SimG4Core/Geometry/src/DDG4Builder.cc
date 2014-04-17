#include "DetectorDescription/Core/interface/DDSpecifics.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Geometry/interface/DDG4Builder.h"
#include "SimG4Core/Geometry/interface/DDG4SolidConverter.h"
#include "SimG4Core/Geometry/interface/DDG4SensitiveConverter.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "G4VSolid.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4Trap.hh"
#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4ReflectionFactory.hh"

#include "G4UnitsTable.hh"

#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDG4DispContainer * DDG4Builder::theVectorOfDDG4Dispatchables_ = 0;

DDG4DispContainer * DDG4Builder::theVectorOfDDG4Dispatchables() { 
  return theVectorOfDDG4Dispatchables_; 
}

DDG4Builder::DDG4Builder(const DDCompactView* cpv, bool check) : 
  solidConverter_(new DDG4SolidConverter), compactView(cpv), check_(check) { 
  theVectorOfDDG4Dispatchables_ = new DDG4DispContainer(); 
}

DDG4Builder::~DDG4Builder() { 
  delete solidConverter_; 
}

G4LogicalVolume * DDG4Builder::convertLV(const DDLogicalPart & part) {
  LogDebug("SimG4CoreGeometry") << "DDG4Builder::convertLV(): DDLogicalPart = " << part;
  G4LogicalVolume * result = logs_[part];
  if (!result) {
    G4VSolid * s   = convertSolid(part.solid());
    G4Material * m = convertMaterial(part.material());
    result = new G4LogicalVolume(s,m,part.name().name());
    map_.insert(result,part);
    DDG4Dispatchable * disp = new DDG4Dispatchable(&part,result);	
    theVectorOfDDG4Dispatchables_->push_back(disp);
    LogDebug("SimG4CoreGeometry") << "DDG4Builder::convertLV(): new G4LogicalVolume " << part.name().name()
				  << "\nDDG4Builder: newEvent: dd=" << part.ddname() << " g4=" << result->GetName()
				  << "\nSolid " << s->GetName() << "  Material " << m->GetName();
    logs_[part] = result;  // DDD -> GEANT4  
  }
  return result;
}

G4VSolid * DDG4Builder::convertSolid(const DDSolid & solid) {
  G4VSolid * result = sols_[solid];
  if (!result) { 
    result = solidConverter_->convert(solid); sols_[solid] = result; 
  }
  return result;
}

G4Material * DDG4Builder::convertMaterial(const DDMaterial & material) {
  LogDebug("SimG4CoreGeometry") << "DDDetConstr::ConvertMaterial: material=" << material;
  G4Material * result = 0;
  if (material) {
    // only if it's a valid DDD-material
    if ((result = mats_[material])) {
      LogDebug("SimG4CoreGeometry") << "  is already converted"; 
      return result; }
  } else {
    // only if it's NOT a valid DDD-material
    edm::LogError("SimG4CoreGeometry") << "DDG4Builder::  material " << material.toString() << " is not valid (in the DDD sense!)";
    throw cms::Exception("SimG4CoreGeometry", " material is not valid from the Detector Description: " + material.toString()); 
  }    
  int c = 0;
  if ((c = material.noOfConstituents())) {
    // it's a composite material
    LogDebug("SimG4CoreGeometry") << "  creating a G4-composite material. c=" << c
				  << " d=" << material.density()/g*mole ;
    result = new G4Material(material.name().name(),material.density(),c);
    for (int i=0 ; i<c; ++i) {
      // recursive building of constituents
      LogDebug("SimG4CoreGeometry") << "  adding the composite=" << material.name()
				    << " fm=" << material.constituent(i).second;
      result->AddMaterial
	(convertMaterial(material.constituent(i).first),
	 material.constituent(i).second);// fractionmass
    }
  } else {
    // it's an elementary material
    LogDebug("SimG4CoreGeometry") << "  building an elementary material"
				  << " z=" << material.z()
				  << " a=" << material.a()/g*mole
				  << " d=" << material.density()/g*cm3 ;
    result = new G4Material
      (material.name().name(),material.z(),material.a(),material.density());
  }
  mats_[material] = result;
  return result;
} 

DDGeometryReturnType DDG4Builder::BuildGeometry() {
  G4ReflectionFactory * refFact = G4ReflectionFactory::Instance();
  refFact->SetScalePrecision(100.*refFact->GetScalePrecision());

  typedef DDCompactView::graph_type graph_type;
  const graph_type & gra = compactView->graph();
  typedef graph_type::const_adj_iterator adjl_iterator;
  adjl_iterator git = gra.begin();
  adjl_iterator gend = gra.end();    

  graph_type::index_type i=0;
  for (; git != gend; ++git) {
    const DDLogicalPart & ddLP = gra.nodeData(git);
    if ( !(ddLP.isDefined().second) ) {
      edm::LogError("SimG4CoreGeometry") << "DDG4Builder::BuildGeometry() has encountered an undefined DDLogicalPart named " << ddLP.toString();
      throw cms::Exception("SimG4CoreGeometry", " DDG4Builder::BuildGeometry() has encountered an undefined DDLogicalPart named " + ddLP.toString());
    }
    G4LogicalVolume * g4LV = convertLV(ddLP);
    ++i;	
    if (git->size()) {
      // ask for children of ddLP  
      graph_type::edge_list::const_iterator cit  = git->begin();
      graph_type::edge_list::const_iterator cend = git->end();
      for (; cit != cend; ++cit) {
	// fetch specific data
	const DDLogicalPart & ddcurLP = gra.nodeData(cit->first);
	if ( !ddcurLP.isDefined().second ) {
	  std::string err = " DDG4Builder::BuildGeometry() in processing \"children\" has ";
	  err += "encountered an undefined DDLogicalPart named " + ddLP.toString();
	  edm::LogError("SimG4CoreGeometry") << err;
	  throw cms::Exception("SimG4CoreGeometry", err) ;
	}
	int offset = getInt("CopyNoOffset",ddcurLP);
	int tag = getInt("CopyNoTag",ddcurLP);				
	DDRotationMatrix rm(gra.edgeData(cit->second)->rot());
	DD3Vector x, y, z;
	rm.GetComponents(x, y, z);
	if ((x.Cross(y)).Dot(z)<0)
	  LogDebug("SimG4CoreGeometry") << ">>Reflection encountered: " << gra.edgeData(cit->second)->rot_ ;
	G4ThreeVector tempTran(gra.edgeData(cit->second)->trans_.X(), gra.edgeData(cit->second)->trans_.Y(), gra.edgeData(cit->second)->trans_.Z());
	G4Translate3D transl = tempTran;
	CLHEP::HepRep3x3 temp( x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z() ); //matrix representation
	CLHEP::HepRotation hr ( temp );
	LogDebug("SimG4CoreGeometry") << ">>Placement d=" << gra.nodeData(cit->first).ddname() 
				      << " m=" << ddLP.ddname() << " cp=" << gra.edgeData(cit->second)->copyno_
				      << " r=" << gra.edgeData(cit->second)->rot_.ddname() 
				      << " t=" << tempTran;          
	    
	// G3 convention of defining rot-matrices ...
	G4Transform3D trfrm  = transl * G4Rotate3D(hr.inverse());//.inverse();

	refFact->Place(trfrm, // transformation containing a possible reflection
		       gra.nodeData(cit->first).name().name(),
		       convertLV(gra.nodeData(cit->first)), 		// daugther
		       g4LV, 				 		// mother
		       false,                 		 		// 'ONLY'
		       gra.edgeData(cit->second)->copyno_+offset+tag, 	// copy number
		       check_);
      } // iterate over children
    } // if (children)
  } // iterate over graph nodes  
    
  // Looking for in the G4ReflectionFactory secretly created reflected G4LogicalVolumes
  std::map<DDLogicalPart, G4LogicalVolume*>::const_iterator  ddg4_it = logs_.begin();
  for (; ddg4_it != logs_.end(); ++ddg4_it) {
    G4LogicalVolume * reflLogicalVolume = refFact->GetReflectedLV(ddg4_it->second);
    if (reflLogicalVolume) {
      DDLogicalPart ddlv = ddg4_it->first;
      map_.insert(reflLogicalVolume,ddlv);
      DDG4Dispatchable * disp = new DDG4Dispatchable(&(ddg4_it->first),reflLogicalVolume);
      theVectorOfDDG4Dispatchables_->push_back(disp);
      LogDebug("SimG4CoreGeometry") << "DDG4Builder: newEvent: dd=" 
				    << ddlv.ddname() << " g4=" 
				    << reflLogicalVolume->GetName();
    }  
  }
      
  G4LogicalVolume * world = logs_[compactView->root()];

  //
  //  needed for building sensitive detectors
  //
  DDG4SensitiveConverter conv_;
  SensitiveDetectorCatalog catalog = conv_.upDate(*theVectorOfDDG4Dispatchables_);

  return DDGeometryReturnType(world,map_,catalog);    
}

int DDG4Builder::getInt(const std::string & s, const DDLogicalPart & part)
{
  DDValue val(s);
  std::vector<const DDsvalues_type *> result = part.specifics();
  std::vector<const DDsvalues_type *>::iterator it = result.begin();
  bool foundIt = false;
  for (; it != result.end(); ++it) {
    foundIt = DDfetch(*it,val);
    if (foundIt) break;
  }    
  if (foundIt) { 
    std::vector<double> temp = val.doubles();
    if (temp.size() != 1) {
      edm::LogError("SimG4CoreGeometry") << " DDG4Builder - ERROR: I need only 1 " << s ;
      throw SimG4Exception("DDG4Builder: Problem with Region tags - one and only one allowed");
    }      
    return int(temp[0]);
  }
  else return 0;
}

double DDG4Builder::getDouble(const std::string & s, 
			      const DDLogicalPart & part) {
  DDValue val(s);
  std::vector<const DDsvalues_type *> result = part.specifics();
  std::vector<const DDsvalues_type *>::iterator it = result.begin();
  bool foundIt = false;
  for (; it != result.end(); ++it) {
    foundIt = DDfetch(*it,val);
    if (foundIt) break;
  }    
  if (foundIt) { 
    std::vector<std::string> temp = val.strings();
    if (temp.size() != 1) {
      edm::LogError("SimG4CoreGeometry") << " DDG4Builder - ERROR: I need only 1 " << s ;
      throw SimG4Exception("DDG4Builder: Problem with Region tags - one and only one allowed");
    }
    double v;
    std::string unit;
    std::istringstream is(temp[0].c_str());
    is >> v >> unit;
    v  = v*G4UnitDefinition::GetValueOf(unit.substr(1,unit.size()));
    return v;
  }
  else return 0;
}
