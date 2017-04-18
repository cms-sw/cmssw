#include "SimG4Core/PrintGeomInfo/interface/PrintGeomSummary.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
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

PrintGeomSummary::PrintGeomSummary(const edm::ParameterSet &p) : theTopPV_(0) {
  std::vector<std::string> defNames;
  nodeNames_  = p.getUntrackedParameter<std::vector<std::string> >("NodeNames",defNames);
  G4cout << "PrintGeomSummary:: initialised for " << nodeNames_.size()
	 << " nodes:" << G4endl;
  for (unsigned int ii=0; ii<nodeNames_.size(); ii++)
    G4cout << "Node[" << ii << "] : " << nodeNames_[ii] << G4endl;

  solidShape_[ddbox]            = "Box";
  solidShape_[ddtubs]           = "Tube";
  solidShape_[ddtrap]           = "Trapezoid";
  solidShape_[ddcons]           = "Cone";
  solidShape_[ddpolycone_rz]    = "Polycone_rz";
  solidShape_[ddpolyhedra_rz]   = "Polyhedra_rz";
  solidShape_[ddpolycone_rrz]   = "Polycone_rrz";
  solidShape_[ddpolyhedra_rrz]  = "Polyhedra_rrz";
  solidShape_[ddtorus]          = "Torus";
  solidShape_[ddunion]          = "UnionSolid";
  solidShape_[ddsubtraction]    = "SubtractionSolid"; 
  solidShape_[ddintersection]   = "IntersectionSolid";
  solidShape_[ddreflected]      = "ReflectedSolid";
  solidShape_[ddshapeless]      = "ShapelessSolid";
  solidShape_[ddpseudotrap]     = "PseudoTrapezoid";
  solidShape_[ddtrunctubs]      = "TruncatedTube";
  solidShape_[ddsphere]         = "Sphere"; 
  solidShape_[ddorb]            = "Orb"; 
  solidShape_[ddellipticaltube] = "EllipticalTube";
  solidShape_[ddellipsoid]      = "Ellipsoid";
  solidShape_[ddparallelepiped] = "Parallelepiped";
  solidShape_[ddcuttubs]        = "CutTubs";
  solidShape_[dd_not_init]      = "Unknown";
}
 
PrintGeomSummary::~PrintGeomSummary() {}
 
void PrintGeomSummary::update(const BeginOfJob * job) {

  edm::ESTransientHandle<DDCompactView> pDD;
  (*job)()->get<IdealGeometryRecord>().get(pDD);
  const DDCompactView* cpv = &(*pDD);

  const DDCompactView::graph_type & gra = cpv->graph();
  DDCompactView::graph_type::index_type i=0;
  solidMap_.clear();
  for (DDCompactView::graph_type::const_adj_iterator git=gra.begin();
       git != gra.end(); ++git) {
    const DDLogicalPart & ddLP = gra.nodeData(git);
    addSolid(ddLP);
    ++i;
    if (git->size()) {
      // ask for children of ddLP  
      for (DDCompactView::graph_type::edge_list::const_iterator cit  = git->begin();
	   cit != git->end(); ++cit) {
	const DDLogicalPart & ddcurLP = gra.nodeData(cit->first);
	addSolid(ddcurLP);
      }
    }
  }
  G4cout << "Finds " << solidMap_.size() << " different solids in the tree" 
	 << G4endl;
}

void PrintGeomSummary::addSolid(const DDLogicalPart & part) {
  const DDSolid& solid = part.solid();
  std::map<DDSolidShape,std::string>::iterator it = solidShape_.find(solid.shape());
  std::string name = solid.name().name();
  if (it == solidShape_.end()) solidMap_[name] = dd_not_init;
  else                         solidMap_[name] = it->first;
//G4cout << "Solid " << name << " is of shape " << solidMap_[name] << G4endl;
}  

void PrintGeomSummary::update(const BeginOfRun * run) {
  theTopPV_ = getTopPV();
  if (theTopPV_) {
    lvs_.clear(); sls_.clear(); touch_.clear();
    fillLV(theTopPV_->GetLogicalVolume()); 
    std::string name = theTopPV_->GetName();
    dumpSummary(G4cout,name);

    for (unsigned int k=0; k<nodeNames_.size(); ++k) {
      const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
      std::vector<G4LogicalVolume *>::const_iterator lvcite;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
	if ((*lvcite)->GetName() == (G4String)(nodeNames_[k])) {
	  lvs_.clear(); sls_.clear(); touch_.clear();
	  fillLV(*lvcite); 
	  dumpSummary(G4cout,nodeNames_[k]);
	}
      }
    }
  }
}

void PrintGeomSummary::fillLV(G4LogicalVolume * lv) {

  if (std::find(lvs_.begin(),lvs_.end(),lv) == lvs_.end()) lvs_.push_back(lv);
  G4VSolid* sl = lv->GetSolid();
  if (std::find(sls_.begin(),sls_.end(),sl) == sls_.end()) sls_.push_back(sl);
  touch_.push_back(lv);
  for(int ii = 0; ii < (int)(lv->GetNoDaughters()); ii++)
    fillLV(lv->GetDaughter(ii)->GetLogicalVolume());
}

void PrintGeomSummary::dumpSummary(std::ostream & out, std::string name) {

  //---------- Dump number of objects of each class
  out << G4endl << G4endl
      << "@@@@@@@@@@@@@@@@@@ Dumping Summary For Node " << name << G4endl;
  out << " Number of G4VSolid's: " << sls_.size() << G4endl;
  out << " Number of G4LogicalVolume's: " << lvs_.size() << G4endl;
  out << " Number of Touchable's: " << touch_.size() << G4endl;
  std::map<DDSolidShape,std::pair<int,int> > kount;
  int k(0);
  //First the solids
  out << G4endl << "Occurence of each type of shape among Solids" << G4endl;
  for (std::vector<G4VSolid*>::iterator it=sls_.begin(); it!=sls_.end(); ++it) {
    std::string name = (*it)->GetName();
    bool refl(false);
    if (name.find("_refl") < name.size()) {
      refl = true;
      name = name.substr(0,(name.find("_refl")));
    }
    std::map<std::string,DDSolidShape>::const_iterator jt=solidMap_.find(name);
    DDSolidShape shape =  (jt == solidMap_.end()) ? dd_not_init : jt->second;
    std::map<DDSolidShape,std::pair<int,int>>::iterator itr = kount.find(shape);
    if (itr == kount.end()) {
      kount[shape] = (refl) ? std::pair<int,int>(0,1) : std::pair<int,int>(1,0);
    } else {
      kount[shape] = (refl) ? 
	std::pair<int,int>(((itr->second).first),++((itr->second).second)) :
	std::pair<int,int>(++((itr->second).first),((itr->second).second));
    }
  }
  for (std::map<DDSolidShape,std::pair<int,int> >::iterator itr = kount.begin();
       itr != kount.end(); ++itr, ++k) {
    std::string shape = solidShape_[itr->first];
    out << "Shape [" << k << "]  " << shape << " # " << (itr->second).first
	<< " : " << (itr->second).second << G4endl;
  }
  k = 0; kount.clear();
  //Then the logical volumes
  out << G4endl << "Occurence of each type of shape among Logical Volumes" 
      << G4endl;
  for (std::vector<G4LogicalVolume*>::iterator it = lvs_.begin(); 
       it != lvs_.end(); ++it) {
    std::string name = ((*it)->GetSolid())->GetName();
    bool refl(false);
    if (name.find("_refl") < name.size()) {
      refl = true;
      name = name.substr(0,(name.find("_refl")));
    }
    std::map<std::string,DDSolidShape>::const_iterator jt=solidMap_.find(name);
    DDSolidShape shape =  (jt == solidMap_.end()) ? dd_not_init : jt->second;
    std::map<DDSolidShape,std::pair<int,int>>::iterator itr = kount.find(shape);
    if (itr == kount.end()) {
      kount[shape] = (refl) ? std::pair<int,int>(0,1) : std::pair<int,int>(1,0);
    } else {
      kount[shape] = (refl) ? 
	std::pair<int,int>(((itr->second).first),++((itr->second).second)) :
	std::pair<int,int>(++((itr->second).first),((itr->second).second));
    }
  }
  for (std::map<DDSolidShape,std::pair<int,int> >::iterator itr = kount.begin();
       itr != kount.end(); ++itr, ++k) {
    std::string shape = solidShape_[itr->first];
    out << "Shape [" << k << "]  " << shape << " # " << (itr->second).first
	<< " : " << (itr->second).second << G4endl;
  }
  k = 0; kount.clear();
  //Finally the touchables
  out << G4endl << "Occurence of each type of shape among Touchables" 
      << G4endl;
  for (std::vector<G4LogicalVolume*>::iterator it = touch_.begin(); 
       it != touch_.end(); ++it) {
    std::string name = ((*it)->GetSolid())->GetName();
    bool refl(false);
    if (name.find("_refl") < name.size()) {
      refl = true;
      name = name.substr(0,(name.find("_refl")));
    }
    std::map<std::string,DDSolidShape>::const_iterator jt=solidMap_.find(name);
    DDSolidShape shape =  (jt == solidMap_.end()) ? dd_not_init : jt->second;
    std::map<DDSolidShape,std::pair<int,int>>::iterator itr = kount.find(shape);
    if (itr == kount.end()) {
      kount[shape] = (refl) ? std::pair<int,int>(0,1) : std::pair<int,int>(1,0);
    } else {
      kount[shape] = (refl) ? 
	std::pair<int,int>(((itr->second).first),++((itr->second).second)) :
	std::pair<int,int>(++((itr->second).first),((itr->second).second));
    }
  }
  for (std::map<DDSolidShape,std::pair<int,int> >::iterator itr = kount.begin();
       itr != kount.end(); ++itr, ++k) {
    std::string shape = solidShape_[itr->first];
    out << "Shape [" << k << "]  " << shape << " # " << (itr->second).first
	<< " : " << (itr->second).second << G4endl;
  }
}

G4VPhysicalVolume * PrintGeomSummary::getTopPV() {
  
  return G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
}


