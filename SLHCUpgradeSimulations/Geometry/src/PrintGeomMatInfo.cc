#include "SLHCUpgradeSimulations/Geometry/interface/PrintGeomMatInfo.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
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

PrintGeomMatInfo::PrintGeomMatInfo(const edm::ParameterSet &p)
{
    _dumpSummary = p.getUntrackedParameter<bool>("DumpSummary", true);
    _dumpLVTree  = p.getUntrackedParameter<bool>("DumpLVTree",  true);
    _dumpLVMatBudget = p.getUntrackedParameter<bool>("DumpLVMatBudget", false);
    _lvNames2Dump= p.getUntrackedParameter<std::vector<std::string> >("LVNames2Dump");
    _level2Dump = 0;
    _dumpIndex = 0;
    _dumpIt = false;
    _radiusLayer = p.getUntrackedParameter<std::vector<double> >("Radius2Use");
    _zLayer      = p.getUntrackedParameter<std::vector<double> >("Z2Use");
    _maxLevelsCounted = 50;
    _countsPerLevel.assign(_maxLevelsCounted,0);
    _dumpMaterial= p.getUntrackedParameter<bool>("DumpMaterial",false);
    _dumpLVList  = p.getUntrackedParameter<bool>("DumpLVList",  false);
    _dumpLV      = p.getUntrackedParameter<bool>("DumpLV",      false);
    _dumpSolid   = p.getUntrackedParameter<bool>("DumpSolid",   false);
    _dumpAtts    = p.getUntrackedParameter<bool>("DumpAttributes", false);
    _dumpPV      = p.getUntrackedParameter<bool>("DumpPV",      false);
    _dumpRotation= p.getUntrackedParameter<bool>("DumpRotation",false);
    _dumpReplica = p.getUntrackedParameter<bool>("DumpReplica", false);
    _dumpTouch   = p.getUntrackedParameter<bool>("DumpTouch",   false);
    _dumpSense   = p.getUntrackedParameter<bool>("DumpSense",   false);
    name  = p.getUntrackedParameter<std::string>("Name","*");
    nchar = name.find("*");
    name.assign(name,0,nchar);
    names = p.getUntrackedParameter<std::vector<std::string> >("Names");
    std::cout << "size of _lvNames2Dump = " << _lvNames2Dump.size()
              << " size of _radiusLayer = " << _radiusLayer.size()
              << " size of _zLayer = " << _zLayer.size() << std::endl;
    std::cout << "PrintGeomMatInfo:: initialised with verbosity levels:"
	      << " Summary   " << _dumpSummary << " LVTree   " << _dumpLVTree
	      << " LVList    " << _dumpLVList  << " Material " << _dumpMaterial
	      << "\n                                                        "
	      << " LVMatBudget   " << _dumpLVMatBudget << " for";
    _areaLayer.reserve(_lvNames2Dump.size());
    if(_lvNames2Dump.size() == _radiusLayer.size() &&
       _lvNames2Dump.size() == _zLayer.size()) {
       for (unsigned int i=0; i<_lvNames2Dump.size(); i++) {
          _areaLayer[i] = 2.0*3.14159*_radiusLayer[i]*_zLayer[i];
          std::cout << "\n                                                        "
                    << " " << _lvNames2Dump[i] << " radius = " << _radiusLayer[i]
                    << " z = " << _zLayer[i] << " area = " << _areaLayer[i];
       }
    } else {
          _areaLayer.assign(3,0.0);
          std::cout << "\n                                                        "
                    << " Problem with unequal sizes!! Fix and rerun";
    }
    std::cout << "\n                                                        "
              << " and max levels for count = " << _maxLevelsCounted;
    std::cout << "\n                                                        "
              << " LV        " << _dumpLV      << " Solid    " << _dumpSolid 
	      << " Attribs   " << _dumpAtts
	      << "\n                                                        "
	      << " PV        " << _dumpPV      << " Rotation " << _dumpRotation
	      << " Replica   " << _dumpReplica
	      << "\n                                                        "
	      << " Touchable " << _dumpTouch << " for names (0-" << nchar 
	      << ") = " << name 
	      << "\n                                                        "
	      << " Sensitive " << _dumpSense << " for " << names.size()
	      << " names";
    for (unsigned int i=0; i<names.size(); i++) std::cout << " " << names[i];
    std::cout << std::endl;
}
 
PrintGeomMatInfo::~PrintGeomMatInfo() {}
 
void PrintGeomMatInfo::update(const BeginOfJob * job)
{
    if (_dumpSense) {
        edm::ESHandle<DDCompactView> pDD;
	(*job)()->get<IdealGeometryRecord>().get(pDD);

	std::cout << "PrintGeomMatInfo::Get Printout of Sensitive Volumes " 
		  << "for " << names.size() << " Readout Units" << std::endl;
	for (unsigned int i=0; i<names.size(); i++) {
	    std::string attribute = "ReadOutName";
	    std::string sd        = names[i];
	    DDSpecificsFilter filter;
	    DDValue           ddv(attribute,sd,0);
	    filter.setCriteria(ddv,DDSpecificsFilter::equals);
	    DDFilteredView fv(*pDD);
	    std::cout << "PrintGeomMatInfo:: Get Filtered view for " 
		      << attribute << " = " << sd << std::endl;
	    fv.addFilter(filter);
	    bool dodet = fv.firstChild();
 
	    std::string spaces = spacesFromLeafDepth(1);

	    while (dodet) {
	        const DDLogicalPart & log = fv.logicalPart();
		std::string lvname = DDSplit(log.name()).first;
		DDTranslation tran = fv.translation();
		std::vector<int> copy = fv.copyNumbers();

		unsigned int leafDepth = copy.size();
		std::cout << leafDepth << spaces << "### VOLUME = " << lvname 
			  << " Copy No";
		for (int k=leafDepth-1; k>=0; k--) std::cout << " " << copy[k];
		std::cout << " Centre at " << tran << " (r = " << tran.Rho()
			  << ", phi = " << tran.phi()/deg << ")" << std::endl;
		dodet = fv.next();
	    }
 	}
    }
}
  
void PrintGeomMatInfo::update(const BeginOfRun * run)
{
    theTopPV = getTopPV();

    if (_dumpSummary)  dumpSummary(std::cout);
    if (_dumpLVTree)   dumpG4LVTree(std::cout);
    if (_dumpLVMatBudget)   dumpG4LVMatBudget(std::cout);
    
    //---------- Dump list of objects of each class with detail of parameters
    if (_dumpMaterial) dumpMaterialList(std::cout);
    if (_dumpLVList)   dumpG4LVList(std::cout);

    //---------- Dump LV and PV information
    if (_dumpLV || _dumpPV || _dumpTouch) dumpHierarchyTreePVLV(std::cout);
}

void PrintGeomMatInfo::dumpSummary(std::ostream & out)
{
    //---------- Dump number of objects of each class
    out << " @@@@@@@@@@@@@@@@@@ Dumping G4 geometry objects Summary " << std::endl;
    if(theTopPV == 0) 
    {
	out << " No volume created " << std::endl;
	return;
    }
    out << " @@@ Geometry built inside world volume: " << theTopPV->GetName() << std::endl;
    // Get number of solids (< # LV if several LV share a solid)
    const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
    std::vector<G4LogicalVolume *>::const_iterator lvcite;
    std::set<G4VSolid *> theSolids;
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
	theSolids.insert((*lvcite)->GetSolid());
    out << " Number of G4VSolid's: " << theSolids.size() << std::endl;
    out << " Number of G4LogicalVolume's: " << lvs->size() << std::endl;
    const G4PhysicalVolumeStore * pvs = G4PhysicalVolumeStore::GetInstance();
    out << " Number of G4VPhysicalVolume's: " << pvs->size() << std::endl;
    out << " Number of Touchable's: " << countNoTouchables() << std::endl;
    const G4MaterialTable * matTab = G4Material::GetMaterialTable();
    out << " Number of G4Material's: " << matTab->size() << std::endl;
}

void PrintGeomMatInfo::dumpG4LVList(std::ostream & out)
{
    out << " @@@@@@@@@@@@@@@@ DUMPING G4LogicalVolume's List  " << std::endl;
    const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
    std::vector<G4LogicalVolume*>::const_iterator lvcite;
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
      out << "LV:" << (*lvcite)->GetName() << "\tMaterial: " << (*lvcite)->GetMaterial()->GetName() << std::endl;
}

void PrintGeomMatInfo::dumpG4LVTree(std::ostream & out)
{
    out << " @@@@@@@@@@@@@@@@ DUMPING G4LogicalVolume's Tree  " << std::endl;
    G4LogicalVolume * lv = getTopLV(); 
    dumpG4LVLeaf(lv,0,1,out);
}

void PrintGeomMatInfo::dumpMaterialList(std::ostream & out)
{
    out << " @@@@@@@@@@@@@@@@ DUMPING G4Material List ";
    const G4MaterialTable * matTab = G4Material::GetMaterialTable();
    out << " with " << matTab->size() << " materials " << std::endl;
    std::vector<G4Material*>::const_iterator matite;
    for (matite = matTab->begin(); matite != matTab->end(); matite++)
	out << "Material: " << (*matite) << std::endl;
}

void PrintGeomMatInfo::dumpG4LVLeaf(G4LogicalVolume * lv, unsigned int leafDepth, unsigned int count, std::ostream & out)
{
    for (unsigned int ii=0; ii < leafDepth; ii++) out << "  ";
    out << " LV:(" << leafDepth << ") " << lv->GetName() << " (" << count
	<< ")" << std::endl;
    //--- If a volume is placed n types as daughter of this LV, it should only be counted once
    std::map<G4LogicalVolume*, unsigned int> lvCount;
    std::map<G4LogicalVolume*, unsigned int>::const_iterator cite;
    for (int ii = 0; ii < lv->GetNoDaughters(); ii++) {
	cite = lvCount.find(lv->GetDaughter(ii)->GetLogicalVolume());
	if (cite != lvCount.end()) lvCount[cite->first] = (cite->second) + 1;
	else lvCount.insert(std::pair< G4LogicalVolume*,unsigned int>(lv->GetDaughter(ii)->GetLogicalVolume(),1));
    }
    for (cite = lvCount.begin(); cite != lvCount.end(); cite++) 
	dumpG4LVLeaf((cite->first), leafDepth+1, (cite->second), out);
}

void PrintGeomMatInfo::dumpG4LVMatBudget(std::ostream & out)
{
    out << " @@@@@@@@@@@@@@@@ DUMPING G4LogicalVolume's Material Budget Tree  " << std::endl;
    G4LogicalVolume * lv = getTopLV(); 
    dumpG4LVLeafWithMat(lv,0,1,out);
}

void PrintGeomMatInfo::dumpG4LVLeafWithMat(G4LogicalVolume * lv, unsigned int leafDepth, unsigned int count, std::ostream & out)
{
    // switch off dumping at the next same level as the dump
    if(_dumpIt && _level2Dump == leafDepth) {
        _dumpIt = false;
//       std::cout << " stopped dumping " << std::endl;
    }
    // loop over volumes to dump: volumes cannot not overlap in hierarchy!!
    for (unsigned int i=0; i<_lvNames2Dump.size(); i++) {
       if(_lvNames2Dump[i].compare(lv->GetName()) == 0) {
          _dumpIt = true;
          _dumpIndex = i;
          _level2Dump = leafDepth;
//          std::cout << " start dumping " << _lvNames2Dump[i] << " at level " << _level2Dump << std::endl;
          break;
       }
    }

    if(_dumpIt) {
       if(leafDepth < _maxLevelsCounted) _countsPerLevel[leafDepth] = count;
       unsigned int total_multipler = 1;
       for (unsigned int ii=_level2Dump; ii <= leafDepth; ii++) total_multipler *= _countsPerLevel[ii];
       double thick = (lv->GetSolid()->GetCubicVolume() * total_multipler)/_areaLayer[_dumpIndex];
       for (unsigned int ii=0; ii < leafDepth; ii++) out << "  ";
       // print out Level, Logical volume name, volume in mm**3, material name, rad len of mat in mm
       // total number of volumes from dump level start, equivalent thickness when spread over
       // a cylinder of _radiusLayer and length _zLayer; equivalent thick in rad len
       out << " LV::" << leafDepth << ": " << lv->GetName() << " :" << count
           << ": " << lv->GetSolid()->GetName() << " :" << lv->GetSolid()->GetCubicVolume()
           << ": "<< lv->GetMaterial()->GetName() << " :" << lv->GetMaterial()->GetRadlen() 
           << ":" << " :" << total_multipler << ":" 
           << " thk :" << thick << ": x/X0 :" << thick/lv->GetMaterial()->GetRadlen()
           << ":" << " Kg : " << lv->GetMass()/kg << std::endl;
    } else {
       for (unsigned int ii=0; ii < leafDepth; ii++) out << "  ";
       out << " LV:(" << leafDepth << ") " << lv->GetName() << " (" << count << ")" << std::endl;
    }

    //--- If a volume is placed n types as daughter of this LV, it should only be counted once
    std::map<G4LogicalVolume*, unsigned int> lvCount;
    std::map<G4LogicalVolume*, unsigned int>::const_iterator cite;
    for (int ii = 0; ii < lv->GetNoDaughters(); ii++) {
	cite = lvCount.find(lv->GetDaughter(ii)->GetLogicalVolume());
	if (cite != lvCount.end()) lvCount[cite->first] = (cite->second) + 1;
	else lvCount.insert(std::pair< G4LogicalVolume*,unsigned int>(lv->GetDaughter(ii)->GetLogicalVolume(),1));
    }
    for (cite = lvCount.begin(); cite != lvCount.end(); cite++) 
	dumpG4LVLeafWithMat((cite->first), leafDepth+1, (cite->second), out);
}


int PrintGeomMatInfo::countNoTouchables()
{
    int nTouch = 0;
    G4LogicalVolume * lv = getTopLV();
    add1touchable(lv, nTouch);
    return nTouch;
}


void PrintGeomMatInfo::add1touchable(G4LogicalVolume * lv, int & nTouch)
{
    int siz = lv->GetNoDaughters();
    for(int ii = 0; ii < siz; ii++)
	add1touchable(lv->GetDaughter(ii)->GetLogicalVolume(), ++nTouch);
}
 
void PrintGeomMatInfo::dumpHierarchyTreePVLV(std::ostream & out) 
{
    //dumps in the following order: 
    //    1) a LV with details
    //    2) list of PVs daughters of this LV with details
    //    3) list of LVs daughters of this LV and for each go to 1)

    //----- Get top PV
    G4LogicalVolume*  topLV = getTopLV();
  
    //----- Dump this leaf (it will recursively dump all the tree)
    dumpHierarchyLeafPVLV(topLV, 0, out);
    dumpPV(theTopPV, 0, out);
  
    //----- Dump the touchables (it will recursively dump all the tree)
    if (_dumpTouch) dumpTouch(theTopPV, 0, out);
}

void PrintGeomMatInfo::dumpHierarchyLeafPVLV(G4LogicalVolume * lv, unsigned int leafDepth, std::ostream & out)
{
    //----- Dump this LV 
    dumpLV(lv, leafDepth, out);

    //----- Get LV daughters from list of PV daughters
    mmlvpv lvpvDaughters;
    std::set< G4LogicalVolume * > lvDaughters;
    int NoDaughters = lv->GetNoDaughters();
    while ((NoDaughters--)>0)
    {
	G4VPhysicalVolume * pvD = lv->GetDaughter(NoDaughters);
	lvpvDaughters.insert(mmlvpv::value_type(pvD->GetLogicalVolume(), pvD));
	lvDaughters.insert(pvD->GetLogicalVolume());
    }
 
    std::set< G4LogicalVolume * >::const_iterator scite;
    mmlvpv::const_iterator mmcite;

    //----- Dump daughters PV and LV
    for (scite = lvDaughters.begin(); scite != lvDaughters.end(); scite++) 
    {
	std::pair< mmlvpv::iterator, mmlvpv::iterator > mmER = lvpvDaughters.equal_range(*scite);    
	//----- Dump daughters PV of this LV
	for (mmcite = mmER.first ; mmcite != mmER.second; mmcite++) 
	    dumpPV((*mmcite).second, leafDepth+1, out);
	//----- Dump daughters LV
	dumpHierarchyLeafPVLV(*scite, leafDepth+1, out );
    }
}
 
void PrintGeomMatInfo::dumpLV(G4LogicalVolume * lv, unsigned int leafDepth, std::ostream & out)
{
    std::string spaces = spacesFromLeafDepth(leafDepth);

    //----- dump name 
    if (_dumpLV) { 
	out << leafDepth << spaces << "$$$ VOLUME = " << lv->GetName()
	    << "  Solid: " << lv->GetSolid()->GetName() << "  MATERIAL: "
	    << lv->GetMaterial()->GetName() << std::endl;
	if (_dumpSolid)
	  dumpSolid(lv->GetSolid(), leafDepth, out); //----- dump solid

    //----- dump LV info 
    //--- material 
	if (_dumpAtts) {
	  //--- Visualisation attributes
	  const G4VisAttributes * fVA = lv->GetVisAttributes();
	  if (fVA!=0) {
	    out <<  spaces << "  VISUALISATION ATTRIBUTES: " << std::endl;
	    out <<  spaces << "    IsVisible " << fVA->IsVisible() << std::endl;
	    out <<  spaces << "    IsDaughtersInvisible " << fVA->IsDaughtersInvisible() << std::endl;
	    out <<  spaces << "    Colour " << fVA->GetColour() << std::endl;
	    out <<  spaces << "    LineStyle " << fVA->GetLineStyle() << std::endl;
	    out <<  spaces << "    LineWidth " << fVA->GetLineWidth() << std::endl;
	    out <<  spaces << "    IsForceDrawingStyle " << fVA->IsForceDrawingStyle() << std::endl;
	    out <<  spaces << "    ForcedDrawingStyle " << fVA->GetForcedDrawingStyle() << std::endl;
	  }    
    
	  //--- User Limits
	  G4UserLimits * fUL = lv->GetUserLimits();
	  G4Track dummy;
	  if (fUL!=0) {
	    out <<  spaces << "    MaxAllowedStep " << fUL->GetMaxAllowedStep(dummy) << std::endl;
	    out <<  spaces << "    UserMaxTrackLength " << fUL->GetUserMaxTrackLength(dummy) << std::endl;
	    out <<  spaces << "    UserMaxTime " << fUL->GetUserMaxTime(dummy) << std::endl;
	    out <<  spaces << "    UserMinEkine " << fUL->GetUserMinEkine(dummy) << std::endl;
	    out <<  spaces << "    UserMinRange " << fUL->GetUserMinRange(dummy) << std::endl;
	  }
    
	  //--- other LV info
	  if (lv->GetSensitiveDetector()) 
	    out << spaces << "  IS SENSITIVE DETECTOR " << std::endl;
	  if (lv->GetFieldManager()) 
	    out << spaces << "  FIELD ON " << std::endl;

	  // Pointer (possibly NULL) to optimisation info objects.
	  out <<  spaces  
	      << "        Quality for optimisation, average number of voxels to be spent per content " 
	      << lv->GetSmartless() << std::endl;

	  // Pointer (possibly NULL) to G4FastSimulationManager object.
	  if (lv->GetFastSimulationManager()) 
	    out << spaces << "     Logical Volume is an envelope for a FastSimulationManager " 
		<< std::endl;
	  out << spaces << "     Weight used in the event biasing technique = " 
	      << lv->GetBiasWeight() << std::endl;
	} 
    }
}	

void PrintGeomMatInfo::dumpPV(G4VPhysicalVolume * pv, unsigned int leafDepth, std::ostream & out)
{
    std::string spaces = spacesFromLeafDepth(leafDepth);

    //----- PV info
    if (_dumpPV) 
    {
	std::string mother = "World";
	if (pv->GetMotherLogical()) mother = pv->GetMotherLogical()->GetName();
	out << leafDepth << spaces << "### VOLUME = " << pv->GetName() 
	    << " Copy No " << pv->GetCopyNo() << " in " << mother
	    << " at " << pv->GetTranslation();
    }
    if (!pv->IsReplicated()) 
    {
	if (_dumpPV) 
	{
	    if(pv->GetRotation() == 0) out << " with no rotation" << std::endl;
	    else  if(!_dumpRotation)   out << " with rotation" << std::endl; //just rotation name
	    else                       out << " with rotation " << *(pv->GetRotation()) << std::endl;
	}
    }  
    else 
    {
        if (_dumpReplica )
	{
	    out << spaces << "    It is replica: " << std::endl;
	    EAxis axis; 
	    int nReplicas; 
	    double width; 
	    double offset; 
	    bool consuming;
	    pv->GetReplicationData(axis, nReplicas, width, offset, consuming);
	    out << spaces << "     axis " << axis << std::endl
		<< spaces << "     nReplicas " << nReplicas << std::endl;
	    if (pv->GetParameterisation() != 0) 
		out << spaces << "    It is parameterisation " << std::endl;
	    else 
		out << spaces << "     width " << width << std::endl
		    << spaces << "     offset " << offset << std::endl
		    << spaces << "     consuming" <<  consuming << std::endl;
	    if (pv->GetParameterisation() != 0) 
		out << spaces << "    It is parameterisation " << std::endl;
	}
    }
}

void PrintGeomMatInfo::dumpTouch(G4VPhysicalVolume * pv, unsigned int leafDepth, std::ostream & out)
{
    std::string spaces = spacesFromLeafDepth(leafDepth);
    if (leafDepth == 0) fHistory.SetFirstEntry(pv);
    else fHistory.NewLevel(pv, kNormal, pv->GetCopyNo());

    G4ThreeVector globalpoint = fHistory.GetTopTransform().Inverse().
				TransformPoint(G4ThreeVector(0,0,0));
    G4LogicalVolume * lv = pv->GetLogicalVolume();

    std::string mother = "World";
    if (pv->GetMotherLogical()) mother = pv->GetMotherLogical()->GetName();
    std::string lvname = lv->GetName();
    lvname.assign(lvname,0,nchar);
    if (lvname == name)
        out << leafDepth << spaces << "### VOLUME = " << lv->GetName() 
	    << " Copy No " << pv->GetCopyNo() << " in " << mother
	    << " global position of centre " << globalpoint << " (r = " 
	    <<  globalpoint.perp() << ", phi = " <<  globalpoint.phi()/deg
	    << ")" << std::endl;

    int NoDaughters = lv->GetNoDaughters();
    while ((NoDaughters--)>0) 
    {
	G4VPhysicalVolume * pvD = lv->GetDaughter(NoDaughters);
	if (!pvD->IsReplicated()) dumpTouch(pvD, leafDepth+1, out);
    }

    if (leafDepth > 0) fHistory.BackLevel();
}

std::string PrintGeomMatInfo::spacesFromLeafDepth(unsigned int leafDepth)
{
    std::string spaces;
    unsigned int ii;
    for(ii = 0; ii < leafDepth; ii++) { spaces += "  "; }
    return spaces;
}

void PrintGeomMatInfo::dumpSolid(G4VSolid * sol, unsigned int leafDepth, std::ostream & out)
{
    std::string spaces = spacesFromLeafDepth(leafDepth);
    out << spaces << *(sol) << std::endl;
}

G4VPhysicalVolume * PrintGeomMatInfo::getTopPV()
{
    return G4TransportationManager::GetTransportationManager()
	->GetNavigatorForTracking()->GetWorldVolume();
}

G4LogicalVolume * PrintGeomMatInfo::getTopLV()
{ return theTopPV->GetLogicalVolume(); }


