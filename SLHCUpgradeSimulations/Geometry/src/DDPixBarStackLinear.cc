///////////////////////////////////////////////////////////////////////////////
// File: DDPixBarStackLinear.cc
// Description: Position n copies of stacked modules at given intervals along an axis
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "SLHCUpgradeSimulations/Geometry/interface/DDPixBarStackLinear.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDPixBarStackLinear::DDPixBarStackLinear() {
  LogDebug("TrackerGeom") << "DDPixBarStackLinear info: Creating an instance";
}

DDPixBarStackLinear::~DDPixBarStackLinear() {}

void DDPixBarStackLinear::initialize(const DDNumericArguments & nArgs,
				 const DDVectorArguments & vArgs,
				 const DDMapArguments & ,
				 const DDStringArguments & sArgs,
				 const DDStringVectorArguments &) {

  number      = int(nArgs["Number"]);
  theta       = nArgs["Theta"];
  phi         = nArgs["Phi"];
  offset      = nArgs["Offset"];
  delta       = nArgs["Delta"];
  centre      = vArgs["Center"];
  rotMat      = sArgs["Rotation"];
  stackoffset = nArgs["StackOffset"];
  stackoffsetT= int(nArgs["StackOffsetT"]);
  zoffset     = nArgs["ZOffset"];
  
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDPixBarStackLinear debug: Parent " << parentName 
			  << "\tChild " << childName << " NameSpace " 
			  << idNameSpace << "\tNumber " << number 
			  << "\tAxis (theta/phi) " << theta/CLHEP::deg << ", "
			  << phi/CLHEP::deg << "\t(Offset/Delta) " << offset << ", " 
			  << delta << "\tCentre " << centre[0] << ", " 
			  << centre[1] << ", " << centre[2] << "\tRotation "
			  << rotMat;
}

void DDPixBarStackLinear::execute(DDCompactView& cpv) {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  DDTranslation direction(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));
  DDTranslation base(centre[0],centre[1],centre[2]); 
  DDTranslation zbase(centre[0],zoffset,centre[2]);
  std::string rotstr = DDSplit(rotMat).first;
  DDRotation rot;
  if (rotstr != "NULL") {
    std::string rotns  = DDSplit(rotMat).second;
    rot = DDRotation(DDName(rotstr, rotns));
  }

  for (int i=0; i<(number/2); i++) {

    if((stackoffset!=0.0)&&(i!=0)) {
      if(i%stackoffsetT==0) offset+=stackoffset;
    }
	
    DDTranslation tran = base + (offset + double(i)*delta)*direction;
    cpv.position (child, mother, 2*i+1, tran, rot);
    LogDebug("TrackerGeom") << "DDPixBarStackLinear test: " << child << " number "
			    << 2*i+1 << " positioned in " << mother << " at "
			    << tran << " with " << rot;

    DDTranslation tran2 = base - (offset + double(i)*delta)*direction;
    cpv.position (child, mother, 2*i+2, tran2, rot);
    LogDebug("TrackerGeom") << "DDPixBarStackLinear test: " << child << " number "
                            << 2*i+2 << " positioned in " << mother << " at "
                            << tran << " with " << rot;

    if(zoffset!=0.0){ 
      if((i+1)!=number){
        i++;
        DDTranslation tran3 = zbase + (offset + double(i)*delta)*direction;
        cpv.position (child, mother, 2*i+1, tran3, rot);
        LogDebug("TrackerGeom") << "DDPixBarStackLinear test: " << child << " number "
                            << 2*i+1 << " positioned in " << mother << " at "
                            << tran << " with " << rot;
    
        DDTranslation tran4 = zbase - (offset + double(i)*delta)*direction;
        cpv.position (child, mother, 2*i+2, tran4, rot);
        LogDebug("TrackerGeom") << "DDPixBarStackLinear test: " << child << " number "
                            << 2*i+2 << " positioned in " << mother << " at "
                            << tran << " with " << rot;

      }
    }

  }
}
