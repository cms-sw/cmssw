///////////////////////////////////////////////////////////////////////////////
// File: DDPixBarLayerUpgradeAlgoCoverage.cc
// Description: Make one layer of pixel barrel detector for Upgrading.
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "SLHCUpgradeSimulations/Geometry/interface/DDPixBarLayerUpgradeAlgoCoverage.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDPixBarLayerUpgradeAlgoCoverage::DDPixBarLayerUpgradeAlgoCoverage() {
  LogDebug("PixelGeom") <<"DDPixBarLayerUpgradeAlgoCoverage info: Creating an instance";
}

DDPixBarLayerUpgradeAlgoCoverage::~DDPixBarLayerUpgradeAlgoCoverage() {}

void DDPixBarLayerUpgradeAlgoCoverage::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  idNameSpace = DDCurrentNamespace::ns();
  DDName parentName = parent().name();

  genMat    = sArgs["GeneralMaterial"];
  number    = int(nArgs["Ladders"]);
  layerDz   = nArgs["LayerDz"];
  sensorEdge= nArgs["SensorEdge"];
  coolDz    = nArgs["CoolDz"];
  coolWidth = nArgs["CoolWidth"];
  coolSide  = nArgs["CoolSide"];
  coolThick = nArgs["CoolThick"];
  coolDist  = nArgs["CoolDist"];
  coolMat   = sArgs["CoolMaterial"];
  tubeMat   = sArgs["CoolTubeMaterial"];


  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage debug: Parent " << parentName 
			<< " NameSpace " << idNameSpace << "\n"
			<< "\tLadders " << number << "\tGeneral Material " 
			<< genMat << "\tLength " << layerDz << "\tSensorEdge "
			<< sensorEdge << "\tSpecification of Cooling Pieces:\n"
			<< "\tLength " << coolDz << " Width " << coolWidth 
			<< " Side " << coolSide << " Thickness of Shell " 
			<< coolThick << " Radial distance " << coolDist 
			<< " Materials " << coolMat << ", " << tubeMat;

  ladder      = sArgs["LadderName"];
  ladderWidth = nArgs["LadderWidth"];
  ladderThick = nArgs["LadderThick"];
  activeWidth = nArgs["ActiveWidth"];
  outerFirst  = int(nArgs["OuterFirst"]);
 
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage debug: Full Ladder " 
			<< ladder << " width/thickness " << ladderWidth
			<< ", " << ladderThick;
  std::cout << " LadderThick = " << ladderThick << std::endl;
  std::cout << " LadderWidth = " << ladderWidth << std::endl;
  std::cout << " ActiveWidth = " << activeWidth << std::endl;
}

void DDPixBarLayerUpgradeAlgoCoverage::execute(DDCompactView& cpv) {

  DDName      mother = parent().name();
  std::string idName = DDSplit(mother).first;

  // Keep a running tally to check that there are no phi gaps.
  double phi_coverage = 0.0;            // Running total of Phi coverage
  double dphi = CLHEP::twopi/number;
  double d2   = 0.5*coolWidth;
  double d1   = d2 - coolSide*sin(0.5*dphi);
  double x1   = (d1+d2)/(2.*sin(0.5*dphi));
  double x2   = coolDist*sin(0.5*dphi);
  double rmin = (coolDist-0.5*(d1+d2))*cos(0.5*dphi)-0.5*ladderThick;
  double rmax = (coolDist+0.5*(d1+d2))*cos(0.5*dphi)+0.5*ladderThick;
  double rmxh = rmax + 0.5*ladderThick;
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: Rmin/Rmax " << rmin 
			<< ", " << rmax << " d1/d2 " << d1 << ", " << d2 
			<< " x1/x2 " << x1 << ", " << x2;

  double rtmi = rmin - 0.5*ladderThick;
  double rtmx = sqrt(rmxh*rmxh+ladderWidth*ladderWidth);
  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace),0.5*layerDz,
                                       rtmi, rtmx, 0, CLHEP::twopi);
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: " 
			<< DDName(idName, idNameSpace) << " Tubs made of " 
			<< genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
			<< " with Rin " << rtmi << " Rout " << rtmx 
			<< " ZHalf " << 0.5*layerDz;
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  double rr = 0.5*(rmax+rmin);
  double dr = 0.5*(rmax-rmin);
  double h1 = 0.5*coolSide*cos(0.5*dphi);
  std::string name = idName + "CoolTube";
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0,
			       h1, d2, d1, 0, h1, d2, d1, 0);
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: " <<solid.name() 
			<< " Trap made of " << tubeMat << " of dimensions " 
			<< 0.5*coolDz << ", 0, 0, " << h1 << ", " << d2 
			<< ", " << d1 << ", 0, " << h1 << ", " << d2 << ", " 
			<< d1 << ", 0";
  matter = DDMaterial(DDName(DDSplit(tubeMat).first, DDSplit(tubeMat).second));
  DDLogicalPart coolTube(solid.ddname(), matter, solid);

  name = idName + "Coolant";
  h1  -= coolThick;
  d1  -= coolThick;
  d2  -= coolThick;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0,
			       h1, d2, d1, 0, h1, d2, d1, 0);
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: " <<solid.name() 
			<< " Trap made of " << coolMat << " of dimensions " 
			<< 0.5*coolDz << ", 0, 0, " << h1 << ", " << d2
			<< ", " << d1 << ", 0, " << h1 << ", " << d2 << ", " 
			<< d1 << ", 0";
  matter = DDMaterial(DDName(DDSplit(coolMat).first, DDSplit(coolMat).second));
  DDLogicalPart cool(solid.ddname(), matter, solid);
  cpv.position (cool, coolTube, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: " << cool.name() 
			<< " number 1 positioned in " << coolTube.name() 
			<< " at (0,0,0) with no rotation";

  DDName ladderFull(DDSplit(ladder).first, DDSplit(ladder).second);
  int  copy=1, iup=(-1)*outerFirst;
  for (int i=1; i<number+1; i++) {
    double phi = i*dphi;
    double phix, phiy, rrr;
    std::string rots;
    DDTranslation tran;
    DDRotation rot;
    iup  =-iup;
    rrr  = rr + iup*dr;
    tran = DDTranslation(rrr*cos(phi), rrr*sin(phi), 0);
    rots = idName + dbl_to_string(copy);
    if (iup > 0) phix = phi-90*CLHEP::deg;
    else         phix = phi+90*CLHEP::deg;
    phiy = phix+90.*CLHEP::deg;
    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: Creating a new "
	                  << "rotation: " << rots << "\t90., " << phix/CLHEP::deg 
		          << ", 90.," << phiy/CLHEP::deg << ", 0, 0";
    rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);
    cpv.position (ladderFull, layer, copy, tran, rot);
    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: " << ladderFull 
	                  << " number " << copy << " positioned in " 
			  << layer.name() << " at " << tran << " with " 
			  << rot;
    double phi_coverage_i=2*atan2((activeWidth/2.0),(rrr+(ladderThick/2.0)));
  //std::cout << " radius = " << rrr << " phi value " << phi_coverage_i << std::endl;
    phi_coverage += phi_coverage_i;
    copy++;
    rrr  = coolDist*cos(0.5*dphi);
    tran = DDTranslation(rrr*cos(phi)-x2*sin(phi), 
			 rrr*sin(phi)+x2*cos(phi), 0);
    rots = idName + dbl_to_string(i+100);
    phix = phi+0.5*dphi;
    if (iup > 0) phix += 180*CLHEP::deg;
    phiy = phix+90.*CLHEP::deg;
    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: Creating a new "
			  << "rotation: " << rots << "\t90., " << phix/CLHEP::deg 
			  << ", 90.," << phiy/CLHEP::deg << ", 0, 0";
    rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);
    cpv.position (coolTube, layer, i+1, tran, rot);
    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgoCoverage test: " << coolTube.name() 
			  << " number " << i+1 << " positioned in " 
			  << layer.name() << " at " << tran << " with "<< rot;
  }
  std::cout<<"\nLayer covered "<<phi_coverage<<" radians in phi.   (2Pi="<<CLHEP::twopi<<")" << std::endl;
  if (phi_coverage>CLHEP::twopi) std::cout<<"\nPhi coverage is achieved"<< std::endl;
}
