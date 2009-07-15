///////////////////////////////////////////////////////////////////////////////
// File: DDPixBarStackLayerAlgo.cc
// Description: Make one layer of stacked pixel barrel detector
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "SLHCUpgradeSimulations/Geometry/interface/DDPixBarStackLayerAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


////////////////////////////////////////////////////////////////////////////////
// Constructor
DDPixBarStackLayerAlgo::DDPixBarStackLayerAlgo() {
  LogDebug("PixelGeom") <<"DDPixBarStackLayerAlgo info: Creating an instance";
}

////////////////////////////////////////////////////////////////////////////////
// Destructor
DDPixBarStackLayerAlgo::~DDPixBarStackLayerAlgo() {}

////////////////////////////////////////////////////////////////////////////////
// Initialization of algorithm
void DDPixBarStackLayerAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {


// Retrieve the variables from the XML files
  idNameSpace = DDCurrentNamespace::ns();
  DDName parentName = parent().name();

  VolumeMaterial    = sArgs["VolumeMaterial"];
  number    = int(nArgs["Ladders"]);
  layerDz   = nArgs["LayerDz"];
  sensorEdge= nArgs["SensorEdge"];
  coolDz    = nArgs["CoolDz"];
  coolWidth = nArgs["CoolWidth"];
  coolSide  = nArgs["CoolSide"];
  coolThick = nArgs["CoolThick"];
  moduleRadius  = nArgs["ModuleRadius"];
  coolMat   = sArgs["CoolMaterial"];
  tubeMat   = sArgs["CoolTubeMaterial"];
  ladderNameUp  = sArgs["LadderNameUp"];
  ladderNameDown  = sArgs["LadderNameDown"];
  ladderWidth = nArgs["LadderWidth"];
  ladderThick = nArgs["LadderThick"];
  module_offset  = nArgs["ModuleOffset"];
  layout = int(nArgs["LayoutType"]);

// Debug messages
  LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo debug: Parent " << parentName 
			<< " NameSpace " << idNameSpace << "\n"
			<< "\tLadders " << number << "\tGeneral Material " 
			<< VolumeMaterial << "\tLength " << layerDz << "\tSensorEdge "
			<< sensorEdge << "\tSpecification of Cooling Pieces:\n"
			<< "\tLength " << coolDz << " Width " << coolWidth 
			<< " Side " << coolSide << " Thickness of Shell " 
			<< coolThick << " Radial distance " << moduleRadius 
			<< " Materials " << coolMat << ", " << tubeMat;

  LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo debug: Ladder " 
			<< ladderNameUp << " width/thickness " << ladderWidth
			<< ", " << ladderThick;
}


////////////////////////////////////////////////////////////////////////////////
// The algorithm itself
void DDPixBarStackLayerAlgo::execute() {
  if ((number%2==1)&&(layout==1)) { 
        number+=1;
        std::cout << "\nAsking for an invalid layout ... Adjusting the number of ladders to compensate.\n";
  }
//  if ((number%2==1)&&(layout==1)) { throw cms::Exception("DDPixBarStackLayerAlgo") 
//	<< "\nAsking for a TOB like Geometry with an odd number of stacks.\n\n";}
// Define some obscure badly defined variables
  double dphi = twopi/number;

  double phi_offset = module_offset;
  double radius_offset = 0.0;
  if(layout) {
    phi_offset = 0.0;
    radius_offset = ladderThick;
  }

  double delta1=0.5*ladderWidth*sin(phi_offset);
  double delta2=ladderThick*cos(phi_offset);
  double delta3=radius_offset;

  double deltaX, deltaY; //Offset to correct for ladder thickness

  double r_vol_inner = moduleRadius-(delta1+delta2+delta3);
  // for a test
  double r_vol_outer = moduleRadius+(delta1+delta2+delta3);
  //double r_vol_outer = moduleRadius+(delta1+delta2+delta3)+3.0;

  double r_vol_innerT;
  if(r_vol_inner>r_vol_outer) {
    r_vol_innerT=r_vol_inner;
    r_vol_inner=r_vol_outer-30;
    r_vol_outer=r_vol_innerT+30;
  }
  if(layout) {
    double temp = (double)(ladderWidth/2.0);
    double extra_thick = sqrt(temp*temp + r_vol_outer*r_vol_outer) - r_vol_outer;
    r_vol_outer = r_vol_outer + extra_thick;
  }

  std::string name;

  int component_copy_no=1;
  double phi0 = 90*deg;
  double phi =0*deg;
  double phix=0*deg;
  double phiy=0*deg;
  DDTranslation tran;
  DDRotation rot;


  LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo test: r_mid_L_inner/r_mid_L_outer " << r_vol_inner << ", " << r_vol_outer ;
  //<< " d1/d2 " << d1 << ", " << d2 
  //<< " x1/x2 " << x1 << ", " << x2;


//------------------------------------------------------------------------------------------------------------
// Define the volume in which the layer exists

  DDName mother = parent().name();
  std::string idName = DDSplit(mother).first;

  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace), 0.5*layerDz, r_vol_inner, r_vol_outer, 0, twopi);

  DDName matname(DDSplit(VolumeMaterial).first, DDSplit(VolumeMaterial).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo test: " 
			<< DDName(idName, idNameSpace) << " Tubs made of " 
			<< VolumeMaterial << " from 0 to " << twopi/deg 
			<< " with Rin " << r_vol_inner << " Rout " << r_vol_outer 
			<< " ZHalf " << 0.5*layerDz;

//------------------------------------------------------------------------------------------------------------
// Define the cool tube

  name = idName + "CoolTube";
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0, coolWidth/2, coolSide/2, coolSide/2, 0, coolWidth/2, coolSide/2, coolSide/2, 0);

  matter = DDMaterial(DDName(DDSplit(tubeMat).first, DDSplit(tubeMat).second));
  DDLogicalPart coolTube(solid.ddname(), matter, solid);

  LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo test: " <<solid.name() 
			<< " Trap made of " << tubeMat << " of dimensions " 
			<< 0.5*coolDz << ", 0, 0, " << coolWidth/2 << ", " << coolSide/2 
			<< ", " << coolSide/2 << ", 0, " << coolWidth/2 << ", " << coolSide/2 << ", " 
			<< coolSide/2 << ", 0";


//------------------------------------------------------------------------------------------------------------
// Define the coolant within the cool tube = same as cooltube - wall thickness

  name = idName + "Coolant";

  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0, coolWidth/2-coolThick, coolSide/2-coolThick, coolSide/2-coolThick, 0, coolWidth/2-coolThick, coolSide/2-coolThick, coolSide/2-coolThick, 0);
  matter = DDMaterial(DDName(DDSplit(coolMat).first, DDSplit(coolMat).second));
  DDLogicalPart cool(solid.ddname(), matter, solid);

  LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo test: " <<solid.name() 
			<< " Trap made of " << tubeMat << " of dimensions " 
			<< 0.5*coolDz << ", 0, 0, " << coolWidth/2-coolThick << ", " << coolSide/2-coolThick 
			<< ", " << coolSide/2-coolThick << ", 0, " << coolWidth/2-coolThick << ", " << coolSide/2-coolThick << ", " 
			<< coolSide/2-coolThick << ", 0";

 
//------------------------------------------------------------------------------------------------------------
// Put coolant in the cool tube

  DDpos (cool, coolTube, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());

  LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo test: " << cool.name() 
			<< " number 1 positioned in " << coolTube.name() 
			<< " at (0,0,0) with no rotation";

//------------------------------------------------------------------------------------------------------------
// Define the ladder

  DDName ladderFullUp(DDSplit(ladderNameUp).first, DDSplit(ladderNameUp).second);
  DDName ladderFullDown(DDSplit(ladderNameDown).first, DDSplit(ladderNameDown).second);

//------------------------------------------------------------------------------------------------------------


// Iterate over the number of modules

  for (int i=0; i<number; i++) {
	
    // First the modules
    phi = phi0 + i*dphi;
    phix = phi + (90*deg) - phi_offset ;
    phiy = phix + (90*deg) ;

    deltaX= 0.5*ladderThick*cos(phi-phi_offset);
    deltaY= 0.5*ladderThick*sin(phi-phi_offset);

    double radius;
    if((i%2)==0) radius=moduleRadius-radius_offset;
    else radius=moduleRadius+radius_offset;

    //inner layer of stack
    tran = DDTranslation(radius*cos(phi)-deltaX, radius*sin(phi)-deltaY, 0);
    name = idName + dbl_to_string(component_copy_no);
    rot = DDrot(DDName(name,idNameSpace), 90*deg, phix, 90*deg, phiy, 0.,0.);

    DDpos (ladderFullDown, layer, component_copy_no, tran, rot);

    LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo test: " << ladderFullDown 
			    << " number " << component_copy_no
			    << " positioned in " << layer.name()
			    << " at " << tran
			    << " with " << rot;
    component_copy_no++;


    //outer layer of stack
    tran = DDTranslation(radius*cos(phi)+deltaX, radius*sin(phi)+deltaY, 0);
    name = idName + dbl_to_string(component_copy_no);
    rot = DDrot(DDName(name,idNameSpace), 90*deg, phix, 90*deg, phiy, 0.,0.);

    DDpos (ladderFullUp, layer, component_copy_no, tran, rot);

    LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo test: " << ladderFullUp 
			    << " number " << component_copy_no
			    << " positioned in " << layer.name()
			    << " at " << tran
			    << " with " << rot;
    component_copy_no++;

 
  }


// Iterate over the number of cooltubes

  double coolOffset = 0.5*ladderWidth - 0.5*coolSide;

  for (int i=0; i<number; i++) {
    phi = phi0 + i*dphi;
    phix = phi + (90*deg) - phi_offset;
    phiy = phix + (90*deg) ;

    deltaX= coolOffset*cos(90*deg-phi+phi_offset);
    deltaY= coolOffset*sin(90*deg-phi+phi_offset);

    double radius;              
    if((i%2)==0) radius=moduleRadius-radius_offset;
    else radius=moduleRadius+radius_offset;

    tran = DDTranslation(radius*cos(phi)-deltaX, radius*sin(phi)+deltaY, 0);

    name = idName + "xxx"+dbl_to_string(i+10000);

    rot = DDrot(DDName(name,idNameSpace), 90*deg, phix, 90*deg, phiy, 0.,0.);
    DDpos (coolTube, layer, i+1, tran, rot);
    LogDebug("PixelGeom") << "DDPixBarStackLayerAlgo test: " << coolTube.name() 
			  << " number " << i+1 << " positioned in " 
			  << layer.name() << " at " << tran << " with "<< rot;
  }


 // End algorithm
}
