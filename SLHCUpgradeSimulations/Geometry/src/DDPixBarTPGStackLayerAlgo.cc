///////////////////////////////////////////////////////////////////////////////
// File: DDPixBarTPGStackLayerAlgo.cc
// Description: Make one layer of stacked pixel barrel detector
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
#include "SLHCUpgradeSimulations/Geometry/interface/DDPixBarTPGStackLayerAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


////////////////////////////////////////////////////////////////////////////////
// Constructor
DDPixBarTPGStackLayerAlgo::DDPixBarTPGStackLayerAlgo() {
  LogDebug("PixelGeom") <<"DDPixBarTPGStackLayerAlgo info: Creating an instance";
}

////////////////////////////////////////////////////////////////////////////////
// Destructor
DDPixBarTPGStackLayerAlgo::~DDPixBarTPGStackLayerAlgo() {}

////////////////////////////////////////////////////////////////////////////////
// Initialization of algorithm
void DDPixBarTPGStackLayerAlgo::initialize(const DDNumericArguments & nArgs,
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
  coolZ     = nArgs["CoolZ"];
  coolNumber    = int(nArgs["CoolNumber"]);
  hybridThick = nArgs["HybridThick"];
  moduleRadius  = nArgs["ModuleRadius"];
  coolMat   = sArgs["CoolMaterial"];
  tubeMat   = sArgs["CoolTubeMaterial"];
  ladderNameUp  = sArgs["LadderNameUp"];
  ladderNameDown  = sArgs["LadderNameDown"];
  ladderWidth = nArgs["LadderWidth"];
  ladderThick = nArgs["LadderThick"];
  module_offset  = nArgs["ModuleOffset"];
  layout = int(nArgs["LayoutType"]);
  activeWidth = nArgs["ActiveWidth"];

// Debug messages
  //std::cout <<"\nStack sensor with sensorEdge = "<<sensorEdge<<"\tand width = "<<activeWidth<<"\t at R = "<<moduleRadius;
  LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo debug: Parent " << parentName 
			<< " NameSpace " << idNameSpace << "\n"
			<< "\tLadders " << number << "\tGeneral Material " 
			<< VolumeMaterial << "\tLength " << layerDz << "\tSensorEdge "
			<< sensorEdge << "\tSpecification of Cooling Pieces:\n"
			<< "\tLength " << coolDz << " Width " << coolWidth 
			<< " Side " << coolSide << " Thickness of Shell " 
			<< coolThick << " Radial distance " << moduleRadius 
			<< " Materials " << coolMat << ", " << tubeMat;

  LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo debug: Ladder " 
			<< ladderNameUp << " width/thickness " << ladderWidth
			<< ", " << ladderThick;
}


////////////////////////////////////////////////////////////////////////////////
// The algorithm itself
void DDPixBarTPGStackLayerAlgo::execute(DDCompactView& cpv) {
  if ((number%2==1)&&(layout==1)) { 
        number+=1;
        std::cout << "\nAsking for an invalid layout ... Adjusting the number of ladders to compensate.\n";
  }
  // Keep a running tally to check that there are no phi gaps.
  double phi_coverage = 0.0;		// Running total of Phi coverage
  bool covered=0;			// Set to 1 when there is at least 2Pi of coverage in phi
  double dphi = CLHEP::twopi/number;		// Phi difference between successive ladders
  double phi_offset = module_offset;	// Phi rotation of the ladders
  double radius_offset = 0.0;		// Distance from <R> that the stacks are shifted in or out
  double deltaX, deltaY; 		// Offset to correct for ladder thickness
  double deltaX2, deltaY2; 		// Offset for cooling tube 2
  double r_vol_inner = 0.0;		// Define the cylinder that the stacks are in
  double r_vol_outer = 0.0;		// 
  double phi_coverage_pinn =0.0;	// phi coverage, phi_coverage_pinn = phi_left + phi_right
  double phi_left    = 0.0;		// 
  double phi_right   = 0.0;		//


  // Set parameters for the Phi Rotated Stacks as default
  double d1 = (ladderThick)*tan(phi_offset);
  double d2 = (ladderThick)/cos(phi_offset);
  double d3 = (moduleRadius+d2);
  double d4 = ((activeWidth/2.0)-d1);
  double r_right = sqrt( d3*d3 + d4*d4 + 2*d3*d4*sin(phi_offset)) ;	// Radius of the outer edge of the active area
  phi_right=acos(	(r_right*r_right + d3*d3 - d4*d4)/
			(2*d3*r_right)
		);
  double d5 = sqrt(d1*d1+d2*d2);
  double d6 = (moduleRadius-d5);
  double r_left = sqrt ( d4*d4 + d6*d6 - 2*d4*d6*sin(phi_offset) ) ;	 // Radius of the inner edge of the active area
  phi_left=acos(	(r_left*r_left + d6*d6 - d4*d4)/
			(2*d6*r_left)
	       );
  if (r_right> r_left ) {r_vol_outer=r_right;r_vol_inner=r_left;}
  if (r_left > r_right) {r_vol_outer=r_left;r_vol_inner=r_right;}

  phi_coverage_pinn=phi_left+phi_right;
  //std::cout << "\nDetermining the radii, r_in="<<r_vol_inner   <<" mod_R="<<moduleRadius<<" r_out="<<r_vol_outer;
  // Set parameters if High-Low Stacks are requested
  if(layout) {
    phi_offset = 0.0;
    phi_coverage_pinn = 0.0; // Determin for each ladder when placed
    double R_Curvature = ((4*moduleRadius*moduleRadius)+(ladderWidth*ladderWidth/4))/(4*moduleRadius);	// The radius of the ends of the inner stack
    double r2 = (R_Curvature+ladderThick);
    double r1 = sqrt((R_Curvature*R_Curvature)-(ladderWidth*ladderWidth/4.0))-(ladderThick);

    radius_offset = (r1-r2)/2.0;
    r_vol_inner = r1-(ladderThick);
    r_vol_outer = sqrt((ladderWidth*ladderWidth/4.0)+((r2+ladderThick)*(r2+ladderThick)));
    // phi_left and phi_right depend on R so they will be determined later
    // std::cout << "\nDetermining the radii, r_in="<<r_vol_inner   <<" r1="<<r1<< " R_c="<<R_Curvature<<" r2="<<r2<<" r_out="<<r_vol_outer;
  }

  double r_vol_innerT;
  if(r_vol_inner>r_vol_outer) {
    r_vol_innerT=r_vol_inner;
    r_vol_inner=r_vol_outer-30;
    r_vol_outer=r_vol_innerT+30;
  }

  std::string name;

  int component_copy_no=1;
  double phi0 = 90*CLHEP::deg;
  double phi =0*CLHEP::deg;
  double phix=0*CLHEP::deg;
  double phiy=0*CLHEP::deg;
  DDTranslation tran;
  DDRotation rot;


  //std::cout << "\nDDPixBarTPGStackLayerAlgo test: r_mid_L_inner/r_mid_L_outer " << r_vol_inner << ", " << r_vol_outer ;
  //<< " d1/d2 " << d1 << ", " << d2 
  //<< " x1/x2 " << x1 << ", " << x2;


//------------------------------------------------------------------------------------------------------------
// Define the volume in which the layer exists

  DDName mother = parent().name();
  std::string idName = DDSplit(mother).first;

  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace), 0.5*layerDz, r_vol_inner, r_vol_outer, 0, CLHEP::twopi);

  DDName matname(DDSplit(VolumeMaterial).first, DDSplit(VolumeMaterial).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " 
			<< DDName(idName, idNameSpace) << " Tubs made of " 
			<< VolumeMaterial << " from 0 to " << CLHEP::twopi/CLHEP::deg 
			<< " with Rin " << r_vol_inner << " Rout " << r_vol_outer 
			<< " ZHalf " << 0.5*layerDz;

//------------------------------------------------------------------------------------------------------------
// Define the cool tube

  name = idName + "CoolTube";
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0, coolWidth/2, coolSide/2, coolSide/2, 0, coolWidth/2, coolSide/2, coolSide/2, 0);

  matter = DDMaterial(DDName(DDSplit(tubeMat).first, DDSplit(tubeMat).second));
  DDLogicalPart coolTube(solid.ddname(), matter, solid);

  LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " <<solid.name() 
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

  LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " <<solid.name() 
			<< " Trap made of " << tubeMat << " of dimensions " 
			<< 0.5*coolDz << ", 0, 0, " << coolWidth/2-coolThick << ", " << coolSide/2-coolThick 
			<< ", " << coolSide/2-coolThick << ", 0, " << coolWidth/2-coolThick << ", " << coolSide/2-coolThick << ", " 
			<< coolSide/2-coolThick << ", 0";

 
//------------------------------------------------------------------------------------------------------------
// Put coolant in the cool tube

  cpv.position (cool, coolTube, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());

  LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " << cool.name() 
			<< " number 1 positioned in " << coolTube.name() 
			<< " at (0,0,0) with no rotation";

//------------------------------------------------------------------------------------------------------------
// Define the ladder

  DDName ladderFullUp(DDSplit(ladderNameUp).first, DDSplit(ladderNameUp).second);
  DDName ladderFullDown(DDSplit(ladderNameDown).first, DDSplit(ladderNameDown).second);

//------------------------------------------------------------------------------------------------------------


// Iterate over the number of modules

  for (int i=0; i<number; i++) {
	
    double phi_coverage_i=0.0;
    // First the modules
    phi = phi0 + i*dphi;
    phix = phi + (90*CLHEP::deg) - phi_offset ;
    phiy = phix + (90*CLHEP::deg) ;

    deltaX= 0.5*ladderThick*cos(phi-phi_offset);
    deltaY= 0.5*ladderThick*sin(phi-phi_offset);

    double radius;
    if((i%2)==0) radius=moduleRadius-radius_offset;
    else radius=moduleRadius+radius_offset;

    //inner layer of stack
    tran = DDTranslation(radius*cos(phi)-deltaX, radius*sin(phi)-deltaY, 0);
    name = idName + dbl_to_string(component_copy_no);
    rot = DDrot(DDName(name,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);

    cpv.position (ladderFullDown, layer, component_copy_no, tran, rot);

    LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " << ladderFullDown 
			    << " number " << component_copy_no
			    << " positioned in " << layer.name()
			    << " at " << tran
			    << " with " << rot;
    component_copy_no++;


    //outer layer of stack
    tran = DDTranslation(radius*cos(phi)+deltaX, radius*sin(phi)+deltaY, 0);
    name = idName + dbl_to_string(component_copy_no);
    rot = DDrot(DDName(name,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);

    cpv.position (ladderFullUp, layer, component_copy_no, tran, rot);

    LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " << ladderFullUp 
			    << " number " << component_copy_no
			    << " positioned in " << layer.name()
			    << " at " << tran
			    << " with " << rot;
    component_copy_no++;
    // Running total of phi coverage
    phi_coverage_i=phi_coverage_pinn;
    if(layout) {
	phi_coverage_i=2*atan2((activeWidth/2.0),(radius+ladderThick));
    }

    phi_coverage += phi_coverage_i;
    //std::cout<<"\nLooking at phi = "<< phi<<"\tNumber "<<component_copy_no-1<<"\t with "<<phi_coverage_i<<"\trad of coverage for a total coverage of "<<phi_coverage;
    if (phi_coverage>CLHEP::twopi&&covered==0) {
       //std::cout<<"\nPhi coverage is achieved after "<<(component_copy_no-1)/2.0<<" ladders for R="<<radius/10.0<<" cm.\t and "<<number<<" ladders were asked for";
       covered=1;
    }

 
  }
  //std::cout<<"\nLayer covered "<<phi_coverage<<" radians in phi.   (2Pi="<<CLHEP::twopi<<")";
  if (phi_coverage<CLHEP::twopi) { throw cms::Exception("DDPixBarTPGStackLayerAlgo")
      <<"\nAsking for a Geometry with gaps in phi.\n";}

// Iterate over the number of ladders (now 2 cooltubes per ladder)

  DDTranslation tran2;
  double coolOffset = 0.5*ladderWidth - hybridThick - 0.5*coolSide;
  double coolOffset2 = -0.5*ladderWidth + 0.5*coolSide;

  for (int i=0; i<number; i++) {
    phi = phi0 + i*dphi;
    phix = phi + (90*CLHEP::deg) - phi_offset;
    phiy = phix + (90*CLHEP::deg) ;

    deltaX= coolOffset*cos(90*CLHEP::deg-phi+phi_offset);
    deltaY= coolOffset*sin(90*CLHEP::deg-phi+phi_offset);
    deltaX2= coolOffset2*cos(90*CLHEP::deg-phi+phi_offset);
    deltaY2= coolOffset2*sin(90*CLHEP::deg-phi+phi_offset);

    double radius;              
    if((i%2)==0) radius=moduleRadius-radius_offset;
    else radius=moduleRadius+radius_offset;

    tran = DDTranslation(radius*cos(phi)-deltaX, radius*sin(phi)+deltaY, coolZ);
    tran2 = DDTranslation(radius*cos(phi)-deltaX2, radius*sin(phi)+deltaY2, coolZ);

    name = idName + "xxx"+dbl_to_string(i+10000);

    rot = DDrot(DDName(name,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);
    cpv.position (coolTube, layer, i*2+1, tran, rot);
    LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " << coolTube.name() 
			  << " number " << i*2+1 << " positioned in " 
			  << layer.name() << " at " << tran << " with "<< rot;
    cpv.position (coolTube, layer, i*2+2, tran2, rot);
    LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " << coolTube.name() 
			  << " number " << i*2+2 << " positioned in " 
			  << layer.name() << " at " << tran2 << " with "<< rot;
     // check if ring layer and need cooling tubes on both sides
    if(coolNumber == 2) {
       tran = DDTranslation(radius*cos(phi)-deltaX, radius*sin(phi)+deltaY, -coolZ);
       tran2 = DDTranslation(radius*cos(phi)-deltaX2, radius*sin(phi)+deltaY2, -coolZ);

       name = idName + "xxx2"+dbl_to_string(i+10000);

       rot = DDrot(DDName(name,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);
       cpv.position (coolTube, layer, number*2+i*2+1, tran, rot);
       LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " << coolTube.name() 
                             << " number " << number*2+i*2+1 << " positioned in " 
                             << layer.name() << " at " << tran << " with "<< rot;
       cpv.position (coolTube, layer, number*2+i*2+2, tran2, rot);
       LogDebug("PixelGeom") << "DDPixBarTPGStackLayerAlgo test: " << coolTube.name() 
                             << " number " << number*2+i*2+2 << " positioned in " 
                             << layer.name() << " at " << tran2 << " with "<< rot;
    }
  }


 // End algorithm
}
