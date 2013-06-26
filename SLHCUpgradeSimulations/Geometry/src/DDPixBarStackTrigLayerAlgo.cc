///////////////////////////////////////////////////////////////////////////////
// File: DDPixBarStackTrigLayerAlgo.cc
// Description: Make one stack trig layer of pixel barrel detector
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
#include "SLHCUpgradeSimulations/Geometry/interface/DDPixBarStackTrigLayerAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDPixBarStackTrigLayerAlgo::DDPixBarStackTrigLayerAlgo() {
  LogDebug("PixelGeom") <<"DDPixBarStackTrigLayerAlgo info: Creating an instance";
}

DDPixBarStackTrigLayerAlgo::~DDPixBarStackTrigLayerAlgo() {}

void DDPixBarStackTrigLayerAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  idNameSpace = DDCurrentNamespace::ns();

  genMat    = sArgs["GeneralMaterial"];
  number    = int(nArgs["Ladders"]);

  layerDz   = nArgs["LayerDz"];
  layerR    = nArgs["LayerR"];

  ladder      = vsArgs["LadderName"];
  ladderWidth = vArgs["LadderWidth"];
  ladderThick = vArgs["LadderThick"];
  LogDebug("PixelGeom") << " in stacktrig algo: number = " << number
                        << " layerDz = " << layerDz
                        << " layerR  = " << layerR << std::endl;
}

void DDPixBarStackTrigLayerAlgo::execute(DDCompactView& cpv) {

  DDName      mother = parent().name();
  std::string idName = DDSplit(mother).first;

  double dphi = CLHEP::twopi/number;

  // Firstly Create Solid volumon for Stack Trig layer.
  double rtmi = cos(0.5*dphi)*layerR- ladderThick[0] ;  //fix  me here  for temporary test.
  double rtmx = layerR ;
  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace),0.5*layerDz,
                                       rtmi, rtmx, 0, CLHEP::twopi);
  LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: "
                        << DDName(idName, idNameSpace) << " Tubs made of "
                        << genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg
                        << " with Rin " << rtmi << " Rout " << rtmx
                        << " ZHalf " << 0.5*layerDz;
  // Create Logical  volume  for Stack Trig Layer
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);
  // Note, How to place this stack Trig layer will be defined in data/stacktriglayer0.xml

  DDName ladderFull(DDSplit(ladder[0]).first, DDSplit(ladder[0]).second);

  int  copy=1, iup=-1;
  double phi0 = 90*CLHEP::deg;
  for (int i=0; i<number; i++) {
	
    double phi = phi0 + i*dphi;
    double phix, phiy, rrr ;
    std::string rots;
    // Create Translation  for each Ladder.
    DDTranslation tran;
    DDRotation rot;
      iup  =-iup;
      rrr  = rtmi+ 0.5* ladderThick[0];
      tran = DDTranslation(rrr*cos(phi), rrr*sin(phi), 0);
      rots = idName + dbl_to_string(copy);
      if (iup > 0) phix = phi-90*CLHEP::deg;
      else         phix = phi+90*CLHEP::deg;
      phiy = phix+90.*CLHEP::deg;

      LogDebug("PixelGeom") << "DDPixBarStackTrigLayerAlgo test: Creating a new "
			    << "rotation: " << rots << "\t90., " << phix/CLHEP::deg 
			    << ", 90.," << phiy/CLHEP::deg << ", 0, 0"; 
      
      //Create Rotation for each Ladder.
      rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);

     //Place each Ladder into this Stack  Trig Layer
      cpv.position (ladderFull, layer, copy, tran, rot);

      LogDebug("PixelGeom") << "DDPixBarStackTrigLayerAlgo test: " << ladderFull 
			    << " number " << copy << " positioned in " 
			    << layer.name() << " at " << tran << " with " 
			    << rot; 
      copy++;

  }
}
