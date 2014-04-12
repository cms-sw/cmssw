#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Box.h>
#include <DetectorDescription/Core/src/Cons.h>
#include <DetectorDescription/Core/src/Ellipsoid.h>
#include <DetectorDescription/Core/src/EllipticalTube.h>
#include <DetectorDescription/Core/src/Orb.h>
#include <DetectorDescription/Core/src/Parallelepiped.h>
#include <DetectorDescription/Core/src/Polycone.h>
#include <DetectorDescription/Core/src/Polyhedra.h>
#include <DetectorDescription/Core/src/Sphere.h>
#include <DetectorDescription/Core/src/Torus.h>
#include <DetectorDescription/Core/src/Trap.h>
#include <DetectorDescription/Core/src/Tubs.h>

#include <DataFormats/GeometryVector/interface/Pi.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Box.hh>
#include <G4Cons.hh>
#include <G4Ellipsoid.hh>
#include <G4EllipticalTube.hh>
#include <G4Orb.hh>
#include <G4Para.hh>
#include <G4Polycone.hh>
#include <G4Polyhedra.hh>
#include <G4Sphere.hh>
#include <G4Torus.hh>
#include <G4Trap.hh>
#include <G4Trd.hh>
#include <G4Tubs.hh>
#include <string>

//
// See Geant4 documentation for more details:
//
// http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html#sect.Geom.Solids
//
//
// This test verifies convertion of the DDD solids to Geant4 Constructed Solid Geometry (CSG) Solids
//

//
// 1. Box:
//
// G4Box(const G4String& pName,                         
//       G4double  pX,
//       G4double  pY,
//       G4double  pZ)
void
doBox( const std::string& name, double xHalfLength, double yHalfLength, 
       double zHalfLength )
{
  G4Box g4( name, xHalfLength, yHalfLength, zHalfLength );
  DDI::Box dd( xHalfLength, yHalfLength, zHalfLength );
  DDBox dds = DDSolidFactory::box( name, xHalfLength, yHalfLength, zHalfLength );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 2. Cylindrical Section or Tube:
//
// G4Tubs(const G4String& pName,                        
//        G4double  pRMin,
//        G4double  pRMax,
//        G4double  pDz,
//        G4double  pSPhi,
//        G4double  pDPhi)
void
doTubs( const std::string& name, double rIn, double rOut, 
	double zhalf, double startPhi, double deltaPhi )
{
  G4Tubs g4( name, rIn, rOut, zhalf, startPhi, deltaPhi );
  DDI::Tubs dd( zhalf, rIn, rOut, startPhi, deltaPhi );
  DDTubs dds = DDSolidFactory::tubs( name, zhalf, rIn, rOut, startPhi, deltaPhi );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 3. Cone or Conical section:
//
// G4Cons(const G4String& pName,                        
//        G4double  pRmin1,
//        G4double  pRmax1,
//        G4double  pRmin2,
//        G4double  pRmax2,
//        G4double  pDz,
//        G4double  pSPhi,
//        G4double  pDPhi)
void
doCons( const std::string& name,
	double rIn1, double rOut1,
	double rIn2, double rOut2,
	double zhalf, double startPhi, double deltaPhi )
{
  G4Cons g4( name, rIn1, rOut1, rIn2, rOut2, zhalf, startPhi, deltaPhi );
  DDI::Cons dd( zhalf, rIn1, rOut1, rIn2, rOut2, startPhi, deltaPhi );
  DDCons dds = DDSolidFactory::cons( name, zhalf, rIn1, rOut1, rIn2, rOut2, startPhi, deltaPhi );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 4. Parallelepiped:
//
// G4Para(const G4String& pName,                                  
//        G4double   dx,
//        G4double   dy,
//        G4double   dz,
//        G4double   alpha,
//        G4double   theta,
//        G4double   phi)
void
doPara( const std::string& name, double xHalf, double yHalf, 
	double zHalf, double alpha, double theta, double phi )
{
  G4Para g4(name, xHalf, yHalf, zHalf, alpha, theta, phi );
  DDI::Parallelepiped dd( xHalf, yHalf, zHalf, alpha, theta, phi );
  DDParallelepiped dds = DDSolidFactory::parallelepiped( name, xHalf, yHalf, zHalf, alpha, theta, phi );
  dd.stream( std::cout );
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 5. Trapezoid:
//
// G4Trd(const G4String& pName,                            
//       G4double  dx1,
//       G4double  dx2,
//       G4double  dy1,
//       G4double  dy2,
//       G4double  dz)
void
doTrd( const std::string& name, double dx1, double dx2, 
       double dy1, double dy2, double dz )
{
  G4Trd g4( name, dx1, dx2, dy1, dy2, dz );
  /////////////////////////////////////////
  // DDD does not have direct implementation of Trd.
  // Use generic trapezoid instead.
  DDI::Trap dd( dz,  0.0 /* pTheta */, 0.0 /* pPhi */,
		dy1, dx1, dx1, 0.0 /* pAlp1 */,
		dy2, dx2, dx2, 0.0 /* pAlp2 */);
  DDTrap dds = DDSolidFactory::trap( name, dz, 0.0 /* pTheta */, 0.0 /* pPhi */,
				     dy1, dx1, dx1, 0.0 /* pAlp1 */,
				     dy2, dx2, dx2, 0.0 /* pAlp2 */);
  dd.stream( std::cout );
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 6. Generic Trapezoid:
//
// G4Trap(const G4String& pName,                        
//        G4double   pZ,
//        G4double   pY,
//        G4double   pX,
//        G4double   pLTX)
//
// G4Trap(const G4String& pName,                         
//        G4double   pDz,   G4double   pTheta,
//        G4double   pPhi,  G4double   pDy1,
//        G4double   pDx1,  G4double   pDx2,
//        G4double   pAlp1, G4double   pDy2,
//        G4double   pDx3,  G4double   pDx4,
//        G4double   pAlp2)
void
doTrap( const std::string& name, double dz, double pTheta, double pPhi,
	double pDy1, double pDx1, double pDx2, double pAlp1,
	double pDy2, double pDx3, double pDx4, double pAlp2 )
{
  G4Trap g4( name, dz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2 );
  
  // Note, the order of parameters is different:
  DDI::Trap dd( dz, pTheta, pPhi, pDy2, pDx3, pDx4, pAlp1, pDy1, pDx1, pDx2, pAlp2 );
  DDTrap dds = DDSolidFactory::trap( name, dz, pTheta, pPhi, pDy2, pDx3, pDx4, pAlp1, pDy1, pDx1, pDx2, pAlp2 );
  dd.stream( std::cout );
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 7. Sphere or Spherical Shell Section:
//
// G4Sphere(const G4String& pName,                      
// 	    G4double   pRmin,
// 	    G4double   pRmax,
// 	    G4double   pSPhi,
// 	    G4double   pDPhi,
// 	    G4double   pSTheta,
// 	    G4double   pDTheta )
void
doSphere( const std::string& name, double innerRadius, double outerRadius, 
	  double startPhi, double deltaPhi, double startTheta, double deltaTheta )
{  
  G4Sphere g4(name,innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
  DDI::Sphere dd(innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
  DDSphere dds = DDSolidFactory::sphere(name, innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 8. Full Solid Sphere:
//
// G4Orb(const G4String& pName,                                   
//       G4double  pRmax)
void
doOrb( const std::string& name, double radius )
{  
  G4Orb g4(name,radius);
  DDI::Orb dd(radius);
  DDI::Sphere dds(0.*deg, radius, 0.*deg, 360.*deg, 0., 180.*deg);
  DDOrb ddo = DDSolidFactory::orb(name, radius);
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << ddo << " vol= " << ddo.volume() << std::endl;
  std::cout << "\tcross check sphere " << std::endl;
  dds.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tsphere volume = " << dds.volume()/cm3 << " cm3" << std::endl;
}

//
// 9. Torus:
//
// G4Torus(const G4String& pName,                       
// 	   G4double   pRmin,
// 	   G4double   pRmax,
// 	   G4double   pRtor,
// 	   G4double   pSPhi,
// 	   G4double   pDPhi)
void
doTorus( const std::string& name, double rMin, double rMax, double radius, double sPhi, double dPhi )
{  
  G4Torus g4( name, rMin, rMax, radius, sPhi, dPhi );
  DDI::Torus dd( rMin, rMax, radius, sPhi, dPhi );
  DDTorus dds = DDSolidFactory::torus( name, rMin, rMax, radius, sPhi, dPhi );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

// Specific CSG Solids
//

// 
// 10. Polycons:
//
// G4Polycone(const G4String& pName,                    
// 	   G4double   phiStart,
// 	   G4double   phiTotal,
// 	   G4int      numZPlanes,
// 	   const G4double   zPlane[],
// 	   const G4double   rInner[],
// 	   const G4double   rOuter[])
//
// G4Polycone(const G4String& pName,                      
// 	   G4double   phiStart,
// 	   G4double   phiTotal,
// 	   G4int      numRZ,
// 	   const G4double  r[],
// 	   const G4double  z[])
void
doPolycone1( const std::string& name, double phiStart, double phiTotal,
	     const std::vector<double> & z,
	     const std::vector<double> & rInner,
	     const std::vector<double> & rOuter )
{  
  G4Polycone g4( name, phiStart, phiTotal, z.size(), &z[0], &rInner[0], &rOuter[0] );
  DDI::Polycone dd( phiStart, phiTotal, z, rInner, rOuter );
  DDPolycone dds = DDSolidFactory::polycone( name, phiStart, phiTotal, z, rInner, rOuter );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

void
doPolycone2( const std::string& name, double phiStart, double phiTotal,
	     const std::vector<double> & z,
	     const std::vector<double> & r )
{  
  std::cout << "### doPolycone_RZ: " << "phi1=" << phiStart/deg 
  	    << " phi2=" << phiTotal/deg 
  	    << " N= " << z.size() << std::endl;
  for(size_t i=0; i<z.size(); ++i) { 
    std::cout << " R= " << r[i] << " Z= " << z[i] << std::endl;
  }
  G4Polycone g4( name, phiStart, phiTotal, z.size(), &r[0], &z[0] );
  DDI::Polycone dd( phiStart, phiTotal, z, r );
  DDPolycone dds = DDSolidFactory::polycone( name, phiStart, phiTotal, z, r );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 11. Polyhedra (PGON):
//
// G4Polyhedra(const G4String& pName,                   
// 	    G4double  phiStart,
// 	    G4double  phiTotal,
// 	    G4int     numSide,
// 	    G4int     numZPlanes,
// 	    const G4double  zPlane[],
// 	    const G4double  rInner[],
// 	    const G4double  rOuter[] )
//
// G4Polyhedra(const G4String& pName,
// 	    G4double  phiStart,
// 	    G4double  phiTotal,
// 	    G4int     numSide,
// 	    G4int     numRZ,
// 	    const G4double  r[],
// 	    const G4double  z[] )
void
doPolyhedra1( const std::string& name, int sides, double phiStart, double phiTotal,
	      const std::vector<double> & z,
	      const std::vector<double> & rInner,
	      const std::vector<double> & rOuter )
{  
  G4Polyhedra g4( name, phiStart, phiTotal, sides, z.size(), &z[0], &rInner[0], &rOuter[0] );
  DDI::Polyhedra dd( sides, phiStart, phiTotal, z, rInner, rOuter );
  DDPolyhedra dds = DDSolidFactory::polyhedra( name, sides, phiStart, phiTotal, z, rInner, rOuter );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

void
doPolyhedra2( const std::string& name, int sides, double phiStart, double phiTotal,
	      const std::vector<double> & z,
	      const std::vector<double> & r )
{  
  G4Polyhedra g4( name, phiStart, phiTotal, sides, z.size(), &r[0], &z[0] );
  DDI::Polyhedra dd( sides, phiStart, phiTotal, z, r );
  DDPolyhedra dds = DDSolidFactory::polyhedra( name, sides, phiStart, phiTotal, z, r );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

//
// 12. Tube with an elliptical cross section:
//
// G4EllipticalTube(const G4String& pName,                   
// 		 G4double  Dx,
// 		 G4double  Dy,
// 		 G4double  Dz)
void
doEllipticalTube( const std::string& name, double xSemiaxis, double ySemiAxis, double zHeight )
{  
  G4EllipticalTube g4t( name, xSemiaxis, ySemiAxis, zHeight );
  DDI::EllipticalTube ddt( xSemiaxis, ySemiAxis, zHeight );
  DDEllipticalTube ddet = DDSolidFactory::ellipticalTube( name, xSemiaxis, ySemiAxis, zHeight );
  ddt.stream( std::cout );
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4t.GetCubicVolume() / cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << ddt.volume() / cm3 << " cm3"<<  std::endl;
  std::cout << "\tcalc volume = " << 2 * zHeight * Geom::pi() * ySemiAxis*xSemiaxis / cm3 << " cm3 " <<std::endl;
  std::cout << "\tDD Information: ";
  std::cout << ddet << " vol= " << ddet.volume() << std::endl;
}

//
// 13. General Ellipsoid:
//
// G4Ellipsoid(const G4String& pName,                   
// 	       G4double  pxSemiAxis,
// 	       G4double  pySemiAxis,
// 	       G4double  pzSemiAxis,
// 	       G4double  pzBottomCut=0,
// 	       G4double  pzTopCut=0)
void
doEllipsoid( const std::string& name, double xSemiAxis, double ySemiAxis, 
	     double zSemiAxis, double zBottomCut, double zTopCut )
{  
  G4Ellipsoid g4(name,xSemiAxis,ySemiAxis,zSemiAxis,zBottomCut, zTopCut);
  DDI::Ellipsoid dd(xSemiAxis,ySemiAxis,zSemiAxis,zBottomCut, zTopCut);
  DDEllipsoid dde = DDSolidFactory::ellipsoid(name, xSemiAxis, ySemiAxis, zSemiAxis, zBottomCut, zTopCut);
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dde << " vol= " << dde.volume() << std::endl;
}

//
// 14. Cone with Elliptical Cross Section:
//
// G4EllipticalCone(const G4String& pName,              
// 		 G4double  pxSemiAxis,
// 		 G4double  pySemiAxis,
// 		 G4double  zMax,
// 		 G4double  pzTopCut)

//
// 15. Paraboloid, a solid with parabolic profile:
//
// G4Paraboloid(const G4String& pName,                   
// 	     G4double  Dz,
// 	     G4double  R1,
// 	     G4double  R2)

//
// 16. Tube with Hyperbolic Profile:
//
// G4Hype(const G4String& pName,                        
//        G4double  innerRadius,
//        G4double  outerRadius,
//        G4double  innerStereo,
//        G4double  outerStereo,
//        G4double  halfLenZ)

//
// 17. Tetrahedra:
//
// G4Tet(const G4String& pName,                         
//       G4ThreeVector  anchor,
//       G4ThreeVector  p2,
//       G4ThreeVector  p3,
//       G4ThreeVector  p4,
//       G4bool         *degeneracyFlag=0)

//
// 18. Extruded Polygon:
//
// G4ExtrudedSolid(const G4String&                pName,
// 		std::vector<G4TwoVector> polygon,
// 		std::vector<ZSection>    zsections)
//
// G4ExtrudedSolid(const G4String&                pName,
// 		std::vector<G4TwoVector> polygon,
// 		G4double                 hz,
// 		G4TwoVector off1, G4double scale1,
// 		G4TwoVector off2, G4double scale2)

//
// 19. Box Twisted:
//
// G4TwistedBox(const G4String& pName,                  
// 	     G4double  twistedangle,
// 	     G4double  pDx,
// 	     G4double  pDy,
// 	     G4double  pDz)

//
// 20. Trapezoid Twisted along One Axis:
//
// G4TwistedTrap(const G4String& pName,                 
// 	      G4double  twistedangle,
// 	      G4double  pDxx1, 
// 	      G4double  pDxx2,
// 	      G4double  pDy, 
// 	      G4double   pDz)
//  
// G4TwistedTrap(const G4String& pName,
// 	      G4double  twistedangle,
// 	      G4double  pDz,
// 	      G4double  pTheta, 
// 	      G4double  pPhi,
// 	      G4double  pDy1, 
// 	      G4double  pDx1,
// 	      G4double  pDx2, 
// 	      G4double  pDy2,
// 	      G4double  pDx3, 
// 	      G4double  pDx4,
// 	      G4double  pAlph)

//
// 21. Twisted Trapezoid with x and y dimensions varying along z:
//
// G4TwistedTrd(const G4String& pName,                  
// 	     G4double  pDx1,
// 	     G4double  pDx2,
// 	     G4double  pDy1,
// 	     G4double  pDy2,
// 	     G4double  pDz,
// 	     G4double  twistedangle)

//
// 22. Generic trapezoid with optionally collapsing vertices:
//
// G4GenericTrap(const G4String& pName,                  
// 	      G4double  pDz,
// 	      const std::vector<G4TwoVector>& vertices)

//
// 23. Tube Section Twisted along Its Axis: 
// 
// G4TwistedTubs(const G4String& pName,                 
// 	      G4double  twistedangle,
// 	      G4double  endinnerrad,
// 	      G4double  endouterrad,
// 	      G4double  halfzlen,
// 	      G4double  dphi)

int
main( int argc, char *argv[] )
{
  double xSemiaxis(2.*cm);
  double ySemiAxis(2.*cm);
  double zHeight(2.*cm);
  std::string name("fred1");

//
// 1. Box:
//
  std::cout << "\n\nBox tests\n" << std::endl;
  doBox( name, xSemiaxis, ySemiAxis, zHeight );
  std::cout << std::endl;
  
//
// 2. Cylindrical Section or Tube:
//
  std::cout << "\n\nTub tests\n" << std::endl;
  double rIn = 10.*cm;
  double rOut = 15.*cm;
  double zhalf = 20.*cm;
  double startPhi = 0.*deg;
  double deltaPhi = 90.*deg;
  doTubs( name, rIn, rOut, zhalf, startPhi, deltaPhi );
  std::cout << std::endl;

//
// 3. Cone or Conical section:
//
  std::cout << "\n\nCons tests\n" << std::endl;
  double rIn2 = 20.*cm;
  double rOut2 = 25.*cm;
  doCons( name, rIn, rOut, rIn2, rOut2, zhalf, startPhi, deltaPhi );
  std::cout << std::endl;

//
// 4. Parallelepiped:
//
  std::cout << "\n\nParallelepiped tests\n" << std::endl;
  std::cout << "This next should be the same as a xhalf=5cm, yhalf=6cm, zhalf=7cm, alpha=15deg, theta=30deg, phi=45deg" << std::endl;
  doPara("fred1", 5.*cm, 6.*cm, 7.*cm, 15*deg, 30*deg, 45*deg);

//
// 5. Trapezoid:
//
  std::cout << "\n\nTrapezoid tests\n" << std::endl;
  double dx1 = 10.*cm;
  double dx2 = 30.*cm;
  double dy1 = 15.*cm;
  double dy2 = 30.*cm;
  double dz = 60.*cm;
  doTrd( name, dx1, dx2, dy1, dy2, dz );
  std::cout << std::endl;

//
// 6. Generic Trapezoid:
//
  std::cout << "\n\nGeneric Trapezoid tests\n" << std::endl;
  double pTheta = 0.*deg;
  double pPhi = 0.*deg;
  double pDy1 = 30.*cm;
  double pDx1 = 30.*cm;
  double pDx2 = 30.*cm;
  double pAlp1 = 0.*deg;
  double pDy2 = 15.*cm;
  double pDx3 = 10.*cm;
  double pDx4 = 10.*cm;
  double pAlp2 = 0.*deg;
  doTrap( name, dz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2 );
  std::cout << std::endl;

//
// 7. Sphere or Spherical Shell Section:
//
  std::cout << "\n\nSphere tests\n" << std::endl;
  std::cout << "This next should be the same as a 2cm ball: " << std::endl;
  doSphere("fred1", 0.0*cm, 2.0*cm, 0.*deg, 360.*deg, 0., 180.*deg);
  std::cout << "Manual computation gives: " 
	    << 4./3. * Geom::pi() * 2.0*cm * 2.0*cm *2.0*cm / cm3
	    <<std::endl;
  std::cout << "If you mess up phi and theta you get: " << std::endl;
  doSphere("fred1", 0.0*cm, 2.0*cm, 0.*deg, 180.*deg, 0., 360.*deg);
  std::cout << "\n1 cm thick shell: " << std::endl;
  doSphere("fred1", 2.0*cm, 3.0*cm, 0.*deg, 360.*deg, 0., 180.*deg);
  std::cout << "Manual computation gives: "
	    << 4./3. * Geom::pi() * 3.0*cm * 3.0*cm *3.0*cm / cm3 - 4./3. * Geom::pi() * 2.0*cm * 2.0*cm *2.0*cm / cm3
	    <<std::endl;
  std::cout << "\nHALF of the above 1 cm thick shell: " << std::endl;
  doSphere("fred1", 2.0*cm, 3.0*cm, 0.*deg, 180.*deg, 0., 180.*deg);
  std::cout << "Manual computation gives: "
	    << (4./3. * Geom::pi() * 3.0*cm * 3.0*cm *3.0*cm / cm3 - 4./3. * Geom::pi() * 2.0*cm * 2.0*cm *2.0*cm / cm3) / 2.
	    <<std::endl;
  std::cout << "\n30 degree span in theta; full phi \"top\" hemisphere" << std::endl;
  doSphere("fred1", 2.0*cm, 3.0*cm, 0.*deg, 360.*deg, 10.*deg, 30.*deg);
  std::cout << "\n30 degree span in theta; full phi \"bottom\" hemisphere; mirror of above, so should be same." << std::endl;
  doSphere("fred1", 2.0*cm, 3.0*cm, 0.*deg, 360.*deg, 140.*deg, 30.*deg);
  std::cout << "\n30 degree span in theta; full phi around equator (should be bigger than above)" << std::endl;
  doSphere("fred1", 2.0*cm, 3.0*cm, 0.*deg, 360.*deg, 75.*deg, 30.*deg);

//
// 8. Full Solid Sphere:
//
  std::cout << "\n\nOrb\n" << std::endl;
  std::cout << "This next should be the same as a 2cm ball (also the sphere above): " << std::endl;
  doOrb("fred1", 2.0*cm);
  
//
// 9. Torus:
//
  std::cout << "\n\nTorus tests\n" << std::endl;
  double radius = 200.*cm;
  doTorus( name, rIn, rOut, radius, startPhi, deltaPhi );
  std::cout << std::endl;

// 
// 10. Polycons:
//
  std::cout << "\n\nPolycons tests\n" << std::endl;
  double phiStart = 45.*deg;
  double phiTotal = 325.*deg;
  double inner[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  std::vector<double> rInner( inner, inner + sizeof( inner ) / sizeof( double ));
  double outer[] = { 0, 10, 10, 5, 5, 10, 10, 2, 2 };
  std::vector<double> rOuter( outer, outer + sizeof( outer ) / sizeof( double ));
  double pl[] = { 5, 7, 9, 11, 25, 27, 29, 31, 35 };
  std::vector<double> z( pl, pl + sizeof( pl ) / sizeof( double ));
  doPolycone1( name, phiStart, phiTotal, z, rInner, rOuter );
  std::cout << std::endl;

  doPolycone2( name, phiStart, phiTotal, z, rOuter);
  std::cout << std::endl;

//
// 11. Polyhedra (PGON):
//
  std::cout << "\n\nPolyhedra tests\n" << std::endl;
  int sides = 3;
  doPolyhedra1( name, sides, phiStart, phiTotal, z, rInner, rOuter );
  std::cout << std::endl;

  doPolyhedra2( name, sides, phiStart, phiTotal, z, rOuter );
  std::cout << std::endl;

//
// 12. Tube with an elliptical cross section:
//  
  
  std::cout << "\n\nElliptical Tube tests\n" << std::endl;
  doEllipticalTube(name, xSemiaxis, ySemiAxis, zHeight);
  std::cout << std::endl;
  ySemiAxis = 3.*cm;
  doEllipticalTube(name, xSemiaxis, ySemiAxis, zHeight);
  std::cout << std::endl;
  xSemiaxis = 3.* cm;
  ySemiAxis = 2.* cm;
  zHeight = 10.* cm;
  doEllipticalTube(name, xSemiaxis, ySemiAxis, zHeight);
  std::cout << std::endl;
  xSemiaxis = 300.* cm;
  ySemiAxis = 400.* cm;
  zHeight = 3000. * cm;
  doEllipticalTube(name, xSemiaxis, ySemiAxis, zHeight);

//
// 13. General Ellipsoid:
//
  std::cout << "\n\nEllipsoid tests\n" << std::endl;
  std::cout << "This next should be the same as a x = 3cm; y = 2cm; and z = 5cm " << std::endl;
  doEllipsoid("fred1", 3.0*cm, 2.0*cm, 5.*cm, 0.*cm, 0.*cm);
  std::cout << "\nThis one has a top cut off at z=1cm  and should be half of the above + some bit." << std::endl;
  doEllipsoid("fred1", 3.0*cm, 2.0*cm, 5.*cm, 0.*cm, 1.*cm);
  std::cout << "\nThis has a bottom cut off at z= -1cm  and should be the same as the above (symmetric)" << std::endl;
  doEllipsoid("fred1", 3.0*cm, 2.0*cm, 5.*cm, -1.*cm, 0.*cm);
  std::cout << "\nThis has a bottom cut off at z= -1cm  and top cut at z=1cm and should be smaller (just the fat bit around the middle)." << std::endl;
  doEllipsoid("fred1", 3.0*cm, 2.0*cm, 5.*cm, -1.*cm, 1.*cm);

//
// 14. Cone with Elliptical Cross Section:
//

//
// 15. Paraboloid, a solid with parabolic profile:
//

//
// 16. Tube with Hyperbolic Profile:
//
  
//
// 17. Tetrahedra:
//

//
// 18. Extruded Polygon:
//

//
// 19. Box Twisted:
//

//
// 20. Trapezoid Twisted along One Axis:
//

//
// 21. Twisted Trapezoid with x and y dimensions varying along z:
//

//
// 22. Generic trapezoid with optionally collapsing vertices:
//

//
// 23. Tube Section Twisted along Its Axis: 
//   
  
  return EXIT_SUCCESS;
}

