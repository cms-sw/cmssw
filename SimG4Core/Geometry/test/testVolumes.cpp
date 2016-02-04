#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>
#include <DetectorDescription/Core/src/EllipticalTube.h>
#include <DetectorDescription/Core/src/Ellipsoid.h>
#include <DetectorDescription/Core/src/Sphere.h>
#include <DetectorDescription/Core/src/Parallelepiped.h>
#include <DetectorDescription/Core/src/Orb.h>
#include <DataFormats/GeometryVector/interface/Pi.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4EllipticalTube.hh>
#include <G4Ellipsoid.hh>
#include <G4Sphere.hh>
#include <G4Para.hh>
#include <G4Orb.hh>
#include <string>

void doEllipticalTube (const std::string& name, double xSemiaxis, double ySemiAxis, double zHeight) {
  
  G4EllipticalTube g4t(name,xSemiaxis, ySemiAxis, zHeight);
  DDI::EllipticalTube ddt(xSemiaxis, ySemiAxis, zHeight);
  DDEllipticalTube ddet =DDSolidFactory::ellipticalTube(name, xSemiaxis, ySemiAxis, zHeight);
  ddt.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4t.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << ddt.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tcalc volume = " << 2*zHeight*Geom::pi()*ySemiAxis*xSemiaxis / cm3 << " cm3 " <<std::endl;
  std::cout << "\tDD Information: ";
  std::cout << ddet << " vol= " << ddet.volume() << std::endl;
}

void doEllipsoid (const std::string& name, double xSemiAxis, double ySemiAxis, 
		  double zSemiAxis, double zBottomCut, double zTopCut ) {
  
  G4Ellipsoid g4(name,xSemiAxis,ySemiAxis,zSemiAxis,zBottomCut, zTopCut);
  DDI::Ellipsoid dd(xSemiAxis,ySemiAxis,zSemiAxis,zBottomCut, zTopCut);
  DDEllipsoid dde = DDSolidFactory::ellipsoid(name, xSemiAxis, ySemiAxis, zSemiAxis, zBottomCut, zTopCut);
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dde << " vol= " << dde.volume() << std::endl;
}

void doSphere (const std::string& name, double innerRadius, double outerRadius, 
	       double startPhi, double deltaPhi, double startTheta, double deltaTheta ) {
  
  G4Sphere g4(name,innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
  DDI::Sphere dd(innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
  DDSphere dds = DDSolidFactory::sphere(name, innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
}

void doOrb (const std::string& name, double radius) {
  
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

void doPara (const std::string& name, double xHalf, double yHalf, 
	       double zHalf, double alpha, double theta, double phi ) {

  G4Para g4(name,xHalf, yHalf, zHalf, alpha, theta, phi);
  DDI::Parallelepiped dd(xHalf, yHalf, zHalf, alpha, theta, phi);
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;

}

int main(int argc, char *argv[]) {

  std::cout << "\n\nElliptical Tube tests\n" << std::endl;
  double xSemiaxis(2.*cm);
  double ySemiAxis(2.*cm);
  double zHeight(2.*cm);
  std::string name("fred1");
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

  std::cout << "\n\nSphere tests\n" << std::endl;
  std::cout << "This next should be the same as a 2cm ball: " << std::endl;
  doSphere("fred1", 0.0*cm, 2.0*cm, 0.*deg, 360.*deg, 0., 180.*deg);
  std::cout << "Manual computation gives: " 
	    << 4./3. * Geom::pi() * 2.0*cm * 2.0*cm *2.0*cm / cm3
	    <<std::endl;
  std::cout << "If you mess up phi and theta you get: " << std::endl;
  doSphere("fred1", 0.0*cm, 2.0*cm, 0.*deg, 180.*deg, 0., 360.*deg);
  std::cout << "\n1 cm thick shell: " << std::endl;
  doSphere ("fred1", 2.0*cm, 3.0*cm, 0.*deg, 360.*deg, 0., 180.*deg);
  std::cout << "Manual computation gives: "
	    << 4./3. * Geom::pi() * 3.0*cm * 3.0*cm *3.0*cm / cm3 - 4./3. * Geom::pi() * 2.0*cm * 2.0*cm *2.0*cm / cm3
	    <<std::endl;
  std::cout << "\nHALF of the above 1 cm thick shell: " << std::endl;
  doSphere ("fred1", 2.0*cm, 3.0*cm, 0.*deg, 180.*deg, 0., 180.*deg);
  std::cout << "Manual computation gives: "
	    << (4./3. * Geom::pi() * 3.0*cm * 3.0*cm *3.0*cm / cm3 - 4./3. * Geom::pi() * 2.0*cm * 2.0*cm *2.0*cm / cm3) / 2.
	    <<std::endl;
  std::cout << "\n30 degree span in theta; full phi \"top\" hemisphere" << std::endl;
  doSphere ("fred1", 2.0*cm, 3.0*cm, 0.*deg, 360.*deg, 10.*deg, 30.*deg);
  std::cout << "\n30 degree span in theta; full phi \"bottom\" hemisphere; mirror of above, so should be same." << std::endl;
  doSphere ("fred1", 2.0*cm, 3.0*cm, 0.*deg, 360.*deg, 140.*deg, 30.*deg);
  std::cout << "\n30 degree span in theta; full phi around equator (should be bigger than above)" << std::endl;
  doSphere ("fred1", 2.0*cm, 3.0*cm, 0.*deg, 360.*deg, 75.*deg, 30.*deg);

  std::cout << "\n\nOrb\n" << std::endl;
  std::cout << "This next should be the same as a 2cm ball (also the sphere above): " << std::endl;
  doOrb("fred1", 2.0*cm);

  std::cout << "\n\nEllipsoid tests\n" << std::endl;
  std::cout << "This next should be the same as a x = 3cm; y = 2cm; and z = 5cm " << std::endl;
  doEllipsoid("fred1", 3.0*cm, 2.0*cm, 5.*cm, 0.*cm, 0.*cm);
  std::cout << "\nThis one has a top cut off at z=1cm  and should be half of the above + some bit." << std::endl;
  doEllipsoid("fred1", 3.0*cm, 2.0*cm, 5.*cm, 0.*cm, 1.*cm);
  std::cout << "\nThis has a bottom cut off at z= -1cm  and should be the same as the above (symmetric)" << std::endl;
  doEllipsoid("fred1", 3.0*cm, 2.0*cm, 5.*cm, -1.*cm, 0.*cm);
  std::cout << "\nThis has a bottom cut off at z= -1cm  and top cut at z=1cm and should be smaller (just the fat bit around the middle)." << std::endl;
  doEllipsoid("fred1", 3.0*cm, 2.0*cm, 5.*cm, -1.*cm, 1.*cm);

  std::cout << "\n\nParallelepiped tests\n" << std::endl;
  std::cout << "This next should be the same as a xhalf=5cm, yhalf=6cm, zhalf=7cm, alpha=15deg, theta=30deg, phi=45deg" << std::endl;
  doPara("fred1", 5.*cm, 6.*cm, 7.*cm, 15*deg, 30*deg, 45*deg);

  return EXIT_SUCCESS;
}

