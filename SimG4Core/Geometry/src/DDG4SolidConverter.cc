#include "SimG4Core/Geometry/interface/DDG4SolidConverter.h"
#include "G4VSolid.hh"

#include "DetectorDescription/Core/interface/DDSolid.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "G4SystemOfUnits.hh"

using namespace std;
const vector<double> * DDG4SolidConverter::par_ = 0; 

DDG4SolidConverter::DDG4SolidConverter()
{
   // could also be done 'dynamically' from outside 
   // would then need to have a 'register' method ...
  par_=0;
  convDispatch_[ddbox]            = DDG4SolidConverter::box; 
  convDispatch_[ddtubs]           = DDG4SolidConverter::tubs;
  convDispatch_[ddtrap]           = DDG4SolidConverter::trap;
  convDispatch_[ddcons]           = DDG4SolidConverter::cons;
  convDispatch_[ddpolycone_rrz]   = DDG4SolidConverter::polycone_rrz;
  convDispatch_[ddpolycone_rz]    = DDG4SolidConverter::polycone_rz;
  convDispatch_[ddpolyhedra_rrz]  = DDG4SolidConverter::polyhedra_rrz;
  convDispatch_[ddpolyhedra_rz]   = DDG4SolidConverter::polyhedra_rz;   
  convDispatch_[ddtorus]          = DDG4SolidConverter::torus;   
  convDispatch_[ddreflected]      = DDG4SolidConverter::reflected;
  convDispatch_[ddunion]          = DDG4SolidConverter::unionsolid;
  convDispatch_[ddintersection]   = DDG4SolidConverter::intersection;
  convDispatch_[ddsubtraction]    = DDG4SolidConverter::subtraction;
  convDispatch_[ddpseudotrap]     = DDG4SolidConverter::pseudotrap;
  convDispatch_[ddtrunctubs]      = DDG4SolidConverter::trunctubs;
  convDispatch_[ddsphere]         = DDG4SolidConverter::sphere;   
  convDispatch_[ddorb]            = DDG4SolidConverter::orb;   
  convDispatch_[ddellipticaltube] = DDG4SolidConverter::ellipticaltube;   
  convDispatch_[ddellipsoid]      = DDG4SolidConverter::ellipsoid;   
  convDispatch_[ddparallelepiped] = DDG4SolidConverter::para;   
}


DDG4SolidConverter::~DDG4SolidConverter() { }

G4VSolid * DDG4SolidConverter::convert(const DDSolid & s)
{
  if ( !s ) {
    edm::LogError("SimG4CoreGeometry") <<" DDG4SolidConverter::convert(..) found an undefined DDSolid " << s.toString();
    throw cms::Exception("SimG4CoreGeometry", "DDG4SolidConverter::convert(..) found an undefined DDSolid " + s.toString());
  }
   G4VSolid * result = 0;
   par_ = &(s.parameters());
   map<DDSolidShape,FNPTR>::iterator it = convDispatch_.find(s.shape());
   if (it != convDispatch_.end()) {
     result = it->second(s);
   } 
   else {
     throw cms::Exception("DetectorDescriptionFault") 
       <<  "DDG4SolidConverter::convert: conversion failed for s=" << s
       << "\n solid.shape()=" << s.shape()
       << std::endl;
   }
   return result;
}


#include "G4Box.hh"
G4VSolid * DDG4SolidConverter::box(const DDSolid & s)
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: box = " << s ;
   return new G4Box(s.name().name(), (*par_)[0],(*par_)[1],(*par_)[2]);
}


#include "G4Tubs.hh"
G4VSolid * DDG4SolidConverter::tubs(const DDSolid & s)
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: tubs = " << s ;  
   return new G4Tubs(s.name().name(), (*par_)[1], // rmin
                               (*par_)[2], // rmax
			       (*par_)[0], // dzHalf
			       (*par_)[3], // phiStart
			       (*par_)[4]);// deltaPhi
}


#include "G4Trap.hh"
G4VSolid * DDG4SolidConverter::trap(const DDSolid & s)
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: trap = " << s ;
   return new G4Trap(s.name().name(), (*par_)[0],  // pDz
                               (*par_)[1],  // theta
			       (*par_)[2],  // phi
			       (*par_)[3],  // y1
			       (*par_)[4],  // x1
			       (*par_)[5],  // x2
			       (*par_)[6],  // alpha1
			       (*par_)[7],  // y2
			       (*par_)[8],  // x3
			       (*par_)[9],  // x4
			       (*par_)[10]);// alpha2
}


#include "G4Cons.hh"
G4VSolid * DDG4SolidConverter::cons(const DDSolid & s) 
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: cons = " << s ;  
   return new G4Cons(s.name().name(), (*par_)[1],  // rmin -z
                               (*par_)[2],  // rmax -z
			       (*par_)[3],  // rmin +z
			       (*par_)[4],  // rmax +z
			       (*par_)[0],  // zHalf
			       (*par_)[5],  // phistart
			       (*par_)[6]); // deltaphi
}

	    
#include "G4Polycone.hh"
G4VSolid * DDG4SolidConverter::polycone_rz(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: pcon_rz = " << s ;  
  vector<double> r;
  vector<double> z;
  vector<double>::const_iterator i = (*par_).begin()+2;
  int count=0;
  for(; i!=(*par_).end(); ++i) {
    LogDebug("SimG4CoreGeometry") << "z=" << *i ;
    z.push_back(*i); ++i;
    LogDebug("SimG4CoreGeometry") << " r=" << *i ;
    r.push_back(*i);
    count++;
   }
   LogDebug("SimG4CoreGeometry") << "sp=" << (*par_)[0]/deg << " ep=" << (*par_)[1]/deg ;
   /*
   std::cout << "### Polycone_RZ: " << "sp=" << (*par_)[0]/deg 
	     << " ep=" << (*par_)[1]/deg 
	     << " N= " << count << std::endl;
   for(int i=0; i<count; ++i) { 
     std::cout << " R= " << r[i] << " Z= " << z[i] << std::endl;
   }
   */
   return new G4Polycone(s.name().name(), (*par_)[0], (*par_)[1], // start,delta-phi
			 count, // numRZ
			 &(r[0]),
			 &(z[0]));
}


G4VSolid * DDG4SolidConverter::polycone_rrz(const DDSolid & s)
{
    LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: pcon_rrz = " << s ;  
    vector<double> z_p;
    vector<double> rmin_p;
    vector<double> rmax_p;
    vector<double>::const_iterator i = par_->begin()+2;
    int count = 0;
    for (; i!=par_->end(); ++i) {
        LogDebug("SimG4CoreGeometry") << "z=" << *i ;
	z_p.push_back(*i); ++i;
        LogDebug("SimG4CoreGeometry") << "rmin=" << *i ;
	rmin_p.push_back(*i); ++i;
        LogDebug("SimG4CoreGeometry") << "rmax=" << *i ;
	rmax_p.push_back(*i); 
      count++;
    }
    LogDebug("SimG4CoreGeometry") << "sp=" << (*par_)[0]/deg << " ep=" << (*par_)[1]/deg ;
    /*
    std::cout << "### Polycone_RRZ: " << "sp=" << (*par_)[0]/deg 
	      << " ep=" << (*par_)[1]/deg 
	      << " N= " << count << std::endl;
    for(int i=0; i<count; ++i) { 
      std::cout << " R1= " << rmin_p[i] << " R1= " << rmax_p[i] << " Z= " << z_p[i] << std::endl;
    }
    */
    return new G4Polycone(s.name().name(), (*par_)[0], (*par_)[1], // start,delta-phi
                        count, // sections
			&(z_p[0]),
			&(rmin_p[0]),
			&(rmax_p[0]));
			
}


#include "G4Polyhedra.hh"
G4VSolid * DDG4SolidConverter::polyhedra_rz(const DDSolid & s)
{
    LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: phed_rz = " << s ;  
    vector<double> r;
    vector<double> z;
    vector<double>::const_iterator i = par_->begin()+3;
    int count=0;
    
    for(; i!=par_->end(); ++i) {
      z.push_back(*i); ++i;
      r.push_back(*i);
      count++;
    }
      
    return new G4Polyhedra(s.name().name(), (*par_)[1], (*par_)[2], int((*par_)[0]),// start,delta-phi;sides
			   count, // numRZ
			   &(r[0]),
			   &(z[0]));
}


G4VSolid * DDG4SolidConverter::polyhedra_rrz(const DDSolid & s)
{
    LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: phed_rrz = " << s ;  
    vector<double> z_p;
    vector<double> rmin_p;
    vector<double> rmax_p;
    vector<double>::const_iterator i = par_->begin()+3;
    int count = 0;
    for (; i!=par_->end(); ++i) {
        LogDebug("SimG4CoreGeometry") << "z=" << *i ;
	z_p.push_back(*i); ++i;
        LogDebug("SimG4CoreGeometry") << "rmin=" << *i ;
	rmin_p.push_back(*i); ++i;
        LogDebug("SimG4CoreGeometry") << "rmax=" << *i ;
	rmax_p.push_back(*i); 
      count++;
    }
    LogDebug("SimG4CoreGeometry") << "sp=" << (*par_)[0]/deg << " ep=" << (*par_)[1]/deg ;
    return new G4Polyhedra(s.name().name(), (*par_)[1], (*par_)[2], int((*par_)[0]), // start,delta-phi,sides
			   count, // sections
			   &(z_p[0]),
			   &(rmin_p[0]),
			   &(rmax_p[0]));  
}

#include "G4Torus.hh"
G4VSolid * DDG4SolidConverter::torus(const DDSolid & s)
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: torus = " << s ;  
   return new G4Torus(s.name().name(), (*par_)[0], // rmin
                               (*par_)[1], // rmax
			       (*par_)[2], // Rtor
			       (*par_)[3], // phiStart
			       (*par_)[4]);// deltaPhi
}


#include "G4ReflectedSolid.hh"

namespace {
  static const HepGeom::ReflectZ3D z_reflection;
}

G4VSolid * DDG4SolidConverter::reflected(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: reflected = " << s ;  
  G4ReflectedSolid * rs = 0;
  DDReflectionSolid rfs(s); 
  if (rfs) {	
    rs = new G4ReflectedSolid(s.name().name(), 
                              DDG4SolidConverter().convert(rfs.unreflected()), 
			      z_reflection);
    
  } // else ?
  return rs;
}


#include "G4UnionSolid.hh"
G4VSolid * DDG4SolidConverter::unionsolid(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: unionsolid = " << s.name() ;
  G4UnionSolid * us = 0;
  DDBooleanSolid bs(s);
  if (bs) {
    LogDebug("SimG4CoreGeometry") << "SolidA=" << bs.solidA();
    G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
    LogDebug("SimG4CoreGeometry") << "SolidB=" << bs.solidB();
    G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
    LogDebug("SimG4CoreGeometry") << " name:" << s.name() << " t=" << bs.translation() << flush;
    LogDebug("SimG4CoreGeometry") << " " << bs.rotation().rotation()->Inverse() << flush;
    std::vector<double> tdbl(9);
    bs.rotation().rotation()->Inverse().GetComponents(tdbl.begin(), tdbl.end());
    CLHEP::HepRep3x3 temprep(tdbl[0], tdbl[1], tdbl[2], tdbl[3], tdbl[4], tdbl[5], tdbl[6], tdbl[7], tdbl[8]);
    CLHEP::Hep3Vector temphvec(bs.translation().X(), bs.translation().Y(), bs.translation().Z()); 
    us = new G4UnionSolid(s.name().name(),
			  sa,
			  sb,
			  new CLHEP::HepRotation(temprep),
			  temphvec);
    
  } // else?
  return us;	   
}


#include "G4SubtractionSolid.hh"
#include <sstream>
G4VSolid * DDG4SolidConverter::subtraction(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: subtraction = " << s ;
  G4SubtractionSolid * us = 0;
  DDBooleanSolid bs(s);
  if (bs) {
    G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
    G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
    LogDebug("SimG4CoreGeometry") << " name:" << s.name() << " t=" << bs.translation() << flush;
    //       stringstream sst;
    //       bs.rotation().rotation()->inverse().print(sst);
    //       LogDebug("SimG4CoreGeometry") << " " << sst.str() << flush;
    LogDebug("SimG4CoreGeometry") << " " << bs.rotation().rotation()->Inverse() << flush;
    std::vector<double> tdbl(9);
    bs.rotation().rotation()->Inverse().GetComponents(tdbl.begin(), tdbl.end());
    CLHEP::HepRep3x3 temprep(tdbl[0], tdbl[1], tdbl[2], tdbl[3], tdbl[4], tdbl[5], tdbl[6], tdbl[7], tdbl[8]);
    CLHEP::Hep3Vector temphvec(bs.translation().X(), bs.translation().Y(), bs.translation().Z()); 
    us = new G4SubtractionSolid(s.name().name(),
				sa,
				sb,
				new CLHEP::HepRotation(temprep),
				temphvec);
  }
  return us;	   
}


#include "G4IntersectionSolid.hh"
G4VSolid * DDG4SolidConverter::intersection(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: intersection = " << s ;
  G4IntersectionSolid * us = 0;
  DDBooleanSolid bs(s);
  if (bs) {
    G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
    G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
    LogDebug("SimG4CoreGeometry") << " name:" << s.name() << " t=" << bs.translation() << flush;
    LogDebug("SimG4CoreGeometry") << " " << bs.rotation().rotation()->Inverse() << flush;
    std::vector<double> tdbl(9);
    bs.rotation().rotation()->Inverse().GetComponents(tdbl.begin(), tdbl.end());
    CLHEP::HepRep3x3 temprep(tdbl[0], tdbl[1], tdbl[2], tdbl[3], tdbl[4], tdbl[5], tdbl[6], tdbl[7], tdbl[8]);
    CLHEP::Hep3Vector temphvec(bs.translation().X(), bs.translation().Y(), bs.translation().Z()); 
    us = new G4IntersectionSolid(s.name().name(),
				 sa,
				 sb,
				 new CLHEP::HepRotation(temprep),
				 temphvec);
  }
  return us;	   
}


#include "G4Trd.hh"
G4VSolid * DDG4SolidConverter::pseudotrap(const DDSolid & s)
{
  static G4RotationMatrix * rot = 0;
  static bool firstTime=true;
  if (firstTime) {
    firstTime=false;
    rot = new G4RotationMatrix;
    rot->rotateX(90.*deg);
    
  }
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: pseudoTrap = " << s ;
  G4Trd * trap = 0;
  G4Tubs * tubs = 0;
  G4VSolid * result = 0;
  DDPseudoTrap pt(s); // pt...PseudoTrap
  double r = pt.radius();
  bool atMinusZ = pt.atMinusZ();
  double x = 0;
  double h = 0;
  bool intersec = false; // union or intersection solid
  if (pt.atMinusZ()) {
    x = pt.x1(); // tubs radius
  } 
  else {
    x = pt.x2(); // tubs radius
  }
  double openingAngle = 2.*asin(x/abs(r));
  //trap = new G4Trd(s.name().name(), 
  double displacement=0;
  double startPhi=0;
  /* calculate the displacement of the tubs w.r.t. to the trap,
     determine the opening angle of the tubs */
  double delta = sqrt(r*r-x*x);
  if (r < 0 && abs(r) >= x) {
    intersec = true; // intersection solid
    h = pt.y1() < pt.y2() ? pt.y2() : pt.y1(); // tubs half height
    h += h/20.; // enlarge a bit - for subtraction solid
    if (atMinusZ) {
      displacement = - pt.halfZ() - delta; 
      startPhi = 270.*deg - openingAngle/2.;
    }
    else {
      displacement =   pt.halfZ() + delta;
      startPhi = 90.*deg - openingAngle/2.;
    }
  }
  else if ( r > 0 && abs(r) >= x )
    {
      if (atMinusZ) {
        displacement = - pt.halfZ() + delta;
	startPhi = 90.*deg - openingAngle/2.;
	h = pt.y1();
      }
      else {
        displacement =   pt.halfZ() - delta; 
	startPhi = 270.*deg - openingAngle/2.;
	h = pt.y2();
      }    
    }
  else {
    throw cms::Exception("DetectorDescriptionFault", "Check parameters of the PseudoTrap! name=" + pt.name().name());
   }
  G4ThreeVector displ(0.,0.,displacement); // displacement of the tubs w.r.t. trap
  LogDebug("SimG4CoreGeometry") << "DDSolidConverter::pseudotrap(): displacement=" << displacement 
				<< " openingAngle=" << openingAngle/deg << " x=" << x << " h=" << h;
    
  // Now create two solids (trd & tubs), and a boolean solid out of them 
  string name=pt.name().name();
  trap = new G4Trd(name, pt.x1(), pt.x2(), pt.y1(), pt.y2(), pt.halfZ());
  tubs = new G4Tubs(name, 
		    0., // rMin
		    abs(r), // rMax
		    h, // half height
		    startPhi, // start angle
		    openingAngle);
  if (intersec) {
    result = new G4SubtractionSolid(name, trap, tubs, rot, displ);
  }
  else {
    /** correct implementation, but fails to visualize due to G4/Iguana limitations */
    G4VSolid * tubicCap = new G4SubtractionSolid(name, 
						 tubs, 
						 new G4Box(name, 1.1*x, sqrt(r*r-x*x), 1.1*h),  
						 0, 
						 G4ThreeVector());
    result = new G4UnionSolid(name, trap, tubicCap, rot, displ);
            
    // approximative implementation - also fails to visualize due to G4/Iguana limitations
    /*
      delete tubs;
      tubs = new G4Tubs(name, 
      sqrt(r*r-x*x), // rMin-approximation!
      abs(r), // rMax
      h, // half height
      startPhi, // start angle
      openingAngle);
      result = new G4UnionSolid(name, trap, tubs, rot, displ);
    */
  }			 	   
  return result;
}


G4VSolid * DDG4SolidConverter::trunctubs(const DDSolid & s)
{
  // truncated tube-section: a boolean subtraction solid:
  //                         from a tube-section a box is subtracted according to the  
  //                         given parameters
  LogDebug("SimG4CoreGeometry") << "MantisConverter: solidshape=" << DDSolidShapesName::name(s.shape()) << " " << s;
  LogDebug("SimG4CoreGeometry") << "before";
  DDTruncTubs tt(s);
  LogDebug("SimG4CoreGeometry") << "after";
  double rIn(tt.rIn()), rOut(tt.rOut()), zHalf(tt.zHalf()),
    startPhi(tt.startPhi()), deltaPhi(tt.deltaPhi()), 
    cutAtStart(tt.cutAtStart()), cutAtDelta(tt.cutAtDelta());
  bool cutInside(bool(tt.cutInside()));
  string name=tt.name().name();

  // check the parameters
  if (rIn <= 0 || rOut <=0 || cutAtStart <=0 || cutAtDelta <= 0) {
    throw cms::Exception("DetectorDescriptionFault", "TruncTubs " + string(tt.name().fullname()) + ": 0 <= rIn,cutAtStart,rOut,cutAtDelta,rOut violated!");
  }
  if (rIn >= rOut) {
    throw cms::Exception("DetectorDescriptionFault", "TruncTubs " + string(tt.name().fullname()) + ": rIn<rOut violated!");
  }
  if (startPhi != 0.) {
    throw cms::Exception("DetectorDescriptionFault", "TruncTubs " + string(tt.name().fullname()) + ": startPhi != 0 not supported!");
  }
  //     if (cutInside != false) {
  //       throw cms::Exception("DetectorDescriptionFault", "TruncTubs " + string(tt.name()) + " cutInside == true not supported!");
  //     }

  startPhi=0.;
  double r(cutAtStart), R(cutAtDelta);
  G4VSolid * result(0);
  G4VSolid * tubs = new G4Tubs(name,rIn,rOut,zHalf,startPhi,deltaPhi);
  LogDebug("SimG4CoreGeometry") << "G4Tubs: " << rIn << ' ' << rOut << ' ' << zHalf << ' ' << startPhi/deg << ' ' << deltaPhi/deg;
  LogDebug("SimG4CoreGeometry") << s;
  // length & hight of the box 
  double boxX(30.*rOut), boxY(20.*rOut); // exaggerate dimensions - does not matter, it's subtracted!
   
  // width of the box > width of the tubs
  double boxZ(1.1*zHalf);
   
  // angle of the box w.r.t. tubs
  double cath = r-R*cos(deltaPhi);
  double hypo = sqrt(r*r+R*R-2.*r*R*cos(deltaPhi));
  double cos_alpha = cath/hypo;

  double alpha = -acos(cos_alpha);
  LogDebug("SimG4CoreGeometry") << "cath=" << cath/m;
  LogDebug("SimG4CoreGeometry") << "hypo=" << hypo/m;
  LogDebug("SimG4CoreGeometry") << "al=" << acos(cath/hypo)/deg;
  LogDebug("SimG4CoreGeometry") << "deltaPhi=" << deltaPhi/deg << "\n"
				<< "r=" << r/m << "\n"
				<<  "R=" << R/m;

  LogDebug("SimG4CoreGeometry") << "alpha=" << alpha/deg;
    
  // rotationmatrix of box w.r.t. tubs
  G4RotationMatrix * rot = new G4RotationMatrix;
  rot->rotateZ(-alpha);
  LogDebug("SimG4CoreGeometry") << (*rot);

  // center point of the box
  double xBox;
  if (!cutInside) {
    xBox = r+boxY/sin(abs(alpha));
  } else {
    xBox = -(boxY/sin(abs(alpha))-r);
  }

  G4ThreeVector trans(xBox,0.,0.);
  LogDebug("SimG4CoreGeometry") << "trans=" << trans;

  G4VSolid * box = new G4Box(name,boxX,boxY,boxZ);
  result = new G4SubtractionSolid(name,tubs,box,rot,trans);
      
  return result;

}

#include "G4Sphere.hh"
G4VSolid * DDG4SolidConverter::sphere(const DDSolid & s) 
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: sphere = " << s ;  
   DDSphere sp(s);
   return new G4Sphere(s.name().name(), sp.innerRadius(),
		       sp.outerRadius(),
		       sp.startPhi(),
		       sp.deltaPhi(),
		       sp.startTheta(),
		       sp.deltaTheta());
}

#include "G4Orb.hh"
G4VSolid * DDG4SolidConverter::orb(const DDSolid & s) 
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: orb = " << s ;  
   DDOrb sp(s);
   return new G4Orb(s.name().name(), sp.radius());
}

#include "G4EllipticalTube.hh"
G4VSolid * DDG4SolidConverter::ellipticaltube(const DDSolid & s) 
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: ellipticaltube = " << s ;  
   DDEllipticalTube sp(s);
   return new G4EllipticalTube(s.name().name(),
			       sp.xSemiAxis(),
                               sp.ySemiAxis(),
			       sp.zHeight());
}

#include "G4Ellipsoid.hh"
G4VSolid * DDG4SolidConverter::ellipsoid(const DDSolid & s) 
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: ellipsoid = " << s ;  
   DDEllipsoid sp(s);
   return new G4Ellipsoid(s.name().name(),
			  sp.xSemiAxis(),
			  sp.ySemiAxis(),
			  sp.zSemiAxis(),
			  sp.zBottomCut(),
			  sp.zTopCut());
}

#include "G4Para.hh"
G4VSolid * DDG4SolidConverter::para(const DDSolid & s) 
{
   LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: parallelepiped = " << s ;  
   DDParallelepiped sp(s);
   return new G4Para(s.name().name(),
		     sp.xHalf(),
		     sp.yHalf(),
		     sp.zHalf(),
		     sp.alpha(),
		     sp.theta(),
		     sp.phi());
}

