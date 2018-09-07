#include "SimG4Core/Geometry/interface/DDG4SolidConverter.h"
#include "G4VSolid.hh"

#include "DetectorDescription/Core/interface/DDSolid.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "G4SystemOfUnits.hh"

const std::vector<double> * DDG4SolidConverter::par_ = nullptr; 
G4RotationMatrix* DDG4SolidConverter::rot = nullptr;

DDG4SolidConverter::DDG4SolidConverter() {
  // could also be done 'dynamically' from outside 
  // would then need to have a 'register' method ...
  convDispatch_[DDSolidShape::ddbox]            = DDG4SolidConverter::box; 
  convDispatch_[DDSolidShape::ddtubs]           = DDG4SolidConverter::tubs;
  convDispatch_[DDSolidShape::ddcuttubs]        = DDG4SolidConverter::cuttubs;
  convDispatch_[DDSolidShape::ddtrap]           = DDG4SolidConverter::trap;
  convDispatch_[DDSolidShape::ddcons]           = DDG4SolidConverter::cons;
  convDispatch_[DDSolidShape::ddpolycone_rrz]   = DDG4SolidConverter::polycone_rrz;
  convDispatch_[DDSolidShape::ddpolycone_rz]    = DDG4SolidConverter::polycone_rz;
  convDispatch_[DDSolidShape::ddpolyhedra_rrz]  = DDG4SolidConverter::polyhedra_rrz;
  convDispatch_[DDSolidShape::ddpolyhedra_rz]   = DDG4SolidConverter::polyhedra_rz;
  convDispatch_[DDSolidShape::ddextrudedpolygon]= DDG4SolidConverter::extrudedpolygon;
  convDispatch_[DDSolidShape::ddtorus]          = DDG4SolidConverter::torus;   
  convDispatch_[DDSolidShape::ddunion]          = DDG4SolidConverter::unionsolid;
  convDispatch_[DDSolidShape::ddintersection]   = DDG4SolidConverter::intersection;
  convDispatch_[DDSolidShape::ddsubtraction]    = DDG4SolidConverter::subtraction;
  convDispatch_[DDSolidShape::ddpseudotrap]     = DDG4SolidConverter::pseudotrap;
  convDispatch_[DDSolidShape::ddtrunctubs]      = DDG4SolidConverter::trunctubs;
  convDispatch_[DDSolidShape::ddsphere]         = DDG4SolidConverter::sphere;   
  convDispatch_[DDSolidShape::ddellipticaltube] = DDG4SolidConverter::ellipticaltube;   
}

DDG4SolidConverter::~DDG4SolidConverter() {}

G4VSolid * DDG4SolidConverter::convert(const DDSolid & solid) {
  if ( !solid ) {
    edm::LogError("SimG4CoreGeometry") <<" DDG4SolidConverter::convert(..) found an undefined DDSolid " << solid.toString();
    throw cms::Exception("SimG4CoreGeometry", "DDG4SolidConverter::convert(..) found an undefined DDSolid " + solid.toString());
  }
  G4VSolid * result = nullptr;
  par_ = &(solid.parameters());
  std::map<DDSolidShape,FNPTR>::iterator it = convDispatch_.find(solid.shape());
  if (it != convDispatch_.end()) {
    result = it->second(solid);
  } else {
    throw cms::Exception("DetectorDescriptionFault") 
      <<  "DDG4SolidConverter::convert: conversion failed for s=" << solid
      << "\n solid.shape()=" << DDSolidShapesName::name(solid.shape())
      << std::endl;
  }
  return result;
}


#include "G4Box.hh"
G4VSolid * DDG4SolidConverter::box(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: box = " << solid;
  return new G4Box(solid.name().name(), (*par_)[0],(*par_)[1],(*par_)[2]);
}


#include "G4Tubs.hh"
G4VSolid * DDG4SolidConverter::tubs(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: tubs = " << solid;  
  return new G4Tubs(solid.name().name(), (*par_)[1], // rmin
		    (*par_)[2], // rmax
		    (*par_)[0], // dzHalf
		    (*par_)[3], // phiStart
		    (*par_)[4]);// deltaPhi
}

#include "G4CutTubs.hh"
G4VSolid * DDG4SolidConverter::cuttubs(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: tubs = " << solid;  
  return new G4CutTubs(solid.name().name(), (*par_)[1], // rmin
		       (*par_)[2], // rmax
		       (*par_)[0], // dzHalf
		       (*par_)[3], // phiStart
		       (*par_)[4], // deltaPhi
		       G4ThreeVector((*par_)[5],(*par_)[6],(*par_)[7]),
		       G4ThreeVector((*par_)[8],(*par_)[9],(*par_)[10]));
}


#include "G4Trap.hh"
G4VSolid * DDG4SolidConverter::trap(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: trap = " << solid;
  return new G4Trap(solid.name().name(), (*par_)[0],  // pDz
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
G4VSolid * DDG4SolidConverter::cons(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: cons = " << solid;  
  return new G4Cons(solid.name().name(), (*par_)[1],  // rmin -z
		    (*par_)[2],  // rmax -z
		    (*par_)[3],  // rmin +z
		    (*par_)[4],  // rmax +z
		    (*par_)[0],  // zHalf
		    (*par_)[5],  // phistart
		    (*par_)[6]); // deltaphi
}

	    
#include "G4Polycone.hh"
G4VSolid * DDG4SolidConverter::polycone_rz(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: pcon_rz = " << solid;  
  std::vector<double> r;
  std::vector<double> z;
  std::vector<double>::const_iterator i = (*par_).begin()+2;
  int count=0;
  for(; i!=(*par_).end(); ++i) {
    LogDebug("SimG4CoreGeometry") << " z=" << *i/CLHEP::cm;
    z.push_back(*i); ++i;
    LogDebug("SimG4CoreGeometry") << " r=" << *i/CLHEP::cm;
    r.push_back(*i);
    count++;
  }
  LogDebug("SimG4CoreGeometry") << "sp=" << (*par_)[0]/CLHEP::deg << " ep=" 
				<< (*par_)[1]/CLHEP::deg;
  /*
   std::cout << "### Polycone_RZ: " << "sp=" << (*par_)[0]/CLHEP::deg 
   << " ep=" << (*par_)[1]/CLHEP::deg 
   << " N= " << count << std::endl;
   for(int i=0; i<count; ++i) { 
   std::cout << " R= " << r[i] << " Z= " << z[i] << std::endl;
   }
  */
  return new G4Polycone(solid.name().name(), (*par_)[0], (*par_)[1], // start,delta-phi
			count, // numRZ
			&(r[0]),
			&(z[0]));
}


G4VSolid * DDG4SolidConverter::polycone_rrz(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: pcon_rrz = " << solid;
  std::vector<double> z_p;
  std::vector<double> rmin_p;
  std::vector<double> rmax_p;
  std::vector<double>::const_iterator i = par_->begin()+2;
  int count = 0;
  for (; i!=par_->end(); ++i) {
    LogDebug("SimG4CoreGeometry") << "z=" << *i/CLHEP::cm;
    z_p.push_back(*i); ++i;
    LogDebug("SimG4CoreGeometry") << "rmin=" << *i/CLHEP::cm;
    rmin_p.push_back(*i); ++i;
    LogDebug("SimG4CoreGeometry") << "rmax=" << *i/CLHEP::cm;
    rmax_p.push_back(*i); 
    count++;
  }
  LogDebug("SimG4CoreGeometry") << "sp=" << (*par_)[0]/CLHEP::deg << " ep=" 
				<< (*par_)[1]/CLHEP::deg;
  /*
    std::cout << "### Polycone_RRZ: " << "sp=" << (*par_)[0]/CLHEP::deg 
	      << " ep=" << (*par_)[1]/CLHEP::deg 
	      << " N= " << count << std::endl;
    for(int i=0; i<count; ++i) { 
      std::cout << " R1= " << rmin_p[i] << " R1= " << rmax_p[i] << " Z= " << z_p[i] << std::endl;
    }
  */
  return new G4Polycone(solid.name().name(), (*par_)[0], (*par_)[1], // start,delta-phi
                        count, // sections
			&(z_p[0]),
			&(rmin_p[0]),
			&(rmax_p[0]));
  
}


#include "G4Polyhedra.hh"
G4VSolid * DDG4SolidConverter::polyhedra_rz(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: phed_rz = " << solid;  
  std::vector<double> r;
  std::vector<double> z;
  std::vector<double>::const_iterator i = par_->begin()+3;
  int count=0;
    
  for(; i!=par_->end(); ++i) {
    z.push_back(*i); ++i;
    r.push_back(*i);
    count++;
  }
      
  return new G4Polyhedra(solid.name().name(), (*par_)[1], (*par_)[2], int((*par_)[0]),// start,delta-phi;sides
			 count, // numRZ
			 &(r[0]),
			 &(z[0]));
}


G4VSolid * DDG4SolidConverter::polyhedra_rrz(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: phed_rrz = " << solid;
  std::vector<double> z_p;
  std::vector<double> rmin_p;
  std::vector<double> rmax_p;
  std::vector<double>::const_iterator i = par_->begin()+3;
  int count = 0;
  for (; i!=par_->end(); ++i) {
    LogDebug("SimG4CoreGeometry") << "z=" << *i/CLHEP::cm;
    z_p.push_back(*i); ++i;
    LogDebug("SimG4CoreGeometry") << "rmin=" << *i/CLHEP::cm;
    rmin_p.push_back(*i); ++i;
    LogDebug("SimG4CoreGeometry") << "rmax=" << *i/CLHEP::cm;
    rmax_p.push_back(*i); 
    count++;
  }
  LogDebug("SimG4CoreGeometry") << "sp=" << (*par_)[0]/CLHEP::deg << " ep=" 
				<< (*par_)[1]/CLHEP::deg;
  return new G4Polyhedra(solid.name().name(), (*par_)[1], (*par_)[2], int((*par_)[0]), // start,delta-phi,sides
			 count, // sections
			 &(z_p[0]),
			 &(rmin_p[0]),
			 &(rmax_p[0]));  
}

#include "G4ExtrudedSolid.hh"
G4VSolid * DDG4SolidConverter::extrudedpolygon(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: extr_pgon = " << solid;
  std::vector<double> x = static_cast<DDExtrudedPolygon>(solid).xVec();
  std::vector<double> y = static_cast<DDExtrudedPolygon>(solid).yVec();
  std::vector<double> z = static_cast<DDExtrudedPolygon>(solid).zVec();
  std::vector<double> zx = static_cast<DDExtrudedPolygon>(solid).zxVec();
  std::vector<double> zy = static_cast<DDExtrudedPolygon>(solid).zyVec();
  std::vector<double> zs = static_cast<DDExtrudedPolygon>(solid).zscaleVec();

  std::vector<G4TwoVector> polygon;
  std::vector<G4ExtrudedSolid::ZSection> zsections;
  for( unsigned int it = 0; it < x.size(); ++it )
    polygon.emplace_back( x[it], y[it] );
  for( unsigned int it = 0; it < z.size(); ++it )
    zsections.emplace_back( z[it], G4TwoVector(zx[it], zy[it]), zs[it] );
  return new G4ExtrudedSolid( solid.name().name(), polygon, zsections );
}

#include "G4Torus.hh"
G4VSolid * DDG4SolidConverter::torus(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: torus = " << solid; 
  return new G4Torus(solid.name().name(), (*par_)[0], // rmin
		     (*par_)[1], // rmax
		     (*par_)[2], // Rtor
		     (*par_)[3], // phiStart
		     (*par_)[4]);// deltaPhi
}

#include "G4UnionSolid.hh"
G4VSolid * DDG4SolidConverter::unionsolid(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: unionsolid = " << solid.name();
  G4UnionSolid * us = nullptr;
  DDBooleanSolid bs(solid);
  if (bs) {
    LogDebug("SimG4CoreGeometry") << "SolidA=" << bs.solidA();
    G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
    LogDebug("SimG4CoreGeometry") << "SolidB=" << bs.solidB();
    G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
    LogDebug("SimG4CoreGeometry") << " name:" << solid.name() << " t=" << bs.translation() << std::flush;
    LogDebug("SimG4CoreGeometry") << " " << bs.rotation().rotation().Inverse() << std::flush;
    std::vector<double> tdbl(9);
    bs.rotation().rotation().Inverse().GetComponents(tdbl.begin(), tdbl.end());
    CLHEP::HepRep3x3 temprep(tdbl[0], tdbl[1], tdbl[2], tdbl[3], tdbl[4], tdbl[5], tdbl[6], tdbl[7], tdbl[8]);
    CLHEP::Hep3Vector temphvec(bs.translation().X(), bs.translation().Y(), bs.translation().Z()); 
    us = new G4UnionSolid(solid.name().name(),
			  sa,
			  sb,
			  new CLHEP::HepRotation(temprep),
			  temphvec);
    
  } // else?
  return us;	   
}

#include "G4SubtractionSolid.hh"
#include <sstream>
G4VSolid * DDG4SolidConverter::subtraction(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: subtraction = " << solid;
  G4SubtractionSolid * us = nullptr;
  DDBooleanSolid bs(solid);
  if (bs) {
    G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
    G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
    LogDebug("SimG4CoreGeometry") << " name:" << solid.name() << " t=" << bs.translation() << std::flush;
    LogDebug("SimG4CoreGeometry") << " " << bs.rotation().rotation().Inverse() << std::flush;
    std::vector<double> tdbl(9);
    bs.rotation().rotation().Inverse().GetComponents(tdbl.begin(), tdbl.end());
    CLHEP::HepRep3x3 temprep(tdbl[0], tdbl[1], tdbl[2], tdbl[3], tdbl[4], tdbl[5], tdbl[6], tdbl[7], tdbl[8]);
    CLHEP::Hep3Vector temphvec(bs.translation().X(), bs.translation().Y(), bs.translation().Z()); 
    us = new G4SubtractionSolid(solid.name().name(),
				sa,
				sb,
				new CLHEP::HepRotation(temprep),
				temphvec);
  }
  return us;	   
}

#include "G4IntersectionSolid.hh"
G4VSolid * DDG4SolidConverter::intersection(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: intersection = " << solid;
  G4IntersectionSolid * us = nullptr;
  DDBooleanSolid bs(solid);
  if (bs) {
    G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
    G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
    LogDebug("SimG4CoreGeometry") << " name:" << solid.name() << " t=" << bs.translation() << std::flush;
    LogDebug("SimG4CoreGeometry") << " " << bs.rotation().rotation().Inverse() << std::flush;
    std::vector<double> tdbl(9);
    bs.rotation().rotation().Inverse().GetComponents(tdbl.begin(), tdbl.end());
    CLHEP::HepRep3x3 temprep(tdbl[0], tdbl[1], tdbl[2], tdbl[3], tdbl[4], tdbl[5], tdbl[6], tdbl[7], tdbl[8]);
    CLHEP::Hep3Vector temphvec(bs.translation().X(), bs.translation().Y(), bs.translation().Z()); 
    us = new G4IntersectionSolid(solid.name().name(),
				 sa,
				 sb,
				 new CLHEP::HepRotation(temprep),
				 temphvec);
  }
  return us;	   
}

#include "G4Trd.hh"
G4VSolid * DDG4SolidConverter::pseudotrap(const DDSolid & solid) {
  if(nullptr == rot) {
    rot = new G4RotationMatrix;
    rot->rotateX(90.*deg);
  }    

  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: pseudoTrap = " << solid;
  G4Trd * trap = nullptr;
  G4Tubs * tubs = nullptr;
  G4VSolid * result = nullptr;
  DDPseudoTrap pt(solid); // pt...PseudoTrap
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
  double openingAngle = 2.*asin(x/std::abs(r));
  //trap = new G4Trd(solid.name().name(), 
  double displacement=0;
  double startPhi=0;
  /* calculate the displacement of the tubs w.r.t. to the trap,
     determine the opening angle of the tubs */
  double delta = sqrt(r*r-x*x);
  if (r < 0 && std::abs(r) >= x) {
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
  else if ( r > 0 && std::abs(r) >= x )
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
  std::string name=pt.name().name();
  trap = new G4Trd(name, pt.x1(), pt.x2(), pt.y1(), pt.y2(), pt.halfZ());
  tubs = new G4Tubs(name, 
		    0., // rMin
		    std::abs(r), // rMax
		    h, // half height
		    startPhi, // start angle
		    openingAngle);
  if (intersec) {
    result = new G4SubtractionSolid(name, trap, tubs, rot, displ);
  }
  else {
    G4VSolid * tubicCap = new G4SubtractionSolid(name, 
						 tubs, 
						 new G4Box(name, 1.1*x, sqrt(r*r-x*x), 1.1*h),  
						 nullptr, 
						 G4ThreeVector());
    result = new G4UnionSolid(name, trap, tubicCap, rot, displ);
  }			 	   
  return result;
}

G4VSolid * DDG4SolidConverter::trunctubs(const DDSolid & solid) {
  // truncated tube-section: a boolean subtraction solid:
  //                         from a tube-section a box is subtracted according to the  
  //                         given parameters
  LogDebug("SimG4CoreGeometry") << "MantisConverter: solidshape=" << DDSolidShapesName::name(solid.shape()) << " " << solid;
  LogDebug("SimG4CoreGeometry") << "before";
  DDTruncTubs tt(solid);
  LogDebug("SimG4CoreGeometry") << "after";
  double rIn(tt.rIn()), rOut(tt.rOut()), zHalf(tt.zHalf()),
    startPhi(tt.startPhi()), deltaPhi(tt.deltaPhi()), 
    cutAtStart(tt.cutAtStart()), cutAtDelta(tt.cutAtDelta());
  bool cutInside(bool(tt.cutInside()));
  std::string name=tt.name().name();

  // check the parameters
  if (rIn <= 0 || rOut <=0 || cutAtStart <=0 || cutAtDelta <= 0) {
    throw cms::Exception("DetectorDescriptionFault", "TruncTubs " + std::string(tt.name().fullname()) + ": 0 <= rIn,cutAtStart,rOut,cutAtDelta,rOut violated!");
  }
  if (rIn >= rOut) {
    throw cms::Exception("DetectorDescriptionFault", "TruncTubs " + std::string(tt.name().fullname()) + ": rIn<rOut violated!");
  }
  if (startPhi != 0.) {
    throw cms::Exception("DetectorDescriptionFault", "TruncTubs " + std::string(tt.name().fullname()) + ": startPhi != 0 not supported!");
  }
  
  startPhi=0.;
  double r(cutAtStart), R(cutAtDelta);
  G4VSolid * result(nullptr);
  G4VSolid * tubs = new G4Tubs(name,rIn,rOut,zHalf,startPhi,deltaPhi);
  LogDebug("SimG4CoreGeometry") << "G4Tubs: " << rIn/CLHEP::cm << ' ' << rOut/CLHEP::cm << ' ' << zHalf/CLHEP::cm << ' ' << startPhi/CLHEP::deg << ' ' << deltaPhi/CLHEP::deg;
  LogDebug("SimG4CoreGeometry") << solid;
  // length & hight of the box 
  double boxX(30.*rOut), boxY(20.*rOut); // exaggerate dimensions - does not matter, it's subtracted!
   
  // width of the box > width of the tubs
  double boxZ(1.1*zHalf);
   
  // angle of the box w.r.t. tubs
  double cath = r-R*cos(deltaPhi);
  double hypo = sqrt(r*r+R*R-2.*r*R*cos(deltaPhi));
  double cos_alpha = cath/hypo;

  double alpha = -acos(cos_alpha);
  LogDebug("SimG4CoreGeometry") << "cath=" << cath/CLHEP::cm;
  LogDebug("SimG4CoreGeometry") << "hypo=" << hypo/CLHEP::cm;
  LogDebug("SimG4CoreGeometry") << "al=" << acos(cath/hypo)/CLHEP::deg;
  LogDebug("SimG4CoreGeometry") << "deltaPhi=" << deltaPhi/CLHEP::deg << "\n"
				<< "r=" << r/CLHEP::cm << "\n"
				<<  "R=" << R/CLHEP::cm;

  LogDebug("SimG4CoreGeometry") << "alpha=" << alpha/CLHEP::deg;
    
  // rotationmatrix of box w.r.t. tubs
  G4RotationMatrix * rot = new G4RotationMatrix;
  rot->rotateZ(-alpha);
  LogDebug("SimG4CoreGeometry") << (*rot);

  // center point of the box
  double xBox;
  if (!cutInside) {
    xBox = r+boxY/sin(std::abs(alpha));
  } else {
    xBox = -(boxY/sin(std::abs(alpha))-r);
  }

  G4ThreeVector trans(xBox,0.,0.);
  LogDebug("SimG4CoreGeometry") << "trans=" << trans;

  G4VSolid * box = new G4Box(name,boxX,boxY,boxZ);
  result = new G4SubtractionSolid(name,tubs,box,rot,trans);
      
  return result;

}

#include "G4Sphere.hh"
G4VSolid * DDG4SolidConverter::sphere(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: sphere = " << solid;
  DDSphere sp(solid);
  return new G4Sphere(solid.name().name(), sp.innerRadius(),
		      sp.outerRadius(),
		      sp.startPhi(),
		      sp.deltaPhi(),
		      sp.startTheta(),
		      sp.deltaTheta());
}

#include "G4EllipticalTube.hh"
G4VSolid * DDG4SolidConverter::ellipticaltube(const DDSolid & solid) {
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: ellipticaltube = " << solid;
  DDEllipticalTube sp(solid);
  return new G4EllipticalTube(solid.name().name(),
			      sp.xSemiAxis(),
			      sp.ySemiAxis(),
			      sp.zHeight());
}
