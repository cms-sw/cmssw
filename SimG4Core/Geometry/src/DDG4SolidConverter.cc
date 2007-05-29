#include<sstream>
#include "SimG4Core/Geometry/interface/DDG4SolidConverter.h"
#include "G4VSolid.hh"

#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
const vector<double> * DDG4SolidConverter::par_ = 0; 

DDG4SolidConverter::DDG4SolidConverter()
{
   // could also be done 'dynamically' from outside 
   // would then need to have a 'register' method ...
  par_=0;
   convDispatch_[ddbox]    = DDG4SolidConverter::box; 
   convDispatch_[ddtubs]   = DDG4SolidConverter::tubs;
   convDispatch_[ddtrap]   = DDG4SolidConverter::trap;
   convDispatch_[ddcons]   = DDG4SolidConverter::cons;
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
   
}


DDG4SolidConverter::~DDG4SolidConverter() { }

G4VSolid * DDG4SolidConverter::convert(const DDSolid & s)
{
   G4VSolid * result = 0;
   par_ = &(s.parameters());
   map<DDSolidShape,FNPTR>::iterator it = convDispatch_.find(s.shape());
   if (it != convDispatch_.end()) {
     result = it->second(s);
   } 
   else {
     ostringstream o; 
     o << "DDG4SolidConverter::convert: conversion failed for s=" << s << endl;
     o << " solid.shape()=" << s.shape() << endl;
     throw DDException(o.str());
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
  vector<double>* r_p = new vector<double>;
  vector<double>* z_p = new vector<double>;
  vector<double>& r = *r_p;
  vector<double>& z = *z_p;
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
   return new G4Polycone(s.name().name(), (*par_)[0], (*par_)[1], // start,delta-phi
                                 count, // numRZ
	 			 &(*r.begin()),
				 &(*z.begin()));
}


G4VSolid * DDG4SolidConverter::polycone_rrz(const DDSolid & s)
{
    LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: pcon_rrz = " << s ;  
    vector<double>* z_p = new vector<double>;
    vector<double>* rmin_p = new vector<double>;
    vector<double>* rmax_p = new vector<double>;
    vector<double>::const_iterator i = par_->begin()+2;
    int count = 0;
    for (; i!=par_->end(); ++i) {
        LogDebug("SimG4CoreGeometry") << "z=" << *i ;
      (*z_p).push_back(*i); ++i;
        LogDebug("SimG4CoreGeometry") << "rmin=" << *i ;
      (*rmin_p).push_back(*i); ++i;
        LogDebug("SimG4CoreGeometry") << "rmax=" << *i ;
      (*rmax_p).push_back(*i); 
      count++;
    }
    LogDebug("SimG4CoreGeometry") << "sp=" << (*par_)[0]/deg << " ep=" << (*par_)[1]/deg ;
    return new G4Polycone(s.name().name(), (*par_)[0], (*par_)[1], // start,delta-phi
                        count, // sections
			&((*z_p)[0]),
			&((*rmin_p)[0]),
			&((*rmax_p)[0]));
			
}


#include "G4Polyhedra.hh"
G4VSolid * DDG4SolidConverter::polyhedra_rz(const DDSolid & s)
{
    LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: phed_rz = " << s ;  
    vector<double>* r_p = new vector<double>; // geant gets the memory!
    vector<double>* z_p = new vector<double>;
    vector<double>& r = *r_p;
    vector<double>& z = *z_p;
    vector<double>::const_iterator i = par_->begin()+3;
    int count=0;
    
    for(; i!=par_->end(); ++i) {
      z.push_back(*i); ++i;
      r.push_back(*i);
      count++;
    }
      
    return new G4Polyhedra(s.name().name(), (*par_)[1], (*par_)[2], int((*par_)[0]),// start,delta-phi;sides
                                count, // numRZ
				&(*r.begin()),
				&(*z.begin()));
}


G4VSolid * DDG4SolidConverter::polyhedra_rrz(const DDSolid & s)
{
    LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: phed_rrz = " << s ;  
    vector<double>* z_p = new vector<double>;
    vector<double>* rmin_p = new vector<double>;
    vector<double>* rmax_p = new vector<double>;
    vector<double>::const_iterator i = par_->begin()+3;
    int count = 0;
    for (; i!=par_->end(); ++i) {
        LogDebug("SimG4CoreGeometry") << "z=" << *i ;
      (*z_p).push_back(*i); ++i;
        LogDebug("SimG4CoreGeometry") << "rmin=" << *i ;
      (*rmin_p).push_back(*i); ++i;
        LogDebug("SimG4CoreGeometry") << "rmax=" << *i ;
      (*rmax_p).push_back(*i); 
      count++;
    }
    LogDebug("SimG4CoreGeometry") << "sp=" << (*par_)[0]/deg << " ep=" << (*par_)[1]/deg ;
    return new G4Polyhedra(s.name().name(), (*par_)[1], (*par_)[2], int((*par_)[0]), // start,delta-phi,sides
                         count, // sections
			 &((*z_p)[0]),
			 &((*rmin_p)[0]),
			 &((*rmax_p)[0]));  
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
G4VSolid * DDG4SolidConverter::reflected(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: reflected = " << s ;  
  G4ReflectedSolid * rs = 0;
  try {
    DDReflectionSolid rfs(s); 
    if (rfs) {	
    static /* G4Transform3D */ HepReflectZ3D z_reflection; // = HepReflectZ3D;	
    rs = new G4ReflectedSolid(s.name().name(), 
                              DDG4SolidConverter().convert(rfs.unreflected()), 
			      z_reflection);
    
    }
  }
  catch(...) {
    cerr << " conversion to unreflected solid failed! " << endl
         << " Reflectionsolid name=" << s.name() << endl;
  } 
  return rs;
}


#include "G4UnionSolid.hh"
G4VSolid * DDG4SolidConverter::unionsolid(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: unionsolid = " << s.name() ;
  G4UnionSolid * us = 0;
  try {
    DDBooleanSolid bs(s);
    if (bs) {
      LogDebug("SimG4CoreGeometry") << "SolidA=" << bs.solidA();
      G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
      LogDebug("SimG4CoreGeometry") << "SolidB=" << bs.solidB();
      G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
      LogDebug("SimG4CoreGeometry") << " name:" << s.name() << " t=" << bs.translation() << flush;
      stringstream sst;
      bs.rotation().rotation()->inverse().print(sst);
      LogDebug("SimG4CoreGeometry") << " " << sst.str() << flush;
      us = new G4UnionSolid(s.name().name(),
                            sa,
			    sb,
			    new DDRotationMatrix(bs.rotation().rotation()->inverse()),
			    bs.translation());
			   
			    
    }
  }
  catch(...) {
    cerr << " conversion to a unionsolid failed! " << endl
         << " UnionSolid name=" << s.name() << endl;
  }
  return us;	   
}


#include "G4SubtractionSolid.hh"
#include <sstream>
G4VSolid * DDG4SolidConverter::subtraction(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: subtraction = " << s ;
  G4SubtractionSolid * us = 0;
  try {
    DDBooleanSolid bs(s);
    if (bs) {
      G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
      G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
      LogDebug("SimG4CoreGeometry") << " name:" << s.name() << " t=" << bs.translation() << flush;
      stringstream sst;
      bs.rotation().rotation()->inverse().print(sst);
      LogDebug("SimG4CoreGeometry") << " " << sst.str() << flush;
      us = new G4SubtractionSolid(s.name().name(),
                            sa,
			    sb,
			    new DDRotationMatrix(bs.rotation().rotation()->inverse()),
			    bs.translation());
			   
			    
    }
  }
  catch(...) {
    cerr << " conversion to a subtractionsolid failed! " << endl
         << " SubtractionSolid name=" << s.name() << endl;
  }
  return us;	   
}


#include "G4IntersectionSolid.hh"
G4VSolid * DDG4SolidConverter::intersection(const DDSolid & s)
{
  LogDebug("SimG4CoreGeometry") << "DDG4SolidConverter: intersection = " << s ;
  G4IntersectionSolid * us = 0;
  try {
    DDBooleanSolid bs(s);
    if (bs) {
      G4VSolid * sa = DDG4SolidConverter().convert(bs.solidA());
      G4VSolid * sb = DDG4SolidConverter().convert(bs.solidB());
      LogDebug("SimG4CoreGeometry") << " name:" << s.name() << " t=" << bs.translation() << flush;
      stringstream sst;
      bs.rotation().rotation()->inverse().print(sst);
      LogDebug("SimG4CoreGeometry") << " " << sst.str() << flush;
      us = new G4IntersectionSolid(s.name().name(),
                            sa,
			    sb,
			    new DDRotationMatrix(bs.rotation().rotation()->inverse()),
			    bs.translation());
			   
			    
    }
  }
  catch(...) {
    cerr << " conversion to a intersection failed! " << endl
         << " IntersectionSolid name=" << s.name() << endl;
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
  try {
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
     throw DDException("Check parameters of the PseudoTrap! name=" + pt.name().name());   
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
  }
  catch (DDException e) {
    cerr << "DDException: " << e << endl;
    throw;
  }
  catch (...) {
    cerr << " conversion to a PseudoTrap failed!" << endl
         << " PseudoTrap = " << s << endl;
    throw;
  }	 
  return result;
}


G4VSolid * DDG4SolidConverter::trunctubs(const DDSolid & s)
{
  // truncated tube-section: a boolean subtraction solid:
  //                         from a tube-section a box is subtracted according to the  
  //                         given parameters
  LogDebug("SimG4CoreGeometry") << "MantisConverter: solidshape=" << DDSolidShapesName::name(s.shape()) << " " << s;
  try {
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
      string s = "TruncTubs " + string(tt.name()) + ": 0 <= rIn,cutAtStart,rOut,cutAtDelta,rOut violated!";
      throw DDException(s);
    }
    if (rIn >= rOut) {
      string s = "PseudoTrap " + string(tt.name()) + ": rIn<rOut violated!";
      throw DDException(s);
    }
    if (startPhi != 0.) {
      string s= "TruncTubs " + string(tt.name()) + ": startPhi != 0 not supported!";
      throw DDException(s);
    }
    if (cutInside != false) {
      string s = "TruncTubs " + string(tt.name()) + " cutInside == true not supported!";
      throw DDException(s);
    }

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
    double xBox(r+boxY/sin(abs(alpha)));
    G4ThreeVector trans(xBox,0.,0.);
    LogDebug("SimG4CoreGeometry") << "trans=" << trans;

    G4VSolid * box = new G4Box(name,boxX,boxY,boxZ);
    result = new G4SubtractionSolid(name,tubs,box,rot,trans);
      
    return result;
  }
  catch(const DDException & e) {
    cerr << "DDException: " << e << endl << "Tried to convert: " << s << " to a TrucTubs" << endl;
    throw;
  }
  catch(...) {
    cerr << " conversion to a TruncTubs failed!" << endl
	 << " TrucTubs = " << s << endl;
    throw;
  }

}
