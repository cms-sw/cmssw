#ifndef SimG4Core_DDG4SolidConverter_h
#define SimG4Core_DDG4SolidConverter_h

#include "DetectorDescription/Core/interface/DDSolidShapes.h"

#include <map>
#include <vector>

class G4VSolid;
class DDSolid;

class DDG4SolidConverter
{
public:
    DDG4SolidConverter();
    ~DDG4SolidConverter();
    typedef G4VSolid * (*FNPTR) (const DDSolid &); // pointer to function
    G4VSolid * convert(const DDSolid &);  

private:  
    /* foreach supported solid add a static conversion routine ,
       register this method in the convDispatch_-map */    
    static G4VSolid * box(const DDSolid &);
    static G4VSolid * tubs(const DDSolid &);
    static G4VSolid * trap(const DDSolid &);
    static G4VSolid * cons(const DDSolid &);
    static G4VSolid * reflected(const DDSolid &);
    static G4VSolid * unionsolid(const DDSolid &);
    static G4VSolid * subtraction(const DDSolid &);
    static G4VSolid * intersection(const DDSolid &);
    static G4VSolid * shapeless(const DDSolid &);
    static G4VSolid * polycone_rz(const DDSolid &);
    static G4VSolid * polycone_rrz(const DDSolid &);
    static G4VSolid * polyhedra_rz(const DDSolid &);
    static G4VSolid * polyhedra_rrz(const DDSolid &);
    static G4VSolid * pseudotrap(const DDSolid & s);
    static G4VSolid * torus(const DDSolid &);
    static G4VSolid * trunctubs(const DDSolid &);
    static G4VSolid * sphere(const DDSolid &);
    static G4VSolid * orb(const DDSolid &);
    static G4VSolid * ellipsoid(const DDSolid &);
    static G4VSolid * ellipticaltube(const DDSolid &);
    static G4VSolid * para(const DDSolid &);
    static const std::vector<double>* par_;
    std::map<DDSolidShape,FNPTR> convDispatch_;

    friend class testTruncTubs;
    friend class testPseudoTrap;
};

#endif
