#ifndef SimG4Core_FieldBuilder_H
#define SimG4Core_FieldBuilder_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Field;
class DDLogicalPart;
class MagneticField;

class G4FieldManager;
class G4Mag_UsualEqRhs;
class G4PropagatorInField;
class G4LogicalVolume;

class FieldBuilder
{
public:
    static FieldBuilder * instance();
    ~FieldBuilder();
    void readFieldParameters(DDLogicalPart theLogicalPart,
			     std::string keywordField);
    void setField(const MagneticField * f, const edm::ParameterSet & p);
    void configure(std::string keywordField,G4FieldManager * fM = 0,
		   G4PropagatorInField * fP = 0);
    G4LogicalVolume * fieldTopVolume();
private:
    FieldBuilder();
    void configureFieldManager(G4FieldManager * fM);
    void configurePropagatorInField(G4PropagatorInField * fP);  
private:
    static FieldBuilder * theBuilder;
    Field * theField;
    G4Mag_UsualEqRhs * theFieldEquation;
    G4LogicalVolume * theTopVolume;
    std::string keywordField;
    std::string fieldType;
    double fieldValue;
    std::string stepper;
    double minStep;
    double dChord;
    double dOneStep;
    double dIntersection;
    double dIntersectionAndOneStep;
    double maxLoopCount;
    double minEpsilonStep;
    double maxEpsilonStep;
};

#endif
