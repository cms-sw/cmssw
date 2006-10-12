#ifndef MaterialBudgetData_h
#define MaterialBudgetData_h 1

#include "Validation/Geometry/interface/MaterialBudgetCategorizer.h"
#include "G4ThreeVector.hh"
class MaterialBudgetData;
class G4Step;
class G4Track;

typedef std::map< std::string, float > msf;

class MaterialBudgetData {
public:

  MaterialBudgetData();
  ~MaterialBudgetData(){ }

  void dataStartTrack( const G4Track* aTrack );
  void dataEndTrack( const G4Track* aTrack );
  void dataPerStep( const G4Step* aStep );

  void SetAllStepsToTree();
 public:
  float getTotalMB() const {
    return theTotalMB; }
  // rr
  float getSupportFractionMB() const {
    return theSupportFractionMB; }
  float getSensitiveFractionMB() const {
    return theSensitiveFractionMB; }
  float getCablesFractionMB() const {
    return theCablesFractionMB; }
  float getCoolingFractionMB() const {
    return theCoolingFractionMB; }
  float getElectronicsFractionMB() const {
    return theElectronicsFractionMB; }
  float getOtherFractionMB() const {
    return theOtherFractionMB; }
  float getAirFractionMB() const {
    return theAirFractionMB; }
  float getSupportMB() const {
    return theSupportMB; }
  float getSensitiveMB() const {
    return theSensitiveMB; }
  float getCablesMB() const {
    return theCablesMB; }
  float getCoolingMB() const {
    return theCoolingMB; }
  float getElectronicsMB() const {
    return theElectronicsMB; }
  float getOtherMB() const {
    return theOtherMB; }
  float getAirMB() const {
    return theAirMB; }
  // rr
  float getEta() const {
    return theEta; }
  float getPhi() const {
    return thePhi; }
  int getNumberOfSteps() const {
    return theStepN; }

  float getTrkLen() const {
    return theTrkLen; }
  std::string getPVname() const {
    return thePVname; }
  int getPVcopyNo() const {
    return thePVcopyNo; }
  float getRadLen() const {
    return theRadLen; }

  float getStepDmb( int is ) {
    return theDmb[is];
  }
  float getStepX( int is ) {
    return theX[is];
  }
  float getStepY( int is ) {
    return theY[is];
  }
  float getStepZ( int is ) {
    return theZ[is];
  }
  float getStepVoluId( int is ) {
    return theVoluId[is];
  }
  float getStepMateId( int is ) {
    return theMateId[is];
  }

  bool allStepsON() {
    return allStepsToTree;
  }

 private:
  float theTotalMB, theEta, thePhi;
  // rr
  float theSupportFractionMB, theSensitiveFractionMB, theCablesFractionMB,
    theCoolingFractionMB, theElectronicsFractionMB, theOtherFractionMB, theAirFractionMB;
  float theSupportMB, theSensitiveMB, theCablesMB, theCoolingMB, theElectronicsMB, theOtherMB, theAirMB;
  // rr
  int theStepN;
  float *theX, *theY, *theZ;
  float *theDmb;
  int *theVoluId;
  int *theMateId;
  float theTrkLen;
  std::string thePVname;
  int thePVcopyNo;

  MaterialBudgetCategorizer* myMaterialBudgetCategorizer;

  float theRadLen;
  int stepN;
  int MAXNUMBERSTEPS;
  bool allStepsToTree;
};

#endif
