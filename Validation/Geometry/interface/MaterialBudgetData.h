#ifndef MaterialBudgetData_h
#define MaterialBudgetData_h 1

#include "Validation/Geometry/interface/MaterialBudgetCategorizer.h"
#include "G4ThreeVector.hh"

// rr
#include <CLHEP/Vector/LorentzVector.h>
// rr

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
  // rr
  int getID() const {
    return theID; }
  float getPt() const {
    return thePt; }
  float getEnergy() const {
    return theEnergy; }
  // rr
  
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
  /*
    int getStepVoluId( int is ) {
    return theVoluId[is];
    }
    int getStepMateId( int is ) {
    return theMateId[is];
    }
  */
  int getStepID( int is) {
    return theStepID[is];
  }
  float getStepPt( int is) {
    return theStepPt[is];
  }
  float getStepEta( int is) {
    return theStepEta[is];
  }
  float getStepPhi( int is) {
    return theStepPhi[is];
  }
  float getStepEnergy( int is) {
    return theStepEnergy[is];
  }
  // rr
  int getStepVolumeID( int is ) {
    return theVolumeID[is];
  }
  std::string getStepVolumeName( int is ) {
    return theVolumeName[is];
  }
  int getStepVolumeCopy( int is ) {
    return theVolumeCopy[is];
  }
  float getStepVolumeX( int is ) {
    return theVolumeX[is];
  }
  float getStepVolumeY( int is ) {
    return theVolumeY[is];
  }
  float getStepVolumeZ( int is ) {
    return theVolumeZ[is];
  }
  HepLorentzVector getStepVolumeXaxis( int is ) {
    return HepLorentzVector(theVolumeXaxis1[is],theVolumeXaxis2[is],theVolumeXaxis3[is]);
  }
  HepLorentzVector getStepVolumeYaxis( int is ) {
    return HepLorentzVector(theVolumeYaxis1[is],theVolumeYaxis2[is],theVolumeYaxis3[is]);
  }
  HepLorentzVector getStepVolumeZaxis( int is ) {
    return HepLorentzVector(theVolumeZaxis1[is],theVolumeZaxis2[is],theVolumeZaxis3[is]);
  }
  int getStepMaterialID( int is ) {
    return theMaterialID[is];
  }
  std::string getStepMaterialName( int is ) {
    return theMaterialName[is];
  }
  float getStepMaterialX0( int is ) {
    return theMaterialX0[is];
  }
  // rr
  
  bool allStepsON() {
    return allStepsToTree;
  }

 private:
  float theTotalMB, theEta, thePhi;
  // rr
  float thePt;
  int   theID;
  float theEnergy;
  float theSupportFractionMB, theSensitiveFractionMB, theCablesFractionMB,
    theCoolingFractionMB, theElectronicsFractionMB, theOtherFractionMB, theAirFractionMB;
  float theSupportMB, theSensitiveMB, theCablesMB, theCoolingMB, theElectronicsMB, theOtherMB, theAirMB;
  // rr
  int theStepN;
  float *theX, *theY, *theZ;
  float *theDmb;
  //  int *theVoluId;
  //  int *theMateId;
  // rr
  int *theVolumeID;
  std::string* theVolumeName;
  int*   theVolumeCopy;
  float* theVolumeX;
  float* theVolumeY;
  float* theVolumeZ;
  float* theVolumeXaxis1;
  float* theVolumeXaxis2;
  float* theVolumeXaxis3;
  float* theVolumeYaxis1;
  float* theVolumeYaxis2;
  float* theVolumeYaxis3;
  float* theVolumeZaxis1;
  float* theVolumeZaxis2;
  float* theVolumeZaxis3;
  int*         theMaterialID;
  std::string* theMaterialName;
  float*       theMaterialX0;
  int*   theStepID;
  float* theStepPt;
  float* theStepEta;
  float* theStepPhi;
  float* theStepEnergy;
  // rr
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
