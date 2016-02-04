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
  ~MaterialBudgetData();

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
  float getSupportFractionIL() const {
    return theSupportFractionIL; }
  float getSensitiveFractionIL() const {
    return theSensitiveFractionIL; }
  float getCablesFractionIL() const {
    return theCablesFractionIL; }
  float getCoolingFractionIL() const {
    return theCoolingFractionIL; }
  float getElectronicsFractionIL() const {
    return theElectronicsFractionIL; }
  float getOtherFractionIL() const {
    return theOtherFractionIL; }
  float getAirFractionIL() const {
    return theAirFractionIL; }
  float getTotalIL() const {
    return theTotalIL; }
  float getSupportIL() const {
    return theSupportIL; }
  float getSensitiveIL() const {
    return theSensitiveIL; }
  float getCablesIL() const {
    return theCablesIL; }
  float getCoolingIL() const {
    return theCoolingIL; }
  float getElectronicsIL() const {
    return theElectronicsIL; }
  float getOtherIL() const {
    return theOtherIL; }
  float getAirIL() const {
    return theAirIL; }
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
  float getMass() const {
    return theMass; }
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
  float getIntLen() const {
    return theIntLen; }
  
  float getStepDmb( int is ) {
    return theDmb[is];
  }
  float getSupportDmb( int is ) const {
    return theSupportDmb[is]; }
  float getSensitiveDmb( int is ) const {
    return theSensitiveDmb[is]; }
  float getCablesDmb( int is ) const {
    return theCablesDmb[is]; }
  float getCoolingDmb( int is ) const {
    return theCoolingDmb[is]; }
  float getElectronicsDmb( int is ) const {
    return theElectronicsDmb[is]; }
  float getOtherDmb( int is ) const {
    return theOtherDmb[is]; }
  float getAirDmb( int is ) const {
    return theAirDmb[is]; }

  float getStepDil( int is ) {
    return theDil[is];
  }
  float getSupportDil( int is ) const {
    return theSupportDil[is]; }
  float getSensitiveDil( int is ) const {
    return theSensitiveDil[is]; }
  float getCablesDil( int is ) const {
    return theCablesDil[is]; }
  float getCoolingDil( int is ) const {
    return theCoolingDil[is]; }
  float getElectronicsDil( int is ) const {
    return theElectronicsDil[is]; }
  float getOtherDil( int is ) const {
    return theOtherDil[is]; }
  float getAirDil( int is ) const {
    return theAirDil[is]; }
  
  double getStepInitialX( int is ) {
    return theInitialX[is];
  }
  double getStepInitialY( int is ) {
    return theInitialY[is];
  }
  double getStepInitialZ( int is ) {
    return theInitialZ[is];
  }
  double getStepFinalX( int is ) {
    return theFinalX[is];
  }
  double getStepFinalY( int is ) {
    return theFinalY[is];
  }
  double getStepFinalZ( int is ) {
    return theFinalZ[is];
  }
  int getStepID( int is) {
    return theStepID[is];
  }
  float getStepInitialPt( int is) {
    return theStepInitialPt[is];
  }
  float getStepInitialEta( int is) {
    return theStepInitialEta[is];
  }
  float getStepInitialPhi( int is) {
    return theStepInitialPhi[is];
  }
  float getStepInitialEnergy( int is) {
    return theStepInitialEnergy[is];
  }
  float getStepInitialPx( int is) {
    return theStepInitialPx[is];
  }
  float getStepInitialPy( int is) {
    return theStepInitialPy[is];
  }
  float getStepInitialPz( int is) {
    return theStepInitialPz[is];
  }
  float getStepInitialBeta( int is) {
    return theStepInitialBeta[is];
  }
  float getStepInitialGamma( int is) {
    return theStepInitialGamma[is];
  }
  float getStepInitialMass( int is) {
    return theStepInitialMass[is];
  }
  float getStepFinalPt( int is) {
    return theStepFinalPt[is];
  }
  float getStepFinalEta( int is) {
    return theStepFinalEta[is];
  }
  float getStepFinalPhi( int is) {
    return theStepFinalPhi[is];
  }
  float getStepFinalEnergy( int is) {
    return theStepFinalEnergy[is];
  }
  float getStepFinalPx( int is) {
    return theStepFinalPx[is];
  }
  float getStepFinalPy( int is) {
    return theStepFinalPy[is];
  }
  float getStepFinalPz( int is) {
    return theStepFinalPz[is];
  }
  float getStepFinalBeta( int is) {
    return theStepFinalBeta[is];
  }
  float getStepFinalGamma( int is) {
    return theStepFinalGamma[is];
  }
  float getStepFinalMass( int is) {
    return theStepFinalMass[is];
  }
  int getStepPreProcess( int is) {
    return theStepPreProcess[is];
  }
  int getStepPostProcess( int is) {
    return theStepPostProcess[is];
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
  CLHEP::HepLorentzVector getStepVolumeXaxis( int is ) {
    return CLHEP::HepLorentzVector(theVolumeXaxis1[is],theVolumeXaxis2[is],theVolumeXaxis3[is]);
  }
  CLHEP::HepLorentzVector getStepVolumeYaxis( int is ) {
    return CLHEP::HepLorentzVector(theVolumeYaxis1[is],theVolumeYaxis2[is],theVolumeYaxis3[is]);
  }
  CLHEP::HepLorentzVector getStepVolumeZaxis( int is ) {
    return CLHEP::HepLorentzVector(theVolumeZaxis1[is],theVolumeZaxis2[is],theVolumeZaxis3[is]);
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
  float getStepMaterialLambda0( int is ) {
    return theMaterialLambda0[is];
  }
  float getStepMaterialDensity( int is ) {
    return theMaterialDensity[is];
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
  float theMass;
  float theSupportFractionMB, theSensitiveFractionMB, theCablesFractionMB,
    theCoolingFractionMB, theElectronicsFractionMB, theOtherFractionMB, theAirFractionMB;
  float theSupportFractionIL, theSensitiveFractionIL, theCablesFractionIL,
    theCoolingFractionIL, theElectronicsFractionIL, theOtherFractionIL, theAirFractionIL;
  float theSupportMB, theSensitiveMB, theCablesMB, theCoolingMB, theElectronicsMB, theOtherMB, theAirMB;
  float theTotalIL;
  float theSupportIL, theSensitiveIL, theCablesIL, theCoolingIL, theElectronicsIL, theOtherIL, theAirIL;
  // rr
  int theStepN;
  double *theInitialX, *theInitialY, *theInitialZ;
  double *theFinalX,   *theFinalY,   *theFinalZ;
  float *theDmb;
  float *theSupportDmb, *theSensitiveDmb, *theCablesDmb, *theCoolingDmb, *theElectronicsDmb, *theOtherDmb, *theAirDmb;
  float *theDil;
  float *theSupportDil, *theSensitiveDil, *theCablesDil, *theCoolingDil, *theElectronicsDil, *theOtherDil, *theAirDil;
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
  float*       theMaterialLambda0;
  float*       theMaterialDensity;
  int*   theStepID;
  float* theStepInitialPt;
  float* theStepInitialEta;
  float* theStepInitialPhi;
  float* theStepInitialEnergy;
  float* theStepInitialPx;
  float* theStepInitialPy;
  float* theStepInitialPz;
  float* theStepInitialBeta;
  float* theStepInitialGamma;
  float* theStepInitialMass;
  float* theStepFinalPt;
  float* theStepFinalEta;
  float* theStepFinalPhi;
  float* theStepFinalEnergy;
  float* theStepFinalPx;
  float* theStepFinalPy;
  float* theStepFinalPz;
  float* theStepFinalBeta;
  float* theStepFinalGamma;
  float* theStepFinalMass;
  int*   theStepPreProcess;
  int*   theStepPostProcess;
  // rr
  float theTrkLen;
  std::string thePVname;
  int thePVcopyNo;

  MaterialBudgetCategorizer* myMaterialBudgetCategorizer;

  float theRadLen;
  float theIntLen;
  int stepN;
  int MAXNUMBERSTEPS;
  bool allStepsToTree;
  
  double densityConvertionFactor;
};

#endif
