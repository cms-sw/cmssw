#ifndef _PPS_PixelG4Hit_h
#define _PPS_PixelG4Hit_h 1
// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelG4Hit
//
/**\class PPSPixelG4Hit PPSPixelG4Hit.h SimG4CMS/PPS/interface/PPSPixelG4Hit.h
 
 Description: Transient Hit class for PPS taken from those for Calorimeters
 
 Usage: One Hit object should be created
   -for each new particle entering the calorimeter
   -for each detector unit (= cristal or fiber or scintillator layer)
   -for each nanosecond of the shower development

   This implies that all hit objects created for a given shower
   have the same value for
   - Entry (= local coordinates of the entrance point of the particle
              in the unit where the shower starts) 
   - the TrackID (= Identification number of the incident particle)
   - the IncidentEnergy (= energy of that particle)
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files

#include "G4VHit.hh"
#include "G4ThreeVector.hh"
#include <iostream>

class PPSPixelG4Hit : public G4VHit {
public:
  // ---------- Constructor and destructor -----------------
  PPSPixelG4Hit();
  ~PPSPixelG4Hit() override = default;
  PPSPixelG4Hit(const PPSPixelG4Hit& right);

  // ---------- operators ----------------------------------
  const PPSPixelG4Hit& operator=(const PPSPixelG4Hit& right);
  int operator==(const PPSPixelG4Hit&) { return 0; }

  // ---------- member functions ---------------------------
  void Draw() override {}
  void Print() override;

  const G4ThreeVector& getMeanPosition() const { return MeanPosition_; };
  void setMeanPosition(const G4ThreeVector& a) { MeanPosition_ = a; };

  const G4ThreeVector& getEntryPoint() const;
  void setEntryPoint(const G4ThreeVector&);
  const G4ThreeVector& getExitPoint() const;
  void setExitPoint(const G4ThreeVector&);

  double getEM() const;
  void setEM(double e);

  double getHadr() const;
  void setHadr(double e);

  double getIncidentEnergy() const;
  void setIncidentEnergy(double e);

  int getTrackID() const;
  void setTrackID(int i);

  uint32_t getUnitID() const;
  void setUnitID(uint32_t i);

  double getTimeSlice() const;
  void setTimeSlice(double d);
  int getTimeSliceID() const;

  void addEnergyDeposit(double em, double hd);
  void addEnergyDeposit(const PPSPixelG4Hit& aHit);

  double getEnergyDeposit() const;

  float getPabs() const;
  float getTof() const;
  float getEnergyLoss() const;
  int getParticleType() const;

  void setPabs(float e);
  void setTof(float e);
  void setEnergyLoss(float e);
  void setParticleType(short i);

  float getThetaAtEntry() const;
  float getPhiAtEntry() const;

  void setThetaAtEntry(float t);
  void setPhiAtEntry(float f);
  float getPx() const;
  float getPy() const;
  float getPz() const;
  float getVPx() const;
  float getVPy() const;
  float getVPz() const;

  void setPx(float e);
  void setPy(float e);
  void setPz(float e);
  void setVPx(float e);
  void setVPy(float e);
  void setVPz(float e);
  float getX() const;
  float getY() const;
  float getZ() const;
  void setX(float t);
  void setY(float t);
  void setZ(float t);

  int getParentId() const;
  float getVx() const;
  float getVy() const;
  float getVz() const;

  void setParentId(int p);
  void setVx(float p);
  void setVy(float p);
  void setVz(float p);

private:
  G4ThreeVector MeanPosition_;
  double elem_;               //EnergyDeposit of EM particles
  double hadr_;               //EnergyDeposit of HD particles
  double theIncidentEnergy_;  //Energy of the primary particle
  int theTrackID_;            //Identification number of the primary
                              //particle
  uint32_t theUnitID_;        //PPS Unit Number
  double theTimeSlice_;       //Time Slice Identification

  float theX_;
  float theY_;
  float theZ_;
  float thePabs_;
  float theTof_;
  float theEnergyLoss_;
  int theParticleType_;

  float theThetaAtEntry_;
  float thePhiAtEntry_;
  G4ThreeVector theEntryPoint_;
  G4ThreeVector theExitPoint_;
  float thePx_, thePy_, thePz_, theVPx_, theVPy_, theVPz_;
  int theParentId_;
  float theVx_;
  float theVy_;
  float theVz_;
};

std::ostream& operator<<(std::ostream&, const PPSPixelG4Hit&);

#endif
