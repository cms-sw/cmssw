//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
//---------------------------------------------------------------
//      GEANT 4 class header file
//
//  G4Physics2DVector95.hh 
//
//  Class description:
//
//  A 2-dimentional vector with linear interpolation.

//  Author:        Vladimir Ivanchenko
//
//  Creation date: 25.09.2011
//---------------------------------------------------------------

#ifndef G4Physics2DVector95_h
#define G4Physics2DVector95_h 1

#include "globals.hh"
#include "G4ios.hh"

#include <iostream>
#include <fstream>
#include <vector>

#include "G4Physics2DVectorCache95.hh"
#include "G4PhysicsVectorType.hh"

typedef std::vector<G4double> G4PV2DDataVector;

class G4Physics2DVector95 
{
public:  // with description

  G4Physics2DVector95();
  G4Physics2DVector95(size_t nx, size_t ny);
  // constructors

  G4Physics2DVector95(const G4Physics2DVector95&);
  G4Physics2DVector95& operator=(const G4Physics2DVector95&);
  // Copy constructor and assignment operator.

  ~G4Physics2DVector95();
  // destructor

  inline G4double Value(G4double x, G4double y);
  // Main method to interpolate 2D vector

  inline void PutX(size_t idx, G4double value);
  inline void PutY(size_t idy, G4double value);
  inline void PutValue(size_t idx, size_t idy, G4double value);
  void PutVectors(const std::vector<G4double>& vecX,
		  const std::vector<G4double>& vecY);
  // Methods to fill vector 
  // Take note that the 'index' starts from '0'.

  void ScaleVector(G4double factor);
  // Scale all values of the vector by factor, 
  // This method may be applied 
  // for example after Retrieve a vector from an external file to 
  // convert values into Geant4 units

  inline G4double GetX(size_t index) const;
  inline G4double GetY(size_t index) const;
  inline G4double GetValue(size_t idx, size_t idy) const;
  // Returns simply the values of the vector by index
  // of the energy vector. The boundary check will not be done. 

  inline size_t GetLengthX() const;
  inline size_t GetLengthY() const;
  // Get the lengths of the vector. 

  inline G4PhysicsVectorType GetType() const;
  // Get physics vector type
  
  void Store(std::ofstream& fOut);
  G4bool Retrieve(std::ifstream& fIn);
  // To store/retrieve persistent data to/from file streams.

  inline G4double GetLastX() const;
  inline G4double GetLastY() const;
  inline G4double GetLastValue() const;
  inline size_t GetLastBinX() const;
  inline size_t GetLastBinY() const;
  // Get cache values 

  inline void SetVerboseLevel(G4int value);
  inline G4int GetVerboseLevel(G4int);
  // Set/Get Verbose level

protected:

  void PrepareVectors();

  void ClearVectors();

  void CopyData(const G4Physics2DVector95& vec);

  void ComputeValue(G4double x, G4double y);
  // Main method to interpolate 2D vector

  size_t FindBinLocation(G4double z, const G4PV2DDataVector&);
  // Main method to local bin

  inline void FindBin(G4double z, const G4PV2DDataVector&, 
			size_t& lastidx);
  inline void FindBinLocationX(G4double x);
  inline void FindBinLocationY(G4double y);
  // Find the bin# in which theEnergy belongs
  // Starting from 0 

private:

  G4int operator==(const G4Physics2DVector95 &right) const ;
  G4int operator!=(const G4Physics2DVector95 &right) const ;

  G4PhysicsVectorType type;   // The type of PhysicsVector (enumerator)    

  size_t numberOfXNodes;
  size_t numberOfYNodes;

  G4Physics2DVectorCache95*  cache;

  G4PV2DDataVector  xVector;
  G4PV2DDataVector  yVector;
  std::vector<G4PV2DDataVector*> value;

  G4int verboseLevel;
};

#include "G4Physics2DVector95.icc"

#endif
