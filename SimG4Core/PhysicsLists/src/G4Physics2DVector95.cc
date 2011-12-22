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
// 
// --------------------------------------------------------------
//      GEANT 4 class implementation file
//
//  G4Physics2DVector95.cc
//
//  Author:        Vladimir Ivanchenko
//
//  Creation date: 25.09.2011
//
// --------------------------------------------------------------

#include "G4Physics2DVector95.hh"
#include <iomanip>

// --------------------------------------------------------------

G4Physics2DVector95::G4Physics2DVector95()
 : type(T_G4PhysicsFreeVector),
   numberOfXNodes(0), numberOfYNodes(0),
   verboseLevel(0)
{
  cache = new G4Physics2DVectorCache95();
}

// --------------------------------------------------------------

G4Physics2DVector95::G4Physics2DVector95(size_t nx, size_t ny)
 : type(T_G4PhysicsFreeVector),
   numberOfXNodes(nx), numberOfYNodes(ny),
   verboseLevel(0)
{
  cache = new G4Physics2DVectorCache95();
  PrepareVectors();
}

// --------------------------------------------------------------

G4Physics2DVector95::~G4Physics2DVector95() 
{
  delete cache;
  ClearVectors();
}

// --------------------------------------------------------------

G4Physics2DVector95::G4Physics2DVector95(const G4Physics2DVector95& right)
{
  type         = right.type;

  numberOfXNodes = right.numberOfXNodes;
  numberOfYNodes = right.numberOfYNodes;

  verboseLevel = right.verboseLevel;

  xVector      = right.xVector;
  yVector      = right.yVector;

  cache = new G4Physics2DVectorCache95();
  PrepareVectors();
  CopyData(right);
}

// --------------------------------------------------------------

G4Physics2DVector95& G4Physics2DVector95::operator=(const G4Physics2DVector95& right)
{
  if (&right==this)  { return *this; }
  ClearVectors();

  type         = right.type;

  numberOfXNodes = right.numberOfXNodes;
  numberOfYNodes = right.numberOfYNodes;

  verboseLevel = right.verboseLevel;

  cache->Clear();
  PrepareVectors();
  CopyData(right);

  return *this;
}

// --------------------------------------------------------------

void G4Physics2DVector95::PrepareVectors()
{
  xVector.resize(numberOfXNodes,0.);
  yVector.resize(numberOfYNodes,0.);
  value.resize(numberOfYNodes,0);
  for(size_t j=0; j<numberOfYNodes; ++j) {
    G4PV2DDataVector* v = new G4PV2DDataVector();
    v->resize(numberOfXNodes,0.);
    value[j] = v;
  }
}

// --------------------------------------------------------------

void G4Physics2DVector95::ClearVectors()
{
  for(size_t j=0; j<numberOfYNodes; ++j) {
    delete value[j];
  }
}

// --------------------------------------------------------------

void G4Physics2DVector95::CopyData(const G4Physics2DVector95 &right)
{
  for(size_t i=0; i<numberOfXNodes; ++i) {
    xVector[i] = right.xVector[i];
  }
  for(size_t j=0; j<numberOfYNodes; ++j) {
    yVector[j] = right.yVector[j];
    G4PV2DDataVector* v0 = right.value[j];
    for(size_t i=0; i<numberOfXNodes; ++i) { 
      PutValue(i,j,(*v0)[i]); 
    }
  }
}

// --------------------------------------------------------------

void G4Physics2DVector95::ComputeValue(G4double xx, G4double yy)
{
  if(xx != cache->lastBinX) {
    if(xx <= xVector[0]) {
      cache->lastX = xVector[0];
      cache->lastBinX = 0;
    } else if(xx >= xVector[numberOfXNodes-1]) {
      cache->lastX = xVector[numberOfXNodes-1];
      cache->lastBinX = numberOfXNodes-2;
    } else {
      cache->lastX = xx;
      FindBinLocationX(xx);
    }
  }
  if(yy != cache->lastBinY) {
    if(yy <= yVector[0]) {
      cache->lastY = yVector[0];
      cache->lastBinY = 0;
    } else if(yy >= yVector[numberOfYNodes-1]) {
      cache->lastY = yVector[numberOfYNodes-1];
      cache->lastBinY = numberOfYNodes-2;
    } else {
      cache->lastY = yy;
      FindBinLocationY(yy);
    }
  }
  size_t idx  = cache->lastBinX;
  size_t idy  = cache->lastBinY;
  G4double x1 = xVector[idx];
  G4double x2 = xVector[idx+1];
  G4double y1 = yVector[idy];
  G4double y2 = yVector[idy+1];
  G4double x  = cache->lastX;
  G4double y  = cache->lastY;
  G4double v11= GetValue(idx,   idy);
  G4double v12= GetValue(idx+1, idy);
  G4double v21= GetValue(idx,   idy+1);
  G4double v22= GetValue(idx+1, idy+1);
  cache->lastValue = 
    ((y2 - y)*(v11*(x2 - x) + v12*(x - x1)) + 
     ((y - y1)*(v21*(x2 - x) + v22*(x - x1))))/((x2 - x1)*(y2 - y1)); 
}

// --------------------------------------------------------------

void 
G4Physics2DVector95::PutVectors(const std::vector<G4double>& vecX,
			      const std::vector<G4double>& vecY)
{
  ClearVectors();
  numberOfXNodes = vecX.size();
  numberOfYNodes = vecY.size();
  PrepareVectors();
  if(!cache) { cache = new G4Physics2DVectorCache95(); }
  cache->Clear();
  for(size_t i = 0; i<numberOfXNodes; ++i) {
    xVector[i] = vecX[i];
  }
  for(size_t j = 0; j<numberOfYNodes; ++j) {
    yVector[j] = vecY[j];
  }
}

// --------------------------------------------------------------

void G4Physics2DVector95::Store(std::ofstream& out)
{
  // binning
  G4int prec = out.precision();
  out << G4int(type) << " " << numberOfXNodes << " " << numberOfYNodes 
      << G4endl; 
  out << std::setprecision(5);

  // contents
  for(size_t i = 0; i<numberOfXNodes-1; ++i) {
    out << xVector[i] << "  ";
  }
  out << xVector[numberOfXNodes-1] << G4endl;
  for(size_t j = 0; j<numberOfYNodes-1; ++j) {
    out << yVector[j] << "  ";
  }
  out << yVector[numberOfYNodes-1] << G4endl;
  for(size_t j = 0; j<numberOfYNodes; ++j) {
    for(size_t i = 0; i<numberOfXNodes-1; ++i) {
      out << GetValue(i, j) << "  ";
    }
    out << GetValue(numberOfXNodes-1,j) << G4endl;
  }
  out.precision(prec);
  out.close();
}

// --------------------------------------------------------------

G4bool G4Physics2DVector95::Retrieve(std::ifstream& in)
{
  // initialisation
  cache->Clear();
  ClearVectors();

  // binning
  G4int k;
  in >> k >> numberOfXNodes >> numberOfYNodes;
  if (in.fail())  { return false; }
  PrepareVectors();
  type = G4PhysicsVectorType(k); 

  // contents
  G4double val;
  for(size_t i = 0; i<numberOfXNodes; ++i) {
    in >> xVector[i];
    if (in.fail())  { return false; }
  }
  for(size_t j = 0; j<numberOfYNodes; ++j) {
    in >> yVector[j];
    if (in.fail())  { return false; }
  }
  for(size_t j = 0; j<numberOfYNodes; ++j) {
    for(size_t i = 0; i<numberOfXNodes; ++i) {
      in >> val;
      if (in.fail())  { return false; }
      PutValue(i, j, val);
    }
  }
  in.close();
  return true;
}

// --------------------------------------------------------------

void 
G4Physics2DVector95::ScaleVector(G4double factor)
{
  G4double val;
  for(size_t j = 0; j<numberOfYNodes; ++j) {
    for(size_t i = 0; i<numberOfXNodes; ++i) {
      val = GetValue(i, j)*factor;
      PutValue(i, j, val);
    }
  }
}

// --------------------------------------------------------------

size_t 
G4Physics2DVector95::FindBinLocation(G4double z, 
				   const G4PV2DDataVector& v)
{
  size_t lowerBound = 0;
  size_t upperBound = v.size() - 2;

  while (lowerBound <= upperBound)
  {
    size_t midBin = (lowerBound + upperBound)/2;
    if( z < v[midBin] ) { upperBound = midBin-1; }
    else                { lowerBound = midBin+1; }
  }

  return upperBound;
}

// --------------------------------------------------------------
