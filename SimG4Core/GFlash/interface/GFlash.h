#ifndef SimG4Core_GFlash_GFlash_H
#define SimG4Core_GFlash_GFlash_H
// Joanna Weng 08.2005
// modifed by Soon Yung Jun, Dongwook Jang

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class GflashHistogram;

class GFlash : public PhysicsList
{
public:
  GFlash(const edm::ParameterSet & p);
  ~GFlash() override;

private:
  GflashHistogram* theHisto;
  edm::ParameterSet thePar;

};

#endif

