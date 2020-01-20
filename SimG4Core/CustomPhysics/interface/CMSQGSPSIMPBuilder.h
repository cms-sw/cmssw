#ifndef SimG4Core_CustomPhysics_CMSQGSPSIMPBuilder_H
#define SimG4Core_CustomPhysics_CMSQGSPSIMPBuilder_H

#include "globals.hh"

#include "G4HadronElasticProcess.hh"
#include "G4HadronFissionProcess.hh"
#include "G4HadronCaptureProcess.hh"
#include "SimG4Core/CustomPhysics/interface/CMSSIMPInelasticProcess.h"

#include "G4TheoFSGenerator.hh"
#include "G4PreCompoundModel.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4QGSModel.hh"
#include "G4QGSParticipants.hh"
#include "G4QGSMFragmentation.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4QuasiElasticChannel.hh"

class CMSQGSPSIMPBuilder {
public:
  CMSQGSPSIMPBuilder(G4bool quasiElastic = false);
  virtual ~CMSQGSPSIMPBuilder();

public:
  virtual void Build(G4HadronElasticProcess* aP);
  virtual void Build(G4HadronFissionProcess* aP);
  virtual void Build(G4HadronCaptureProcess* aP);
  virtual void Build(CMSSIMPInelasticProcess* aP);

  void SetMinEnergy(G4double aM) { theMin = aM; }

private:
  G4TheoFSGenerator* theModel;
  G4PreCompoundModel* thePreEquilib;
  G4GeneratorPrecompoundInterface* theCascade;
  G4QGSModel<G4QGSParticipants>* theStringModel;
  G4ExcitedStringDecay* theStringDecay;
  G4QuasiElasticChannel* theQuasiElastic;

  G4QGSMFragmentation* theQGSM;

  G4double theMin;
};

#endif
