#ifndef SimG4Core_PhysicsLists_CMSFTFPPiKBuilder_h
#define SimG4Core_PhysicsLists_CMSFTFPPiKBuilder_h 1

#include "globals.hh"

#include "G4HadronElasticProcess.hh"
#include "G4HadronFissionProcess.hh"
#include "G4HadronCaptureProcess.hh"
#include "G4NeutronInelasticProcess.hh"
#include "G4VPiKBuilder.hh"

#include "G4TheoFSGenerator.hh"
#include "G4ExcitationHandler.hh"
#include "G4PreCompoundModel.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4FTFModel.hh"
#include "G4LundStringFragmentation.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4QuasiElasticChannel.hh"

#include "G4PiNuclearCrossSection.hh"

class CMSFTFPPiKBuilder : public G4VPiKBuilder {
public:
  CMSFTFPPiKBuilder(G4bool quasiElastic = false);
  ~CMSFTFPPiKBuilder() override;

public:
  void Build(G4HadronElasticProcess* aP) override;
  void Build(G4PionPlusInelasticProcess* aP) override;
  void Build(G4PionMinusInelasticProcess* aP) override;
  void Build(G4KaonPlusInelasticProcess* aP) override;
  void Build(G4KaonMinusInelasticProcess* aP) override;
  void Build(G4KaonZeroLInelasticProcess* aP) override;
  void Build(G4KaonZeroSInelasticProcess* aP) override;

  void SetMinEnergy(G4double aM) override { theMin = aM; }
  void SetMaxEnergy(G4double aM) override { theMax = aM; }

private:
  G4TheoFSGenerator* theModel;
  G4PreCompoundModel* thePreEquilib;
  G4GeneratorPrecompoundInterface* theCascade;
  G4FTFModel* theStringModel;
  G4ExcitedStringDecay* theStringDecay;
  G4QuasiElasticChannel* theQuasiElastic;

  G4PiNuclearCrossSection* thePiData;
  G4double theMin;
  G4double theMax;
};

// 2002 by J.P. Wellisch

#endif
