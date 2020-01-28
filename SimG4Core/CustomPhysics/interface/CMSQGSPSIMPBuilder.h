#ifndef SimG4Core_CustomPhysics_CMSQGSPSIMPBuilder_H
#define SimG4Core_CustomPhysics_CMSQGSPSIMPBuilder_H

#include "globals.hh"

#include "G4QGSParticipants.hh"
#include "G4QGSMFragmentation.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4QGSModel.hh"

class CMSSIMPInelasticProcess;
class G4QGSParticipants;
class G4QGSMFragmentation;
class G4ExcitedStringDecay;

class CMSQGSPSIMPBuilder {
public:
  CMSQGSPSIMPBuilder();
  ~CMSQGSPSIMPBuilder();

  void Build(CMSSIMPInelasticProcess* aP);

private:
  G4QGSModel<G4QGSParticipants>* theStringModel;
  G4ExcitedStringDecay* theStringDecay;
  G4QGSMFragmentation* theQGSM;
};

#endif
