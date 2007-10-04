#ifndef GflashHadronShowerModel_H
#define GflashHadronShowerModel_H

#include "G4VFastSimulationModel.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"

#include "TFile.h"
#include "TH1.h"

class GflashHadronShowerProfile;
class GFlashHitMaker;

class GflashHadronShowerModel : public G4VFastSimulationModel
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashHadronShowerModel (G4String modelName, G4Region* envelope);
  ~GflashHadronShowerModel ();

  //------------------------------------------------------------------------
  // Virtual methods that should be implemented for this hadron shower model
  //------------------------------------------------------------------------

  G4bool IsApplicable(const G4ParticleDefinition&);
  G4bool ModelTrigger(const G4FastTrack &);
  void DoIt(const G4FastTrack&, G4FastStep&);

private:
  G4bool isFirstInelasticInteraction(const G4FastTrack& fastTrack);
  G4bool excludeDetectorRegion(const G4FastTrack& fastTrack);

private:  
  GflashHadronShowerProfile *theProfile;
  GFlashHitMaker *theHitMaker;

  //debugging histograms
 
  TFile* f;

  TH1F*    h_enormal;
  TH1F*    h_ewrapper;
  
  TH1F*    h_energy;
  TH1F*    h_esum;
  TH1F*    h_elos;
  TH1F*    h_ssp;
  TH1F*    h_ratio;
  TH1F*    h_dout;

};

#endif
