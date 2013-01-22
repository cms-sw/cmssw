#ifndef TauAnalysis_MCEmbeddingTools_GenMuonRadiationAlgorithm_h
#define TauAnalysis_MCEmbeddingTools_GenMuonRadiationAlgorithm_h

/** \class GenMuonRadiationAlgorithm
 *
 * Auxiliary class to correct for muon --> muon + photon final state radiation (FSR) in selected Z --> mu+ mu- events.
 * The FSR is estimated on event-by-event basis via a Monte Carlo technique:
 * for each reconstructed muon, PHOTOS is used to obtain a random amount of radiated photon energy.
 * The radiated photon energy is then added to the energy of the reconstructed muon
 * and the energy of generator level tau leptons is set to the sum.
 *
 * NOTE: FSR increases with the energy of the muon.
 *       So, in principle the energy of muon + photon (energy of muon before FSR), 
 *       not the energy of the reconstructed muon (energy of muon after FSR) would need to be used as input to PHOTOS.
 *       As the amount of radiated photon energy is typically small (< 1 GeV on average),
 *       taking the energy of the reconstructed muon is a good approximation.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.5 $
 *
 * $Id: GenMuonRadiationAlgorithm.h,v 1.5 2013/01/04 15:36:25 veelken Exp $
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"
#include "GeneratorInterface/ExternalDecays/interface/PhotosInterface.h"

#include<string>

class GenMuonRadiationAlgorithm
{
 public:
  explicit GenMuonRadiationAlgorithm(const edm::ParameterSet&);
  ~GenMuonRadiationAlgorithm() {}

  reco::Candidate::LorentzVector compFSR(const reco::Candidate::LorentzVector&, int);

 private:
  double beamEnergy_;

  gen::PhotosInterface photos_;
  static bool photos_isInitialized_;

  int verbosity_;
};

#endif


