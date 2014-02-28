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
 * \version $Revision: 1.6 $
 *
 * $Id: GenMuonRadiationAlgorithm.h,v 1.6 2013/01/27 13:53:44 veelken Exp $
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosInterfaceBase.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosFactory.h"

#include<string>

class myPythia6ServiceWithCallback;

class GenMuonRadiationAlgorithm
{
 public:
  explicit GenMuonRadiationAlgorithm(const edm::ParameterSet&);
  ~GenMuonRadiationAlgorithm();

  reco::Candidate::LorentzVector compFSR(const edm::StreamID& streamID, const reco::Candidate::LorentzVector&, int, const reco::Candidate::LorentzVector&, int&);

 private:
  double beamEnergy_;

  enum { kPYTHIA, kPHOTOS };
  int mode_;

  gen::PhotosInterfaceBase* photos_;
  static bool photos_isInitialized_;

  myPythia6ServiceWithCallback* pythia_;
  static bool pythia_isInitialized_;

  int verbosity_;
};

#endif


