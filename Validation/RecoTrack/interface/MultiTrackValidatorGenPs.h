#ifndef MultiTrackValidatorGenPs_h
#define MultiTrackValidatorGenPs_h

/** \class MultiTrackValidatorGenPs
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  \author cerati
 */

#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"

class MultiTrackValidatorGenPs : public MultiTrackValidator {
 public:
  /// Constructor
  MultiTrackValidatorGenPs(const edm::ParameterSet& pset);
  
  /// Destructor
  virtual ~MultiTrackValidatorGenPs();

  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& );

private:

  GenParticleCustomSelector gpSelector;				      
  edm::EDGetTokenT<reco::TrackToGenParticleAssociator> label_gen_associator;

};


#endif
