#ifndef MultiTrackValidatorGenPs_h
#define MultiTrackValidatorGenPs_h

/** \class MultiTrackValidatorGenPs
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2012/12/03 10:53:20 $
 *  $Revision: 1.1 $
 *  \author cerati
 */

#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"

class MultiTrackValidatorGenPs : public MultiTrackValidator {
 public:
  /// Constructor
  MultiTrackValidatorGenPs(const edm::ParameterSet& pset);
  
  /// Destructor
  virtual ~MultiTrackValidatorGenPs();

  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& );

private:

  const TrackAssociatorByChi2* associatorByChi2;
  GenParticleCustomSelector gpSelector;				      

};


#endif
