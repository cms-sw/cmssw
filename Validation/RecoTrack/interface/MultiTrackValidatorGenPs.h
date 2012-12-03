#ifndef MultiTrackValidatorGenPs_h
#define MultiTrackValidatorGenPs_h

/** \class MultiTrackValidatorGenPs
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2011/02/02 11:41:16 $
 *  $Revision: 1.51 $
 *  \author cerati
 */

#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "CommonTools/RecoAlgos/interface/GenParticleSelector.h"

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
  GenParticleSelector gpSelector;				      

};


#endif
