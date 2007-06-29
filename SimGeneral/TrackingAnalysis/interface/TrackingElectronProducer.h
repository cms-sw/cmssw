#ifndef TrackingAnalysis_TrackingElectronProducer_h
#define TrackingAnalysis_TrackingElectronProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Utilities/Timing/interface/TimingReport.h"
//#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

/** Produces one composite TrackingParticle per electron, 
 *  for electron tracking studies: 
 *   - for each generator or Geant electron, finds the list of track segments 
 *     before first brem, in between brems, and after last brem. 
 *   - creates 1 composite TrackingParticle per list of segments. 
 */
class TrackingElectronProducer : public edm::EDProducer {

public:

  explicit TrackingElectronProducer( const edm::ParameterSet & );
//  ~TrackingElectronProducer() { TimingReport::current()->dump(std::cout); }

private:

  void produce( edm::Event &, const edm::EventSetup & );

  int layerFromDetid(const unsigned int&);

  void listElectrons(const TrackingParticleCollection & tPC) const;

  void addG4Track(TrackingParticle&, const TrackingParticle *) const;

  edm::ParameterSet conf_;
  
};

#endif
