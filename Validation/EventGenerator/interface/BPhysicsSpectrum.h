#ifndef BPhysicsSpectrum_H
#define BPhysicsSpectrum_H

/*class BPhysicsSpectrum
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *
 */
#include <iostream>
#include "TMath.h"
// framework & common header files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "Validation/EventGenerator/interface/DQMHelper.h"

class BPhysicsSpectrum : public DQMEDAnalyzer {
public:
  explicit BPhysicsSpectrum(const edm::ParameterSet &);
  ~BPhysicsSpectrum() override;

  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;

private:
  MonitorElement *mass, *Nobj;
  edm::InputTag genparticleCollection_;
  edm::EDGetTokenT<reco::GenParticleCollection> genparticleCollectionToken_;
  std::string name;
  double mass_min, mass_max;
  std::vector<int> Particles;
};

#endif
