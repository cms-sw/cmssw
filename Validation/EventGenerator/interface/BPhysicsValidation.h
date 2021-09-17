#ifndef BPhysicsValidation_H
#define BPhysicsValidation_H

/*class BPhysicsValidation
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

class BPhysicsValidation : public DQMEDAnalyzer {
public:
  explicit BPhysicsValidation(const edm::ParameterSet &);
  ~BPhysicsValidation() override;

  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;

private:
  class ParticleMonitor {
  public:
    ParticleMonitor(std::string name_, const edm::ParameterSet &p_)
        : p(p_), name(name_), pdgid(p.getParameter<int>("pdgid")){};
    ~ParticleMonitor(){};

    void Configure(DQMStore::IBooker &i) {
      std::string pname = p.getParameter<std::string>("pname");
      double mass_min = p.getParameter<double>("massmin");
      double mass_max = p.getParameter<double>("massmax");
      DQMHelper dqm(&i);
      i.setCurrentFolder("Generator/BPhysics");
      // Number of analyzed events
      pt = dqm.book1dHisto(name + "PT", "P_{t} of the " + pname + "s", 100, 0., 100, "P_{t} (GeV)", "Number of Events");
      eta = dqm.book1dHisto(name + "ETA", "#eta of the " + pname + "s", 100, -5., 5., "#eta", "Number of Events");
      phi = dqm.book1dHisto(
          name + "PHI", "#phi of the " + pname + "s", 100, 0, 2 * TMath::Pi(), "#phi", "Number of Events");
      mass = dqm.book1dHisto(
          name + "MASS", "Mass of the " + pname + "s", 100, mass_min, mass_max, "Mass (GeV)", "Number of Events");
    }

    void Fill(const reco::GenParticle *p, double weight) {
      if (abs(p->pdgId()) == abs(pdgid)) {
        pt->Fill(p->pt(), weight);
        eta->Fill(p->eta(), weight);
        phi->Fill(p->phi(), weight);
        mass->Fill(p->mass(), weight);
      }
    }
    int PDGID() { return pdgid; }

  private:
    const edm::ParameterSet p;
    std::string name;
    int pdgid;
    MonitorElement *pt, *eta, *phi, *mass;
  };

  void FillDaughters(const reco::GenParticle *p);
  edm::InputTag genparticleCollection_;
  edm::EDGetTokenT<reco::GenParticleCollection> genparticleCollectionToken_;
  std::string name;
  ParticleMonitor particle;
  std::vector<ParticleMonitor> daughters;
  MonitorElement *Nobj;
};

#endif
