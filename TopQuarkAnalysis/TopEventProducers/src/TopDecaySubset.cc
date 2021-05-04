#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"

#include "FWCore/Common/interface/Provenance.h"

/// default constructor
TopDecaySubset::TopDecaySubset(const edm::ParameterSet& cfg)
    : srcToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("src"))),
      genEventInfo_srcToken_(mayConsume<GenEventInfoProduct>(edm::InputTag("generator"))),
      addRadiation_(cfg.getParameter<bool>("addRadiation")),
      showerModel_(kStart),
      runMode_(kRun1) {
  // mapping of the corresponding fillMode; see FillMode
  // enumerator of TopDecaySubset for available modes
  std::string mode = cfg.getParameter<std::string>("fillMode");
  if (mode == "kME")
    fillMode_ = kME;
  else if (mode == "kStable")
    fillMode_ = kStable;
  else
    throw cms::Exception("Configuration") << mode << " is not a supported FillMode!\n";

  mode = cfg.getParameter<std::string>("runMode");
  if (mode == "Run1")
    runMode_ = kRun1;
  else if (mode == "Run2")
    runMode_ = kRun2;
  else
    throw cms::Exception("Configuration") << mode << " is not a supported RunMode!\n";

  // produces a set of GenParticles following
  // the decay branch of top quarks to the first level of
  // stable decay products
  produces<reco::GenParticleCollection>();
}

/// default destructor
TopDecaySubset::~TopDecaySubset() {}

/// write output into the event
void TopDecaySubset::produce(edm::Event& event, const edm::EventSetup& setup) {
  // create target vector
  std::unique_ptr<reco::GenParticleCollection> target(new reco::GenParticleCollection);

  // get source collection
  edm::Handle<reco::GenParticleCollection> src;
  event.getByToken(srcToken_, src);

  // find out what generator we are dealing with
  if (showerModel_ == kStart && runMode_ == kRun2) {
    showerModel_ = checkShowerModel(event);
  }

  // find top quarks in list of input particles
  std::vector<const reco::GenParticle*> tops;
  if (runMode_ == kRun1)
    tops = findTops(*src);
  else
    tops = findPrimalTops(*src);

  // determine shower model (only in first event)
  if (showerModel_ == kStart && runMode_ == kRun1)
    showerModel_ = checkShowerModel(tops);

  if (showerModel_ != kNone) {
    // check sanity of W bosons
    if (runMode_ == kRun1)
      checkWBosons(tops);
    else {
      // nothing for the moment
    }

    // get ref product from the event
    const reco::GenParticleRefProd ref = event.getRefBeforePut<reco::GenParticleCollection>();
    // clear existing refs
    clearReferences();
    if (runMode_ == kRun1) {
      // fill output
      fillListing(tops, *target);
      // fill references
      fillReferences(ref, *target);
    } else {
      std::vector<const reco::GenParticle*> decaying_tops = findDecayingTops(*src);
      // fill output
      fillListing(tops, decaying_tops, *target);
      // fill references
      fillReferences(ref, *target);
    }
  }

  // write vectors to the event
  event.put(std::move(target));
}

/// find top quarks in list of input particles
std::vector<const reco::GenParticle*> TopDecaySubset::findTops(const reco::GenParticleCollection& parts) {
  std::vector<const reco::GenParticle*> tops;
  for (reco::GenParticleCollection::const_iterator t = parts.begin(); t != parts.end(); ++t) {
    if (std::abs(t->pdgId()) == TopDecayID::tID && t->status() == TopDecayID::unfrag)
      tops.push_back(&(*t));
  }
  return tops;
}

/// find primal top quarks (top quarks from the hard interaction)
/// for Pythia6 this is identical to findDecayingTops
std::vector<const reco::GenParticle*> TopDecaySubset::findPrimalTops(const reco::GenParticleCollection& parts) {
  std::vector<const reco::GenParticle*> tops;
  for (reco::GenParticleCollection::const_iterator t = parts.begin(); t != parts.end(); ++t) {
    if (std::abs(t->pdgId()) != TopDecayID::tID)
      continue;

    bool hasTopMother = false;
    for (unsigned idx = 0; idx < t->numberOfMothers(); ++idx) {
      if (std::abs(t->mother(idx)->pdgId()) == TopDecayID::tID)
        hasTopMother = true;
    }

    if (hasTopMother)  // not a primal top
      continue;
    tops.push_back(&(*t));
  }

  return tops;
}

/// find decaying top quarks (quarks that decay to qW)
/// for Pythia6 this is identical to findPrimalTops
std::vector<const reco::GenParticle*> TopDecaySubset::findDecayingTops(const reco::GenParticleCollection& parts) {
  std::vector<const reco::GenParticle*> tops;
  for (reco::GenParticleCollection::const_iterator t = parts.begin(); t != parts.end(); ++t) {
    if (std::abs(t->pdgId()) != TopDecayID::tID)
      continue;

    bool hasTopDaughter = false;
    for (unsigned idx = 0; idx < t->numberOfDaughters(); ++idx) {
      if (std::abs(t->daughter(idx)->pdgId()) == TopDecayID::tID)
        hasTopDaughter = true;
    }

    if (hasTopDaughter)  // not a decaying top
      continue;
    tops.push_back(&(*t));
  }

  return tops;
}

/// find W bosons that come from top quark decays
/// for Pythia6 this is identical to findDecayingW
const reco::GenParticle* TopDecaySubset::findPrimalW(const reco::GenParticle* top) {
  unsigned int w_index = 0;
  for (unsigned idx = 0; idx < top->numberOfDaughters(); ++idx) {
    if (std::abs(top->daughter(idx)->pdgId()) == TopDecayID::WID) {
      w_index = idx;
      break;
    }
  }
  return static_cast<const reco::GenParticle*>(top->daughter(w_index));
}

/// find W bosons that come from top quark decays and decay themselves (end of the MC chain)
/// for Pythia6 this is identical to findPrimalW
//const reco::GenParticle* TopDecaySubset::findDecayingW(
//		const reco::GenParticle* top) {
//	const reco::GenParticle* decaying_W = findLastParticleInChain(findPrimalW(top));
//	return findLastParticleInChain(findPrimalW(top));
//}

/// find the last particle in a (potentially) long chain of state transitions
/// e.g. top[status==22]-> top[status==44 -> top[status==44] ->
/// top[status==44] -> top[status==62]
/// this function would pick the top with status 62
const reco::GenParticle* TopDecaySubset::findLastParticleInChain(const reco::GenParticle* p) {
  int particleID = std::abs(p->pdgId());
  bool containsItself = false;
  unsigned int d_idx = 0;
  for (unsigned idx = 0; idx < p->numberOfDaughters(); ++idx) {
    if (std::abs(p->daughter(idx)->pdgId()) == particleID) {
      containsItself = true;
      d_idx = idx;
    }
  }

  if (!containsItself)
    return p;
  else {
    if (showerModel_ == kPythia) {
      // Pythia6 has a weird idea of W bosons (and maybe other particles)
      // W (status == 3) -> q qbar' W. The new W is status 2 and has no daughters
      if (p->status() == 3)
        return p;
    }
    return findLastParticleInChain(static_cast<const reco::GenParticle*>(p->daughter(d_idx)));
  }
}

/// check the decay chain for the exploited shower model
TopDecaySubset::ShowerModel TopDecaySubset::checkShowerModel(const std::vector<const reco::GenParticle*>& tops) const {
  for (std::vector<const reco::GenParticle*>::const_iterator it = tops.begin(); it != tops.end(); ++it) {
    const reco::GenParticle* top = *it;
    // check for kHerwig type showers: here the status 3 top quark will
    // have a single status 2 top quark as daughter, which has again 3
    // or more status 2 daughters
    if (top->numberOfDaughters() == 1) {
      if (top->begin()->pdgId() == top->pdgId() && top->begin()->status() == TopDecayID::stable &&
          top->begin()->numberOfDaughters() > 1)
        return kHerwig;
    }
    // check for kPythia type showers: here the status 3 top quark will
    // have all decay products and a status 2 top quark as daughters
    // the status 2 top quark will be w/o further daughters
    if (top->numberOfDaughters() > 1) {
      bool containsWBoson = false, containsQuarkDaughter = false;
      for (reco::GenParticle::const_iterator td = top->begin(); td != top->end(); ++td) {
        if (std::abs(td->pdgId()) < TopDecayID::tID)
          containsQuarkDaughter = true;
        if (std::abs(td->pdgId()) == TopDecayID::WID)
          containsWBoson = true;
      }
      if (containsQuarkDaughter && containsWBoson)
        return kPythia;
    }
  }
  // if neither Herwig nor Pythia like
  if (tops.empty())
    edm::LogInfo("decayChain") << " Failed to find top quarks in decay chain. Will assume that this a \n"
                               << " non-top sample and produce an empty decaySubset.\n";
  else
    throw edm::Exception(edm::errors::LogicError,
                         " Can not find back any of the supported hadronization models. Models \n"
                         " which are supported are:                                            \n"
                         " Pythia  LO(+PS): Top(status 3) --> WBoson(status 3), Quark(status 3)\n"
                         " Herwig NLO(+PS): Top(status 2) --> Top(status 3) --> Top(status 2)  \n");
  return kNone;
}

/// check the embedded MC information for the shower model
TopDecaySubset::ShowerModel TopDecaySubset::checkShowerModel(edm::Event& event) {
  edm::Handle<GenEventInfoProduct> genEvtInfoProduct;
  event.getByToken(genEventInfo_srcToken_, genEvtInfoProduct);

  std::string moduleName = "";
  if (genEvtInfoProduct.isValid()) {
    const edm::StableProvenance& prov = event.getStableProvenance(genEvtInfoProduct.id());
    moduleName = edm::moduleName(prov, event.processHistory());
  }

  ShowerModel shower(kStart);
  if (moduleName.find("Pythia6") != std::string::npos)
    shower = kPythia;
  else if (moduleName.find("Pythia8") != std::string::npos)
    shower = kPythia8;
  else if (moduleName.find("Herwig6") != std::string::npos)
    shower = kHerwig;
  else if (moduleName.find("ThePEG") != std::string::npos)
    // Herwig++
    shower = kHerwig;
  else if (moduleName.find("Sherpa") != std::string::npos)
    shower = kSherpa;
  else
    shower = kNone;
  return shower;
}
/// check whether the W boson is contained in the original gen particle listing
void TopDecaySubset::checkWBosons(std::vector<const reco::GenParticle*>& tops) const {
  unsigned nTops = tops.size();
  for (std::vector<const reco::GenParticle*>::iterator it = tops.begin(); it != tops.end();) {
    const reco::GenParticle* top = *it;
    bool isContained = false;
    bool expectedStatus = false;
    if (showerModel_ != kPythia && top->begin() == top->end())
      throw edm::Exception(edm::errors::LogicError, "showerModel_!=kPythia && top->begin()==top->end()\n");
    for (reco::GenParticle::const_iterator td = ((showerModel_ == kPythia) ? top->begin() : top->begin()->begin());
         td != ((showerModel_ == kPythia) ? top->end() : top->begin()->end());
         ++td) {
      if (std::abs(td->pdgId()) == TopDecayID::WID) {
        isContained = true;
        if (((showerModel_ == kPythia) ? td->status() == TopDecayID::unfrag : td->status() == TopDecayID::stable)) {
          expectedStatus = true;
          break;
        }
      }
    }
    if (!expectedStatus) {
      it = tops.erase(it);
      if (isContained)
        edm::LogInfo("decayChain") << " W boson does not have the expected status. This happens, e.g.,      \n"
                                   << " with MC@NLO in the case of additional ttbar pairs from radiated     \n"
                                   << " gluons. We hope everything is fine, remove the correspdonding top   \n"
                                   << " quark from our list since it is not part of the primary ttbar pair  \n"
                                   << " and try to continue.                                                \n";
    } else
      it++;
  }
  if (tops.empty() && nTops != 0)
    throw edm::Exception(edm::errors::LogicError,
                         " Did not find a W boson with appropriate status for any of the top   \n"
                         " quarks in this event. This means that the hadronization of the W    \n"
                         " boson might be screwed up or there is another problem with the      \n"
                         " particle listing in this MC sample. Please contact an expert.       \n");
}

/// fill output vector for full decay chain
void TopDecaySubset::fillListing(const std::vector<const reco::GenParticle*>& tops,
                                 reco::GenParticleCollection& target) {
  unsigned int statusFlag;
  // determine status flag of the new
  // particle depending on the FillMode
  fillMode_ == kME ? statusFlag = 3 : statusFlag = 2;

  for (std::vector<const reco::GenParticle*>::const_iterator it = tops.begin(); it != tops.end(); ++it) {
    const reco::GenParticle* t = *it;
    // if particle is top or anti-top
    std::unique_ptr<reco::GenParticle> topPtr(
        new reco::GenParticle(t->threeCharge(), p4(it, statusFlag), t->vertex(), t->pdgId(), statusFlag, false));
    target.push_back(*topPtr);
    ++motherPartIdx_;
    // keep the top index for the map to manage the daughter refs
    int iTop = motherPartIdx_;
    std::vector<int> topDaughters;
    // define the W boson index (to be set later) for the map to
    // manage the daughter refs
    int iW = 0;
    std::vector<int> wDaughters;
    // sanity check
    if (showerModel_ != kPythia && t->begin() == t->end())
      throw edm::Exception(edm::errors::LogicError, "showerModel_!=kPythia && t->begin()==t->end()\n");
    //iterate over top daughters
    for (reco::GenParticle::const_iterator td = ((showerModel_ == kPythia) ? t->begin() : t->begin()->begin());
         td != ((showerModel_ == kPythia) ? t->end() : t->begin()->end());
         ++td) {
      if (td->status() == TopDecayID::unfrag && std::abs(td->pdgId()) <= TopDecayID::bID) {
        // if particle is beauty or other quark
        std::unique_ptr<reco::GenParticle> bPtr(
            new reco::GenParticle(td->threeCharge(), p4(td, statusFlag), td->vertex(), td->pdgId(), statusFlag, false));
        target.push_back(*bPtr);
        // increment & push index of the top daughter
        topDaughters.push_back(++motherPartIdx_);
        if (addRadiation_) {
          addRadiation(motherPartIdx_, td, target);
        }
      }
      // sanity check
      if (showerModel_ != kPythia && td->begin() == td->end())
        throw edm::Exception(edm::errors::LogicError, "showerModel_!=kPythia && td->begin()==td->end()\n");
      reco::GenParticle::const_iterator buffer = (showerModel_ == kPythia) ? td : td->begin();
      if (buffer->status() == TopDecayID::unfrag && std::abs(buffer->pdgId()) == TopDecayID::WID) {
        // if particle is a W boson
        std::unique_ptr<reco::GenParticle> wPtr(new reco::GenParticle(
            buffer->threeCharge(), p4(buffer, statusFlag), buffer->vertex(), buffer->pdgId(), statusFlag, true));
        target.push_back(*wPtr);
        // increment & push index of the top daughter
        topDaughters.push_back(++motherPartIdx_);
        // keep the W idx for the map
        iW = motherPartIdx_;
        if (addRadiation_) {
          addRadiation(motherPartIdx_, buffer, target);
        }
        if (showerModel_ != kPythia && buffer->begin() == buffer->end())
          throw edm::Exception(edm::errors::LogicError, "showerModel_!=kPythia && buffer->begin()==buffer->end()\n");
        // iterate over W daughters
        for (reco::GenParticle::const_iterator wd =
                 ((showerModel_ == kPythia) ? buffer->begin() : buffer->begin()->begin());
             wd != ((showerModel_ == kPythia) ? buffer->end() : buffer->begin()->end());
             ++wd) {
          // make sure the W daughter is of status unfrag and not the W itself
          if (wd->status() == TopDecayID::unfrag && !(std::abs(wd->pdgId()) == TopDecayID::WID)) {
            std::unique_ptr<reco::GenParticle> qPtr(new reco::GenParticle(
                wd->threeCharge(), p4(wd, statusFlag), wd->vertex(), wd->pdgId(), statusFlag, false));
            target.push_back(*qPtr);
            // increment & push index of the top daughter
            wDaughters.push_back(++motherPartIdx_);
            if (wd->status() == TopDecayID::unfrag && std::abs(wd->pdgId()) == TopDecayID::tauID) {
              // add tau daughters if the particle is a tau pass
              // the daughter of the tau which is of status 2
              //addDaughters(motherPartIdx_, wd->begin(), target);
              // add tau daughters if the particle is a tau pass
              // the tau itself, which may add a tau daughter of
              // of status 2 to the listing
              addDaughters(motherPartIdx_, wd, target);
            }
          }
        }
      }
      if (addRadiation_ && buffer->status() == TopDecayID::stable &&
          (buffer->pdgId() == TopDecayID::glueID || std::abs(buffer->pdgId()) < TopDecayID::bID)) {
        // collect additional radiation from the top
        std::unique_ptr<reco::GenParticle> radPtr(new reco::GenParticle(
            buffer->threeCharge(), buffer->p4(), buffer->vertex(), buffer->pdgId(), statusFlag, false));
        target.push_back(*radPtr);
        // increment & push index of the top daughter
        topDaughters.push_back(++motherPartIdx_);
      }
    }
    // add potential sisters of the top quark;
    // only for top to prevent double counting
    if (t->numberOfMothers() > 0 && t->pdgId() == TopDecayID::tID) {
      for (reco::GenParticle::const_iterator ts = t->mother()->begin(); ts != t->mother()->end(); ++ts) {
        // loop over all daughters of the top mother i.e.
        // the two top quarks and their potential sisters
        if (std::abs(ts->pdgId()) != t->pdgId() && ts->pdgId() != t->mother()->pdgId()) {
          // add all further particles but the two top quarks and potential
          // cases where the mother of the top has itself as daughter
          reco::GenParticle* cand =
              new reco::GenParticle(ts->threeCharge(), ts->p4(), ts->vertex(), ts->pdgId(), ts->status(), false);
          std::unique_ptr<reco::GenParticle> sPtr(cand);
          target.push_back(*sPtr);
          if (ts->begin() != ts->end()) {
            // in case the sister has daughters increment
            // and add the first generation of daughters
            addDaughters(++motherPartIdx_, ts->begin(), target, false);
          }
        }
      }
    }
    // fill the map for the administration
    // of daughter indices
    refs_[iTop] = topDaughters;
    refs_[iW] = wDaughters;
  }
}

void TopDecaySubset::fillListing(const std::vector<const reco::GenParticle*>& primalTops,
                                 const std::vector<const reco::GenParticle*>& decayingTops,
                                 reco::GenParticleCollection& target) {
  std::vector<const reco::GenParticle*>::const_iterator top_start;
  std::vector<const reco::GenParticle*>::const_iterator top_end;
  if (fillMode_ == kME) {
    top_start = primalTops.begin();
    top_end = primalTops.end();
  } else {
    top_start = decayingTops.begin();
    top_end = decayingTops.end();
  }
  for (std::vector<const reco::GenParticle*>::const_iterator it = top_start; it != top_end; ++it) {
    const reco::GenParticle* t = *it;
    // summation might happen here
    std::unique_ptr<reco::GenParticle> topPtr(
        new reco::GenParticle(t->threeCharge(), t->p4(), t->vertex(), t->pdgId(), t->status(), false));
    target.push_back(*topPtr);
    ++motherPartIdx_;

    // keep the top index for the map to manage the daughter refs
    int iTop = motherPartIdx_;
    std::vector<int> topDaughters;
    // define the W boson index (to be set later) for the map to
    // manage the daughter refs
    int iW = 0;
    std::vector<int> wDaughters;
    const reco::GenParticle* final_top = findLastParticleInChain(t);

    //iterate over top daughters
    for (reco::GenParticle::const_iterator td = final_top->begin(); td != final_top->end(); ++td) {
      if (std::abs(td->pdgId()) <= TopDecayID::bID) {
        // if particle is beauty or other quark
        std::unique_ptr<reco::GenParticle> qPtr(
            new reco::GenParticle(td->threeCharge(), td->p4(), td->vertex(), td->pdgId(), td->status(), false));
        target.push_back(*qPtr);
        // increment & push index of the top daughter
        topDaughters.push_back(++motherPartIdx_);
        if (addRadiation_) {
          // for radation to be added we first need to
          // pick the last quark in the MC chain
          const reco::GenParticle* last_q = findLastParticleInChain(static_cast<const reco::GenParticle*>(&*td));
          addRadiation(motherPartIdx_, last_q, target);
        }
      } else if (std::abs(td->pdgId()) == TopDecayID::WID) {
        // ladies and gentleman, we have a W boson
        std::unique_ptr<reco::GenParticle> WPtr(
            new reco::GenParticle(td->threeCharge(), td->p4(), td->vertex(), td->pdgId(), td->status(), false));
        target.push_back(*WPtr);
        // increment & push index of the top daughter
        topDaughters.push_back(++motherPartIdx_);
        iW = motherPartIdx_;

        // next are the daughers of our W boson
        // for Pythia 6 this is wrong as the last W has no daughters at all!
        // instead the status 3 W has 3 daughters: q qbar' and W (WTF??!)
        const reco::GenParticle* decaying_W = findLastParticleInChain(static_cast<const reco::GenParticle*>(&*td));
        for (reco::GenParticle::const_iterator wd = decaying_W->begin(); wd != decaying_W->end(); ++wd) {
          if (!(std::abs(wd->pdgId()) == TopDecayID::WID)) {
            std::unique_ptr<reco::GenParticle> qPtr(
                new reco::GenParticle(wd->threeCharge(), wd->p4(), wd->vertex(), wd->pdgId(), wd->status(), false));
            target.push_back(*qPtr);
            // increment & push index of the top daughter
            wDaughters.push_back(++motherPartIdx_);
            const reco::GenParticle* last_q = findLastParticleInChain(static_cast<const reco::GenParticle*>(&*wd));
            addRadiation(motherPartIdx_, last_q, target);
            if (std::abs(wd->pdgId()) == TopDecayID::tauID) {
              // add tau daughters
              // currently it adds k-mesons etc as well, which
              // is not what we want.
              addDaughters(motherPartIdx_, wd, target);
            }
          }
        }

      } else {
        if (addRadiation_ && (td->pdgId() == TopDecayID::glueID || std::abs(td->pdgId()) < TopDecayID::bID)) {
          // collect additional radiation from the top
          std::unique_ptr<reco::GenParticle> radPtr(
              new reco::GenParticle(td->threeCharge(), td->p4(), td->vertex(), td->pdgId(), td->status(), false));
          target.push_back(*radPtr);
        }
        //other top daughters like Zq for FCNC
        // for pythia 6 many gluons end up here
        //std::cout << "other top daughters: to be implemented"
        //		<< std::endl;
      }
    }

    // fill the map for the administration
    // of daughter indices
    refs_[iTop] = topDaughters;
    refs_[iW] = wDaughters;
  }
}

/// calculate lorentz vector from input
reco::Particle::LorentzVector TopDecaySubset::p4(const std::vector<const reco::GenParticle*>::const_iterator top,
                                                 int statusFlag) {
  // calculate the four vector for top/anti-top quarks from
  // the W boson and the b quark plain or including all
  // additional radiation depending on switch 'plain'
  if (statusFlag == TopDecayID::unfrag) {
    // return 4 momentum as it is
    return (*top)->p4();
  }
  reco::Particle::LorentzVector vec;
  for (reco::GenParticle::const_iterator p = (*top)->begin(); p != (*top)->end(); ++p) {
    if (p->status() == TopDecayID::unfrag) {
      // descend by one level for each
      // status 3 particle on the way
      vec += p4(p, statusFlag);
    } else {
      if (std::abs((*top)->pdgId()) == TopDecayID::WID) {
        // in case of a W boson skip the status
        // 2 particle to prevent double counting
        if (std::abs(p->pdgId()) != TopDecayID::WID)
          vec += p->p4();
      } else {
        // add all four vectors for each stable
        // particle (status 1 or 2) on the way
        vec += p->p4();
        if (vec.mass() - (*top)->mass() > 0) {
          // continue adding up gluons and qqbar pairs on the top
          // line untill the nominal top mass is reached; then
          // break in order to prevent picking up virtualities
          break;
        }
      }
    }
  }
  return vec;
}

/// calculate lorentz vector from input (dedicated to top reconstruction)
reco::Particle::LorentzVector TopDecaySubset::p4(const reco::GenParticle::const_iterator part, int statusFlag) {
  // calculate the four vector for all top daughters from
  // their daughters including additional radiation
  if (statusFlag == TopDecayID::unfrag) {
    // return 4 momentum as it is
    return part->p4();
  }
  reco::Particle::LorentzVector vec;
  for (reco::GenParticle::const_iterator p = part->begin(); p != part->end(); ++p) {
    if (p->status() <= TopDecayID::stable && std::abs(p->pdgId()) == TopDecayID::WID) {
      vec = p->p4();
    } else {
      if (p->status() <= TopDecayID::stable) {
        // sum up the p4 of all stable particles
        // (of status 1 or 2)
        vec += p->p4();
      } else {
        if (p->status() == TopDecayID::unfrag) {
          // if the particle is unfragmented (i.e.
          // status 3) descend by one level
          vec += p4(p, statusFlag);
        }
      }
    }
  }
  return vec;
}

/// fill vector including all radiations from quarks originating from W/top
void TopDecaySubset::addRadiation(int& idx,
                                  const reco::GenParticle::const_iterator part,
                                  reco::GenParticleCollection& target) {
  std::vector<int> daughters;
  int idxBuffer = idx;
  for (reco::GenParticle::const_iterator daughter = part->begin(); daughter != part->end(); ++daughter) {
    if (daughter->status() <= TopDecayID::stable && daughter->pdgId() != part->pdgId()) {
      // skip comment lines and make sure that
      // the particle is not double counted as
      // daughter of itself
      std::unique_ptr<reco::GenParticle> ptr(new reco::GenParticle(
          daughter->threeCharge(), daughter->p4(), daughter->vertex(), daughter->pdgId(), daughter->status(), false));
      target.push_back(*ptr);
      daughters.push_back(++idx);  //push index of daughter
    }
  }
  if (!daughters.empty()) {
    refs_[idxBuffer] = daughters;
  }
}

void TopDecaySubset::addRadiation(int& idx, const reco::GenParticle* part, reco::GenParticleCollection& target) {
  std::vector<int> daughters;
  int idxBuffer = idx;
  for (reco::GenParticle::const_iterator daughter = part->begin(); daughter != part->end(); ++daughter) {
    // either pick daughters as long as they are different
    // to the initial particle
    if (daughter->pdgId() != part->pdgId()) {
      std::unique_ptr<reco::GenParticle> ptr(new reco::GenParticle(
          daughter->threeCharge(), daughter->p4(), daughter->vertex(), daughter->pdgId(), daughter->status(), false));
      target.push_back(*ptr);
      daughters.push_back(++idx);  //push index of daughter
    }
  }
  if (!daughters.empty()) {
    refs_[idxBuffer] = daughters;
  }
}

/// recursively fill vector for all further decay particles of a given particle
void TopDecaySubset::addDaughters(int& idx,
                                  const reco::GenParticle::const_iterator part,
                                  reco::GenParticleCollection& target,
                                  bool recursive) {
  std::vector<int> daughters;
  int idxBuffer = idx;
  for (reco::GenParticle::const_iterator daughter = part->begin(); daughter != part->end(); ++daughter) {
    std::unique_ptr<reco::GenParticle> ptr(new reco::GenParticle(
        daughter->threeCharge(), daughter->p4(), daughter->vertex(), daughter->pdgId(), daughter->status(), false));
    target.push_back(*ptr);
    // increment & push index of daughter
    daughters.push_back(++idx);
    // continue recursively if desired
    if (recursive) {
      addDaughters(idx, daughter, target);
    }
  }
  if (!daughters.empty()) {
    refs_[idxBuffer] = daughters;
  }
}

/// clear references
void TopDecaySubset::clearReferences() {
  // clear vector of references
  refs_.clear();
  // set idx for mother particles to a start value
  // of -1 (the first entry will raise it to 0)
  motherPartIdx_ = -1;
}

/// fill references for output vector
void TopDecaySubset::fillReferences(const reco::GenParticleRefProd& ref, reco::GenParticleCollection& sel) {
  int idx = 0;
  for (reco::GenParticleCollection::iterator p = sel.begin(); p != sel.end(); ++p, ++idx) {
    //find daughter reference vectors in refs_ and add daughters
    std::map<int, std::vector<int> >::const_iterator daughters = refs_.find(idx);
    if (daughters != refs_.end()) {
      for (std::vector<int>::const_iterator daughter = daughters->second.begin(); daughter != daughters->second.end();
           ++daughter) {
        reco::GenParticle* part = dynamic_cast<reco::GenParticle*>(&(*p));
        if (part == nullptr) {
          throw edm::Exception(edm::errors::InvalidReference, "Not a GenParticle");
        }
        part->addDaughter(reco::GenParticleRef(ref, *daughter));
        sel[*daughter].addMother(reco::GenParticleRef(ref, idx));
      }
    }
  }
}
