#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/TtFullLepKinSolver.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtDilepLRSignalSelObservables.h"

#include <memory>
#include <string>
#include <vector>

class TtDilepEvtSolutionMaker : public edm::stream::EDProducer<> {
public:
  explicit TtDilepEvtSolutionMaker(const edm::ParameterSet& iConfig);
  ~TtDilepEvtSolutionMaker() override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  // next methods are avoidable but they make the code legible
  inline bool PTComp(const reco::Candidate*, const reco::Candidate*) const;
  inline bool LepDiffCharge(const reco::Candidate*, const reco::Candidate*) const;
  inline bool HasPositiveCharge(const reco::Candidate*) const;

private:
  edm::EDGetTokenT<std::vector<pat::Electron> > electronSourceToken_;
  edm::EDGetTokenT<std::vector<pat::Muon> > muonSourceToken_;
  edm::EDGetTokenT<std::vector<pat::Tau> > tauSourceToken_;
  edm::EDGetTokenT<std::vector<pat::MET> > metSourceToken_;
  edm::EDGetTokenT<std::vector<pat::Jet> > jetSourceToken_;
  edm::EDGetTokenT<TtGenEvent> evtSourceToken_;
  int jetCorrScheme_;
  unsigned int nrCombJets_;
  bool matchToGenEvt_, calcTopMass_, useMCforBest_;
  bool eeChannel_, emuChannel_, mumuChannel_, etauChannel_, mutauChannel_, tautauChannel_;
  double tmassbegin_, tmassend_, tmassstep_;
  std::vector<double> nupars_;

  TtDilepLRSignalSelObservables* myLRSignalSelObservables;
  TtFullLepKinSolver* solver;
};

inline bool TtDilepEvtSolutionMaker::PTComp(const reco::Candidate* l1, const reco::Candidate* l2) const {
  return (l1->pt() > l2->pt());
}

inline bool TtDilepEvtSolutionMaker::LepDiffCharge(const reco::Candidate* l1, const reco::Candidate* l2) const {
  return (l1->charge() != l2->charge());
}

inline bool TtDilepEvtSolutionMaker::HasPositiveCharge(const reco::Candidate* l) const { return (l->charge() > 0); }

TtDilepEvtSolutionMaker::TtDilepEvtSolutionMaker(const edm::ParameterSet& iConfig) {
  // configurables
  electronSourceToken_ = consumes<std::vector<pat::Electron> >(iConfig.getParameter<edm::InputTag>("electronSource"));
  muonSourceToken_ = consumes<std::vector<pat::Muon> >(iConfig.getParameter<edm::InputTag>("muonSource"));
  tauSourceToken_ = consumes<std::vector<pat::Tau> >(iConfig.getParameter<edm::InputTag>("tauSource"));
  metSourceToken_ = consumes<std::vector<pat::MET> >(iConfig.getParameter<edm::InputTag>("metSource"));
  jetSourceToken_ = consumes<std::vector<pat::Jet> >(iConfig.getParameter<edm::InputTag>("jetSource"));
  jetCorrScheme_ = iConfig.getParameter<int>("jetCorrectionScheme");
  evtSourceToken_ = mayConsume<TtGenEvent>(iConfig.getParameter<edm::InputTag>("evtSource"));
  nrCombJets_ = iConfig.getParameter<unsigned int>("nrCombJets");
  matchToGenEvt_ = iConfig.getParameter<bool>("matchToGenEvt");
  calcTopMass_ = iConfig.getParameter<bool>("calcTopMass");
  useMCforBest_ = iConfig.getParameter<bool>("bestSolFromMC");
  eeChannel_ = iConfig.getParameter<bool>("eeChannel");
  emuChannel_ = iConfig.getParameter<bool>("emuChannel");
  mumuChannel_ = iConfig.getParameter<bool>("mumuChannel");
  mutauChannel_ = iConfig.getParameter<bool>("mutauChannel");
  etauChannel_ = iConfig.getParameter<bool>("etauChannel");
  tautauChannel_ = iConfig.getParameter<bool>("tautauChannel");
  tmassbegin_ = iConfig.getParameter<double>("tmassbegin");
  tmassend_ = iConfig.getParameter<double>("tmassend");
  tmassstep_ = iConfig.getParameter<double>("tmassstep");
  nupars_ = iConfig.getParameter<std::vector<double> >("neutrino_parameters");

  // define what will be produced
  produces<std::vector<TtDilepEvtSolution> >();

  myLRSignalSelObservables = new TtDilepLRSignalSelObservables(consumesCollector(), jetSourceToken_);

  solver = new TtFullLepKinSolver(tmassbegin_, tmassend_, tmassstep_, nupars_);
}

TtDilepEvtSolutionMaker::~TtDilepEvtSolutionMaker() {}

void TtDilepEvtSolutionMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<pat::Tau> > taus;
  iEvent.getByToken(tauSourceToken_, taus);
  edm::Handle<std::vector<pat::Muon> > muons;
  iEvent.getByToken(muonSourceToken_, muons);
  edm::Handle<std::vector<pat::Electron> > electrons;
  iEvent.getByToken(electronSourceToken_, electrons);
  edm::Handle<std::vector<pat::MET> > mets;
  iEvent.getByToken(metSourceToken_, mets);
  edm::Handle<std::vector<pat::Jet> > jets;
  iEvent.getByToken(jetSourceToken_, jets);

  int selMuonp = -1, selMuonm = -1;
  int selElectronp = -1, selElectronm = -1;
  int selTaup = -1, selTaum = -1;
  bool leptonFound = false;
  bool mumu = false;
  bool emu = false;
  bool ee = false;
  bool etau = false;
  bool mutau = false;
  bool tautau = false;
  bool leptonFoundEE = false;
  bool leptonFoundMM = false;
  bool leptonFoundTT = false;
  bool leptonFoundEpMm = false;
  bool leptonFoundEmMp = false;
  bool leptonFoundEpTm = false;
  bool leptonFoundEmTp = false;
  bool leptonFoundMpTm = false;
  bool leptonFoundMmTp = false;
  bool jetsFound = false;
  bool METFound = false;
  std::vector<int> JetVetoByTaus;

  //select MET (TopMET vector is sorted on ET)
  if (!mets->empty()) {
    METFound = true;
  }

  // If we have electrons and muons available,
  // build a solutions with electrons and muons.
  if (muons->size() + electrons->size() >= 2) {
    // select leptons
    if (electrons->empty())
      mumu = true;
    else if (muons->empty())
      ee = true;
    else if (electrons->size() == 1) {
      if (muons->size() == 1)
        emu = true;
      else if (PTComp(&(*electrons)[0], &(*muons)[1]))
        emu = true;
      else
        mumu = true;
    } else if (electrons->size() > 1) {
      if (PTComp(&(*electrons)[1], &(*muons)[0]))
        ee = true;
      else if (muons->size() == 1)
        emu = true;
      else if (PTComp(&(*electrons)[0], &(*muons)[1]))
        emu = true;
      else
        mumu = true;
    }
    if (ee) {
      if (LepDiffCharge(&(*electrons)[0], &(*electrons)[1])) {
        leptonFound = true;
        leptonFoundEE = true;
        if (HasPositiveCharge(&(*electrons)[0])) {
          selElectronp = 0;
          selElectronm = 1;
        } else {
          selElectronp = 1;
          selElectronm = 0;
        }
      }
    } else if (emu) {
      if (LepDiffCharge(&(*electrons)[0], &(*muons)[0])) {
        leptonFound = true;
        if (HasPositiveCharge(&(*electrons)[0])) {
          leptonFoundEpMm = true;
          selElectronp = 0;
          selMuonm = 0;
        } else {
          leptonFoundEmMp = true;
          selMuonp = 0;
          selElectronm = 0;
        }
      }
    } else if (mumu) {
      if (LepDiffCharge(&(*muons)[0], &(*muons)[1])) {
        leptonFound = true;
        leptonFoundMM = true;
        if (HasPositiveCharge(&(*muons)[0])) {
          selMuonp = 0;
          selMuonm = 1;
        } else {
          selMuonp = 1;
          selMuonm = 0;
        }
      }
    }
    //select Jets (TopJet vector is sorted on ET)
    if (jets->size() >= 2) {
      jetsFound = true;
    }
  }
  // If a tau is needed to have two leptons, then only consider the taus.
  // This is the minimal modification of the dilept selection that includes taus,
  // since we are considering taus only when no other solution exist.
  else if (muons->size() + electrons->size() == 1 && !taus->empty()) {
    // select leptons
    if (muons->size() == 1) {
      mutau = true;
      // depending on the muon charge, set the right muon index and specify channel
      int expectedCharge = -muons->begin()->charge();
      int* tauIdx = nullptr;
      if (expectedCharge < 0) {
        selMuonp = 0;
        tauIdx = &selTaum;
        leptonFoundMpTm = true;
      } else {
        selMuonm = 0;
        tauIdx = &selTaup;
        leptonFoundMmTp = true;
      }
      // loop over the vector of taus to find the ones
      // that have the charge opposite to the muon one, and do not match in eta-phi
      std::vector<std::vector<pat::Tau>::const_iterator> subset1;
      for (std::vector<pat::Tau>::const_iterator tau = taus->begin(); tau < taus->end(); ++tau) {
        if (tau->charge() * expectedCharge >= 0 && DeltaR<pat::Particle>()(*tau, *(muons->begin())) > 0.1) {
          *tauIdx = tau - taus->begin();
          leptonFound = true;
          subset1.push_back(tau);
        }
      }
      // if there are more than one tau with ecalIsol==0, take the smallest E/P
      float iso = 999.;
      for (std::vector<std::vector<pat::Tau>::const_iterator>::const_iterator tau = subset1.begin();
           tau < subset1.end();
           ++tau) {
        if ((*tau)->isPFTau() && (*tau)->isolationPFChargedHadrCandsPtSum() < iso) {
          *tauIdx = *tau - taus->begin();
          iso = (*tau)->isolationPFChargedHadrCandsPtSum();
        }
      }

      // check that one combination has been found
      if (!leptonFound) {
        leptonFoundMpTm = false;
        leptonFoundMmTp = false;
      }
      // discard the jet that matches the tau (if one)
      if (leptonFound) {
        for (std::vector<pat::Jet>::const_iterator jet = jets->begin(); jet < jets->end(); ++jet) {
          if (DeltaR<pat::Particle, pat::Jet>()(*(taus->begin() + *tauIdx), *jet) < 0.1) {
            JetVetoByTaus.push_back(jet - jets->begin());
          }
        }
      }
    } else {
      etau = true;
      // depending on the electron charge, set the right electron index and specify channel
      int expectedCharge = -electrons->begin()->charge();
      int* tauIdx = nullptr;
      if (expectedCharge < 0) {
        selElectronp = 0;
        tauIdx = &selTaum;
        leptonFoundEpTm = true;
      } else {
        selElectronm = 0;
        tauIdx = &selTaup;
        leptonFoundEmTp = true;
      }
      // loop over the vector of taus to find the ones
      // that have the charge opposite to the muon one, and do not match in eta-phi
      std::vector<std::vector<pat::Tau>::const_iterator> subset1;
      for (std::vector<pat::Tau>::const_iterator tau = taus->begin(); tau < taus->end(); ++tau) {
        if (tau->charge() * expectedCharge >= 0 && DeltaR<pat::Particle>()(*tau, *(electrons->begin())) > 0.1) {
          *tauIdx = tau - taus->begin();
          leptonFound = true;
          subset1.push_back(tau);
        }
      }
      // if there are more than one tau with ecalIsol==0, take the smallest E/P
      float iso = 999.;
      for (std::vector<std::vector<pat::Tau>::const_iterator>::const_iterator tau = subset1.begin();
           tau < subset1.end();
           ++tau) {
        if ((*tau)->isPFTau() && (*tau)->isolationPFChargedHadrCandsPtSum() < iso) {
          *tauIdx = *tau - taus->begin();
          iso = (*tau)->isolationPFChargedHadrCandsPtSum();
        }
      }

      // check that one combination has been found
      if (!leptonFound) {
        leptonFoundEpTm = false;
        leptonFoundEmTp = false;
      }
      // discard the jet that matches the tau (if one)
      if (leptonFound) {
        for (std::vector<pat::Jet>::const_iterator jet = jets->begin(); jet < jets->end(); ++jet) {
          if (DeltaR<pat::Particle, pat::Jet>()(*(taus->begin() + *tauIdx), *jet) < 0.1) {
            JetVetoByTaus.push_back(jet - jets->begin());
          }
        }
      }
    }
    // select Jets (TopJet vector is sorted on ET)
    jetsFound = ((jets->size() - JetVetoByTaus.size()) >= 2);
  } else if (taus->size() > 1) {
    tautau = true;
    if (LepDiffCharge(&(*taus)[0], &(*taus)[1])) {
      leptonFound = true;
      leptonFoundTT = true;
      if (HasPositiveCharge(&(*taus)[0])) {
        selTaup = 0;
        selTaum = 1;
      } else {
        selTaup = 1;
        selTaum = 0;
      }
    }
    for (std::vector<pat::Jet>::const_iterator jet = jets->begin(); jet < jets->end(); ++jet) {
      if (DeltaR<pat::Particle, pat::Jet>()((*taus)[0], *jet) < 0.1 ||
          DeltaR<pat::Particle, pat::Jet>()((*taus)[1], *jet) < 0.1) {
        JetVetoByTaus.push_back(jet - jets->begin());
      }
    }
    // select Jets (TopJet vector is sorted on ET)
    jetsFound = ((jets->size() - JetVetoByTaus.size()) >= 2);
  }

  // Check that the above work makes sense
  if (int(ee) + int(emu) + int(mumu) + int(etau) + int(mutau) + int(tautau) > 1)
    std::cout << "[TtDilepEvtSolutionMaker]: "
              << "Lepton selection criteria uncorrectly defined" << std::endl;

  bool correctLepton = (leptonFoundEE && eeChannel_) || ((leptonFoundEmMp || leptonFoundEpMm) && emuChannel_) ||
                       (leptonFoundMM && mumuChannel_) || ((leptonFoundMmTp || leptonFoundMpTm) && mutauChannel_) ||
                       ((leptonFoundEmTp || leptonFoundEpTm) && etauChannel_) || (leptonFoundTT && tautauChannel_);

  std::vector<TtDilepEvtSolution>* evtsols = new std::vector<TtDilepEvtSolution>();
  if (correctLepton && METFound && jetsFound) {
    // protect against reading beyond array boundaries while discounting vetoed jets
    unsigned int nrCombJets = 0;
    unsigned int numberOfJets = 0;
    for (; nrCombJets < jets->size() && numberOfJets < nrCombJets_; ++nrCombJets) {
      if (find(JetVetoByTaus.begin(), JetVetoByTaus.end(), int(nrCombJets)) == JetVetoByTaus.end())
        ++numberOfJets;
    }
    // consider all permutations
    for (unsigned int ib = 0; ib < nrCombJets; ib++) {
      // skipped jet vetoed during components-flagging.
      if (find(JetVetoByTaus.begin(), JetVetoByTaus.end(), int(ib)) != JetVetoByTaus.end())
        continue;
      // second loop of the permutations
      for (unsigned int ibbar = 0; ibbar < nrCombJets; ibbar++) {
        // avoid the diagonal: b and bbar must be distinct jets
        if (ib == ibbar)
          continue;
        // skipped jet vetoed during components-flagging.
        if (find(JetVetoByTaus.begin(), JetVetoByTaus.end(), int(ibbar)) != JetVetoByTaus.end())
          continue;
        // Build and save a solution
        TtDilepEvtSolution asol;
        asol.setJetCorrectionScheme(jetCorrScheme_);
        double xconstraint = 0, yconstraint = 0;
        // Set e+ in the event
        if (leptonFoundEE || leptonFoundEpMm || leptonFoundEpTm) {
          asol.setElectronp(electrons, selElectronp);
          xconstraint += (*electrons)[selElectronp].px();
          yconstraint += (*electrons)[selElectronp].py();
        }
        // Set e- in the event
        if (leptonFoundEE || leptonFoundEmMp || leptonFoundEmTp) {
          asol.setElectronm(electrons, selElectronm);
          xconstraint += (*electrons)[selElectronm].px();
          yconstraint += (*electrons)[selElectronm].py();
        }
        // Set mu+ in the event
        if (leptonFoundMM || leptonFoundEmMp || leptonFoundMpTm) {
          asol.setMuonp(muons, selMuonp);
          xconstraint += (*muons)[selMuonp].px();
          yconstraint += (*muons)[selMuonp].py();
        }
        // Set mu- in the event
        if (leptonFoundMM || leptonFoundEpMm || leptonFoundMmTp) {
          asol.setMuonm(muons, selMuonm);
          xconstraint += (*muons)[selMuonm].px();
          yconstraint += (*muons)[selMuonm].py();
        }
        // Set tau- in the event
        if (leptonFoundEpTm || leptonFoundMpTm || leptonFoundTT) {
          asol.setTaum(taus, selTaum);
          xconstraint += (*taus)[selTaum].px();
          yconstraint += (*taus)[selTaum].py();
        }
        // Set tau+ in the event
        if (leptonFoundEmTp || leptonFoundMmTp || leptonFoundTT) {
          asol.setTaup(taus, selTaup);
          xconstraint += (*taus)[selTaup].px();
          yconstraint += (*taus)[selTaup].py();
        }
        // Set Jets/MET in the event
        asol.setB(jets, ib);
        asol.setBbar(jets, ibbar);
        asol.setMET(mets, 0);
        xconstraint += (*jets)[ib].px() + (*jets)[ibbar].px() + (*mets)[0].px();
        yconstraint += (*jets)[ib].py() + (*jets)[ibbar].py() + (*mets)[0].py();
        // if asked for, match the event solutions to the gen Event
        if (matchToGenEvt_) {
          edm::Handle<TtGenEvent> genEvt;
          iEvent.getByToken(evtSourceToken_, genEvt);
          asol.setGenEvt(genEvt);
        }
        // If asked, use the kin fitter to compute the top mass
        if (calcTopMass_) {
          solver->SetConstraints(xconstraint, yconstraint);
          solver->useWeightFromMC(useMCforBest_);
          asol = solver->addKinSolInfo(&asol);
        }

        // these lines calculate the observables to be used in the TtDilepSignalSelection LR
        (*myLRSignalSelObservables)(asol, iEvent);

        evtsols->push_back(asol);
      }
    }
    // flag the best solution (MC matching)
    if (matchToGenEvt_) {
      double bestSolDR = 9999.;
      int bestSol = -1;
      double dR = 0.;
      for (size_t s = 0; s < evtsols->size(); s++) {
        dR = (*evtsols)[s].getJetResidual();
        if (dR < bestSolDR) {
          bestSolDR = dR;
          bestSol = s;
        }
      }
      if (bestSol != -1)
        (*evtsols)[bestSol].setBestSol(true);
    }
    // put the result in the event
    std::unique_ptr<std::vector<TtDilepEvtSolution> > pOut(evtsols);
    iEvent.put(std::move(pOut));
  } else {
    // no solution: put a dummy solution in the event
    TtDilepEvtSolution asol;
    evtsols->push_back(asol);
    std::unique_ptr<std::vector<TtDilepEvtSolution> > pOut(evtsols);
    iEvent.put(std::move(pOut));
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtDilepEvtSolutionMaker);
