#ifndef TtFullLepKinSolutionProducer_h
#define TtFullLepKinSolutionProducer_h

//
//
#include <memory>
#include <string>
#include <vector>
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtFullLepKinSolver.h"

class TtFullLepKinSolutionProducer : public edm::EDProducer {
public:
  explicit TtFullLepKinSolutionProducer(const edm::ParameterSet& iConfig);
  ~TtFullLepKinSolutionProducer() override;

  void beginJob() override;
  void produce(edm::Event& evt, const edm::EventSetup& iSetup) override;
  void endJob() override;

private:
  // next methods are avoidable but they make the code legible
  inline bool PTComp(const reco::Candidate*, const reco::Candidate*) const;
  inline bool LepDiffCharge(const reco::Candidate*, const reco::Candidate*) const;
  inline bool HasPositiveCharge(const reco::Candidate*) const;

  struct Compare {
    bool operator()(std::pair<double, int> a, std::pair<double, int> b) { return a.first > b.first; }
  };

  edm::EDGetTokenT<std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT<std::vector<pat::Electron> > electronsToken_;
  edm::EDGetTokenT<std::vector<pat::Muon> > muonsToken_;
  edm::EDGetTokenT<std::vector<pat::MET> > metsToken_;

  std::string jetCorrLevel_;
  int maxNJets_, maxNComb_;
  bool eeChannel_, emuChannel_, mumuChannel_, searchWrongCharge_;
  double tmassbegin_, tmassend_, tmassstep_;
  std::vector<double> nupars_;

  TtFullLepKinSolver* solver;
};

inline bool TtFullLepKinSolutionProducer::PTComp(const reco::Candidate* l1, const reco::Candidate* l2) const {
  return (l1->pt() > l2->pt());
}

inline bool TtFullLepKinSolutionProducer::LepDiffCharge(const reco::Candidate* l1, const reco::Candidate* l2) const {
  return (l1->charge() != l2->charge());
}

inline bool TtFullLepKinSolutionProducer::HasPositiveCharge(const reco::Candidate* l) const {
  return (l->charge() > 0);
}

inline TtFullLepKinSolutionProducer::TtFullLepKinSolutionProducer(const edm::ParameterSet& iConfig) {
  // configurables
  jetsToken_ = consumes<std::vector<pat::Jet> >(iConfig.getParameter<edm::InputTag>("jets"));
  electronsToken_ = consumes<std::vector<pat::Electron> >(iConfig.getParameter<edm::InputTag>("electrons"));
  muonsToken_ = consumes<std::vector<pat::Muon> >(iConfig.getParameter<edm::InputTag>("muons"));
  metsToken_ = consumes<std::vector<pat::MET> >(iConfig.getParameter<edm::InputTag>("mets"));
  jetCorrLevel_ = iConfig.getParameter<std::string>("jetCorrectionLevel");
  maxNJets_ = iConfig.getParameter<int>("maxNJets");
  maxNComb_ = iConfig.getParameter<int>("maxNComb");
  eeChannel_ = iConfig.getParameter<bool>("eeChannel");
  emuChannel_ = iConfig.getParameter<bool>("emuChannel");
  mumuChannel_ = iConfig.getParameter<bool>("mumuChannel");
  searchWrongCharge_ = iConfig.getParameter<bool>("searchWrongCharge");
  tmassbegin_ = iConfig.getParameter<double>("tmassbegin");
  tmassend_ = iConfig.getParameter<double>("tmassend");
  tmassstep_ = iConfig.getParameter<double>("tmassstep");
  nupars_ = iConfig.getParameter<std::vector<double> >("neutrino_parameters");

  // define what will be produced
  produces<std::vector<std::vector<int> > >();  // vector of the particle inices (b, bbar, e1, e2, mu1, mu2)
  produces<std::vector<reco::LeafCandidate> >("fullLepNeutrinos");
  produces<std::vector<reco::LeafCandidate> >("fullLepNeutrinoBars");
  produces<std::vector<double> >("solWeight");  //weight for a specific kin solution
  produces<bool>("isWrongCharge");              //true if leptons have the same charge
}

inline TtFullLepKinSolutionProducer::~TtFullLepKinSolutionProducer() {}

inline void TtFullLepKinSolutionProducer::beginJob() {
  solver = new TtFullLepKinSolver(tmassbegin_, tmassend_, tmassstep_, nupars_);
}

inline void TtFullLepKinSolutionProducer::endJob() { delete solver; }

inline void TtFullLepKinSolutionProducer::produce(edm::Event& evt, const edm::EventSetup& iSetup) {
  //create vectors fo runsorted output
  std::vector<std::vector<int> > idcsV;
  std::vector<reco::LeafCandidate> nusV;
  std::vector<reco::LeafCandidate> nuBarsV;
  std::vector<std::pair<double, int> > weightsV;

  //create pointer for products
  std::unique_ptr<std::vector<std::vector<int> > > pIdcs(new std::vector<std::vector<int> >);
  std::unique_ptr<std::vector<reco::LeafCandidate> > pNus(new std::vector<reco::LeafCandidate>);
  std::unique_ptr<std::vector<reco::LeafCandidate> > pNuBars(new std::vector<reco::LeafCandidate>);
  std::unique_ptr<std::vector<double> > pWeight(new std::vector<double>);
  std::unique_ptr<bool> pWrongCharge(new bool);

  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByToken(jetsToken_, jets);
  edm::Handle<std::vector<pat::Electron> > electrons;
  evt.getByToken(electronsToken_, electrons);
  edm::Handle<std::vector<pat::Muon> > muons;
  evt.getByToken(muonsToken_, muons);
  edm::Handle<std::vector<pat::MET> > mets;
  evt.getByToken(metsToken_, mets);

  int selMuon1 = -1, selMuon2 = -1;
  int selElectron1 = -1, selElectron2 = -1;
  bool ee = false;
  bool emu = false;
  bool mumu = false;
  bool isWrongCharge = false;
  bool jetsFound = false;
  bool METFound = false;
  bool electronsFound = false;
  bool electronMuonFound = false;
  bool muonsFound = false;

  //select Jets (TopJet vector is sorted on ET)
  if (jets->size() >= 2) {
    jetsFound = true;
  }

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
    if (ee && eeChannel_) {
      if (LepDiffCharge(&(*electrons)[0], &(*electrons)[1]) || searchWrongCharge_) {
        if (HasPositiveCharge(&(*electrons)[0]) || !LepDiffCharge(&(*electrons)[0], &(*electrons)[1])) {
          selElectron1 = 0;
          selElectron2 = 1;
        } else {
          selElectron1 = 1;
          selElectron2 = 0;
        }
        electronsFound = true;
        if (!LepDiffCharge(&(*electrons)[0], &(*electrons)[1]))
          isWrongCharge = true;
      }
    } else if (emu && emuChannel_) {
      if (LepDiffCharge(&(*electrons)[0], &(*muons)[0]) || searchWrongCharge_) {
        selElectron1 = 0;
        selMuon1 = 0;
        electronMuonFound = true;
        if (!LepDiffCharge(&(*electrons)[0], &(*muons)[0]))
          isWrongCharge = true;
      }
    } else if (mumu && mumuChannel_) {
      if (LepDiffCharge(&(*muons)[0], &(*muons)[1]) || searchWrongCharge_) {
        if (HasPositiveCharge(&(*muons)[0]) || !LepDiffCharge(&(*muons)[0], &(*muons)[1])) {
          selMuon1 = 0;
          selMuon2 = 1;
        } else {
          selMuon1 = 1;
          selMuon2 = 0;
        }
        muonsFound = true;
        if (!LepDiffCharge(&(*muons)[0], &(*muons)[1]))
          isWrongCharge = true;
      }
    }
  }

  *pWrongCharge = isWrongCharge;

  // Check that the above work makes sense
  if (int(ee) + int(emu) + int(mumu) > 1) {
    edm::LogWarning("TtFullLepKinSolutionProducer") << "Lepton selection criteria uncorrectly defined";
  }

  // Check if the leptons for the required Channel are available
  bool correctLeptons =
      ((electronsFound && eeChannel_) || (muonsFound && mumuChannel_) || (electronMuonFound && emuChannel_));
  // Check for equally charged leptons if for wrong charge combinations is searched
  if (isWrongCharge) {
    correctLeptons = correctLeptons && searchWrongCharge_;
  }

  if (correctLeptons && METFound && jetsFound) {
    // run over all jets if input parameter maxNJets is -1 or
    // adapt maxNJets if number of present jets is smaller than selected
    // number of jets
    int stop = maxNJets_;
    if (jets->size() < static_cast<unsigned int>(stop) || stop < 0)
      stop = jets->size();

    // counter for number of found kinematic solutions
    int nSol = 0;

    // consider all permutations
    for (int ib = 0; ib < stop; ib++) {
      // second loop of the permutations
      for (int ibbar = 0; ibbar < stop; ibbar++) {
        // avoid the diagonal: b and bbar must be distinct jets
        if (ib == ibbar)
          continue;

        std::vector<int> idcs;

        // push back the indices of the jets
        idcs.push_back(ib);
        idcs.push_back(ibbar);

        TLorentzVector LV_l1;
        TLorentzVector LV_l2;
        TLorentzVector LV_b = TLorentzVector((*jets)[ib].correctedJet(jetCorrLevel_, "bottom").px(),
                                             (*jets)[ib].correctedJet(jetCorrLevel_, "bottom").py(),
                                             (*jets)[ib].correctedJet(jetCorrLevel_, "bottom").pz(),
                                             (*jets)[ib].correctedJet(jetCorrLevel_, "bottom").energy());
        TLorentzVector LV_bbar = TLorentzVector((*jets)[ibbar].correctedJet(jetCorrLevel_, "bottom").px(),
                                                (*jets)[ibbar].correctedJet(jetCorrLevel_, "bottom").py(),
                                                (*jets)[ibbar].correctedJet(jetCorrLevel_, "bottom").pz(),
                                                (*jets)[ibbar].correctedJet(jetCorrLevel_, "bottom").energy());

        double xconstraint = 0, yconstraint = 0;

        if (ee) {
          idcs.push_back(selElectron1);
          LV_l1.SetXYZT((*electrons)[selElectron1].px(),
                        (*electrons)[selElectron1].py(),
                        (*electrons)[selElectron1].pz(),
                        (*electrons)[selElectron1].energy());
          xconstraint += (*electrons)[selElectron1].px();
          yconstraint += (*electrons)[selElectron1].py();

          idcs.push_back(selElectron2);
          LV_l2.SetXYZT((*electrons)[selElectron2].px(),
                        (*electrons)[selElectron2].py(),
                        (*electrons)[selElectron2].pz(),
                        (*electrons)[selElectron2].energy());
          xconstraint += (*electrons)[selElectron2].px();
          yconstraint += (*electrons)[selElectron2].py();

          idcs.push_back(-1);
          idcs.push_back(-1);
        }

        else if (emu) {
          if (!isWrongCharge) {
            if (HasPositiveCharge(&(*electrons)[selElectron1])) {
              idcs.push_back(selElectron1);
              LV_l1.SetXYZT((*electrons)[selElectron1].px(),
                            (*electrons)[selElectron1].py(),
                            (*electrons)[selElectron1].pz(),
                            (*electrons)[selElectron1].energy());
              xconstraint += (*electrons)[selElectron1].px();
              yconstraint += (*electrons)[selElectron1].py();

              idcs.push_back(-1);
              idcs.push_back(-1);

              idcs.push_back(selMuon1);
              LV_l2.SetXYZT((*muons)[selMuon1].px(),
                            (*muons)[selMuon1].py(),
                            (*muons)[selMuon1].pz(),
                            (*muons)[selMuon1].energy());
              xconstraint += (*muons)[selMuon1].px();
              yconstraint += (*muons)[selMuon1].py();
            } else {
              idcs.push_back(-1);

              idcs.push_back(selMuon1);
              LV_l1.SetXYZT((*muons)[selMuon1].px(),
                            (*muons)[selMuon1].py(),
                            (*muons)[selMuon1].pz(),
                            (*muons)[selMuon1].energy());
              xconstraint += (*muons)[selMuon1].px();
              yconstraint += (*muons)[selMuon1].py();

              idcs.push_back(selElectron1);
              LV_l2.SetXYZT((*electrons)[selElectron1].px(),
                            (*electrons)[selElectron1].py(),
                            (*electrons)[selElectron1].pz(),
                            (*electrons)[selElectron1].energy());
              xconstraint += (*electrons)[selElectron1].px();
              yconstraint += (*electrons)[selElectron1].py();

              idcs.push_back(-1);
            }
          } else {                                                 // means "if wrong charge"
            if (HasPositiveCharge(&(*electrons)[selElectron1])) {  // both leps positive
              idcs.push_back(selElectron1);
              LV_l1.SetXYZT((*electrons)[selElectron1].px(),
                            (*electrons)[selElectron1].py(),
                            (*electrons)[selElectron1].pz(),
                            (*electrons)[selElectron1].energy());
              xconstraint += (*electrons)[selElectron1].px();
              yconstraint += (*electrons)[selElectron1].py();

              idcs.push_back(-1);

              idcs.push_back(selMuon1);
              LV_l2.SetXYZT((*muons)[selMuon1].px(),
                            (*muons)[selMuon1].py(),
                            (*muons)[selMuon1].pz(),
                            (*muons)[selMuon1].energy());
              xconstraint += (*muons)[selMuon1].px();
              yconstraint += (*muons)[selMuon1].py();

              idcs.push_back(-1);
            } else {  // both leps negative
              idcs.push_back(-1);

              idcs.push_back(selElectron1);
              LV_l2.SetXYZT((*electrons)[selElectron1].px(),
                            (*electrons)[selElectron1].py(),
                            (*electrons)[selElectron1].pz(),
                            (*electrons)[selElectron1].energy());
              xconstraint += (*electrons)[selElectron1].px();
              yconstraint += (*electrons)[selElectron1].py();

              idcs.push_back(-1);

              idcs.push_back(selMuon1);
              LV_l1.SetXYZT((*muons)[selMuon1].px(),
                            (*muons)[selMuon1].py(),
                            (*muons)[selMuon1].pz(),
                            (*muons)[selMuon1].energy());
              xconstraint += (*muons)[selMuon1].px();
              yconstraint += (*muons)[selMuon1].py();
            }
          }
        }

        else if (mumu) {
          idcs.push_back(-1);
          idcs.push_back(-1);

          idcs.push_back(selMuon1);
          LV_l1.SetXYZT(
              (*muons)[selMuon1].px(), (*muons)[selMuon1].py(), (*muons)[selMuon1].pz(), (*muons)[selMuon1].energy());
          xconstraint += (*muons)[selMuon1].px();
          yconstraint += (*muons)[selMuon1].py();

          idcs.push_back(selMuon2);
          LV_l2.SetXYZT(
              (*muons)[selMuon2].px(), (*muons)[selMuon2].py(), (*muons)[selMuon2].pz(), (*muons)[selMuon2].energy());
          xconstraint += (*muons)[selMuon2].px();
          yconstraint += (*muons)[selMuon2].py();
        }

        xconstraint += (*jets)[ib].px() + (*jets)[ibbar].px() + (*mets)[0].px();
        yconstraint += (*jets)[ib].py() + (*jets)[ibbar].py() + (*mets)[0].py();

        // calculate neutrino momenta and eventweight
        solver->SetConstraints(xconstraint, yconstraint);
        TtFullLepKinSolver::NeutrinoSolution nuSol = solver->getNuSolution(LV_l1, LV_l2, LV_b, LV_bbar);

        // add solution to the vectors of solutions if solution exists
        if (nuSol.weight > 0) {
          // add the leptons and jets indices to the vector of combinations
          idcsV.push_back(idcs);

          // add the neutrinos
          nusV.push_back(nuSol.neutrino);
          nuBarsV.push_back(nuSol.neutrinoBar);

          // add the solution weight
          weightsV.push_back(std::make_pair(nuSol.weight, nSol));

          nSol++;
        }
      }
    }
  }

  if (weightsV.empty()) {
    //create dmummy vector
    std::vector<int> idcs;
    idcs.reserve(6);
    for (int i = 0; i < 6; ++i)
      idcs.push_back(-1);

    idcsV.push_back(idcs);
    weightsV.push_back(std::make_pair(-1, 0));
    reco::LeafCandidate nu;
    nusV.push_back(nu);
    reco::LeafCandidate nuBar;
    nuBarsV.push_back(nuBar);
  }

  // check if all vectors have correct length
  int weightL = weightsV.size();
  int nuL = nusV.size();
  int nuBarL = nuBarsV.size();
  int idxL = idcsV.size();

  if (weightL != nuL || weightL != nuBarL || weightL != idxL) {
    edm::LogWarning("TtFullLepKinSolutionProducer")
        << "Output vectors are of different length:"
        << "\n weight: " << weightL << "\n     nu: " << nuL << "\n  nubar: " << nuBarL << "\n   idcs: " << idxL;
  }

  // sort vectors by weight in decreasing order
  if (weightsV.size() > 1) {
    sort(weightsV.begin(), weightsV.end(), Compare());
  }

  // determine the number of solutions which is written in the event
  int stop = weightL;
  if (maxNComb_ > 0 && maxNComb_ < stop)
    stop = maxNComb_;

  for (int i = 0; i < stop; ++i) {
    pWeight->push_back(weightsV[i].first);
    pNus->push_back(nusV[weightsV[i].second]);
    pNuBars->push_back(nuBarsV[weightsV[i].second]);
    pIdcs->push_back(idcsV[weightsV[i].second]);
  }

  // put the results in the event
  evt.put(std::move(pIdcs));
  evt.put(std::move(pNus), "fullLepNeutrinos");
  evt.put(std::move(pNuBars), "fullLepNeutrinoBars");
  evt.put(std::move(pWeight), "solWeight");
  evt.put(std::move(pWrongCharge), "isWrongCharge");
}

#endif
