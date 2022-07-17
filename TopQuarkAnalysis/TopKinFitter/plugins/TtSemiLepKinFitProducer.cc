#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/JetMCUtils/interface/combination.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"

template <typename LeptonCollection>
class TtSemiLepKinFitProducer : public edm::stream::EDProducer<> {
public:
  explicit TtSemiLepKinFitProducer(const edm::ParameterSet&);

private:
  // produce
  void produce(edm::Event&, const edm::EventSetup&) override;

  // convert unsigned to Param
  TtSemiLepKinFitter::Param param(unsigned);
  // convert unsigned to Param
  TtSemiLepKinFitter::Constraint constraint(unsigned);
  // convert unsigned to Param
  std::vector<TtSemiLepKinFitter::Constraint> constraints(std::vector<unsigned>&);
  // helper function for b-tagging
  bool doBTagging(bool& useBTag_,
                  const edm::Handle<std::vector<pat::Jet>>& jets,
                  const std::vector<int>& combi,
                  std::string& bTagAlgo_,
                  double& minBTagValueBJets_,
                  double& maxBTagValueNonBJets_);

  edm::EDGetTokenT<std::vector<pat::Jet>> jetsToken_;
  edm::EDGetTokenT<LeptonCollection> lepsToken_;
  edm::EDGetTokenT<std::vector<pat::MET>> metsToken_;

  edm::EDGetTokenT<std::vector<std::vector<int>>> matchToken_;
  /// switch to use only a combination given by another hypothesis
  bool useOnlyMatch_;
  /// input tag for b-tagging algorithm
  std::string bTagAlgo_;
  /// min value of bTag for a b-jet
  double minBTagValueBJet_;
  /// max value of bTag for a non-b-jet
  double maxBTagValueNonBJet_;
  /// switch to tell whether to use b-tagging or not
  bool useBTag_;
  /// maximal number of jets (-1 possible to indicate 'all')
  int maxNJets_;
  /// maximal number of combinations to be written to the event
  int maxNComb_;

  /// maximal number of iterations to be performed for the fit
  unsigned int maxNrIter_;
  /// maximal chi2 equivalent
  double maxDeltaS_;
  /// maximal deviation for contstraints
  double maxF_;
  unsigned int jetParam_;
  unsigned int lepParam_;
  unsigned int metParam_;
  /// constrains
  std::vector<unsigned> constraints_;
  double mW_;
  double mTop_;
  /// scale factors for jet energy resolution
  std::vector<double> jetEnergyResolutionScaleFactors_;
  std::vector<double> jetEnergyResolutionEtaBinning_;
  /// config-file-based object resolutions
  std::vector<edm::ParameterSet> udscResolutions_;
  std::vector<edm::ParameterSet> bResolutions_;
  std::vector<edm::ParameterSet> lepResolutions_;
  std::vector<edm::ParameterSet> metResolutions_;

  std::unique_ptr<TtSemiLepKinFitter> fitter;

  struct KinFitResult {
    int Status;
    double Chi2;
    double Prob;
    pat::Particle HadB;
    pat::Particle HadP;
    pat::Particle HadQ;
    pat::Particle LepB;
    pat::Particle LepL;
    pat::Particle LepN;
    std::vector<int> JetCombi;
    bool operator<(const KinFitResult& rhs) { return Chi2 < rhs.Chi2; };
  };
};

template <typename LeptonCollection>
TtSemiLepKinFitProducer<LeptonCollection>::TtSemiLepKinFitProducer(const edm::ParameterSet& cfg)
    : jetsToken_(consumes<std::vector<pat::Jet>>(cfg.getParameter<edm::InputTag>("jets"))),
      lepsToken_(consumes<LeptonCollection>(cfg.getParameter<edm::InputTag>("leps"))),
      metsToken_(consumes<std::vector<pat::MET>>(cfg.getParameter<edm::InputTag>("mets"))),
      matchToken_(mayConsume<std::vector<std::vector<int>>>(cfg.getParameter<edm::InputTag>("match"))),
      useOnlyMatch_(cfg.getParameter<bool>("useOnlyMatch")),
      bTagAlgo_(cfg.getParameter<std::string>("bTagAlgo")),
      minBTagValueBJet_(cfg.getParameter<double>("minBDiscBJets")),
      maxBTagValueNonBJet_(cfg.getParameter<double>("maxBDiscLightJets")),
      useBTag_(cfg.getParameter<bool>("useBTagging")),
      maxNJets_(cfg.getParameter<int>("maxNJets")),
      maxNComb_(cfg.getParameter<int>("maxNComb")),
      maxNrIter_(cfg.getParameter<unsigned>("maxNrIter")),
      maxDeltaS_(cfg.getParameter<double>("maxDeltaS")),
      maxF_(cfg.getParameter<double>("maxF")),
      jetParam_(cfg.getParameter<unsigned>("jetParametrisation")),
      lepParam_(cfg.getParameter<unsigned>("lepParametrisation")),
      metParam_(cfg.getParameter<unsigned>("metParametrisation")),
      constraints_(cfg.getParameter<std::vector<unsigned>>("constraints")),
      mW_(cfg.getParameter<double>("mW")),
      mTop_(cfg.getParameter<double>("mTop")),
      jetEnergyResolutionScaleFactors_(cfg.getParameter<std::vector<double>>("jetEnergyResolutionScaleFactors")),
      jetEnergyResolutionEtaBinning_(cfg.getParameter<std::vector<double>>("jetEnergyResolutionEtaBinning")),
      udscResolutions_(0),
      bResolutions_(0),
      lepResolutions_(0),
      metResolutions_(0) {
  if (cfg.exists("udscResolutions") && cfg.exists("bResolutions") && cfg.exists("lepResolutions") &&
      cfg.exists("metResolutions")) {
    udscResolutions_ = cfg.getParameter<std::vector<edm::ParameterSet>>("udscResolutions");
    bResolutions_ = cfg.getParameter<std::vector<edm::ParameterSet>>("bResolutions");
    lepResolutions_ = cfg.getParameter<std::vector<edm::ParameterSet>>("lepResolutions");
    metResolutions_ = cfg.getParameter<std::vector<edm::ParameterSet>>("metResolutions");
  } else if (cfg.exists("udscResolutions") || cfg.exists("bResolutions") || cfg.exists("lepResolutions") ||
             cfg.exists("metResolutions")) {
    throw cms::Exception("Configuration") << "Parameters 'udscResolutions', 'bResolutions', 'lepResolutions', "
                                             "'metResolutions' should be used together.\n";
  }

  fitter = std::make_unique<TtSemiLepKinFitter>(param(jetParam_),
                                                param(lepParam_),
                                                param(metParam_),
                                                maxNrIter_,
                                                maxDeltaS_,
                                                maxF_,
                                                constraints(constraints_),
                                                mW_,
                                                mTop_,
                                                &udscResolutions_,
                                                &bResolutions_,
                                                &lepResolutions_,
                                                &metResolutions_,
                                                &jetEnergyResolutionScaleFactors_,
                                                &jetEnergyResolutionEtaBinning_);

  produces<std::vector<pat::Particle>>("PartonsHadP");
  produces<std::vector<pat::Particle>>("PartonsHadQ");
  produces<std::vector<pat::Particle>>("PartonsHadB");
  produces<std::vector<pat::Particle>>("PartonsLepB");
  produces<std::vector<pat::Particle>>("Leptons");
  produces<std::vector<pat::Particle>>("Neutrinos");

  produces<std::vector<std::vector<int>>>();
  produces<std::vector<double>>("Chi2");
  produces<std::vector<double>>("Prob");
  produces<std::vector<int>>("Status");

  produces<int>("NumberOfConsideredJets");
}

template <typename LeptonCollection>
bool TtSemiLepKinFitProducer<LeptonCollection>::doBTagging(bool& useBTag_,
                                                           const edm::Handle<std::vector<pat::Jet>>& jets,
                                                           const std::vector<int>& combi,
                                                           std::string& bTagAlgo_,
                                                           double& minBTagValueBJet_,
                                                           double& maxBTagValueNonBJet_) {
  if (!useBTag_) {
    return true;
  }
  if (useBTag_ && (*jets)[combi[TtSemiLepEvtPartons::HadB]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ &&
      (*jets)[combi[TtSemiLepEvtPartons::LepB]].bDiscriminator(bTagAlgo_) >= minBTagValueBJet_ &&
      (*jets)[combi[TtSemiLepEvtPartons::LightQ]].bDiscriminator(bTagAlgo_) < maxBTagValueNonBJet_ &&
      (*jets)[combi[TtSemiLepEvtPartons::LightQBar]].bDiscriminator(bTagAlgo_) < maxBTagValueNonBJet_) {
    return true;
  } else {
    return false;
  }
}

template <typename LeptonCollection>
void TtSemiLepKinFitProducer<LeptonCollection>::produce(edm::Event& evt, const edm::EventSetup& setup) {
  std::unique_ptr<std::vector<pat::Particle>> pPartonsHadP(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle>> pPartonsHadQ(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle>> pPartonsHadB(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle>> pPartonsLepB(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle>> pLeptons(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle>> pNeutrinos(new std::vector<pat::Particle>);

  std::unique_ptr<std::vector<std::vector<int>>> pCombi(new std::vector<std::vector<int>>);
  std::unique_ptr<std::vector<double>> pChi2(new std::vector<double>);
  std::unique_ptr<std::vector<double>> pProb(new std::vector<double>);
  std::unique_ptr<std::vector<int>> pStatus(new std::vector<int>);

  std::unique_ptr<int> pJetsConsidered(new int);

  const edm::Handle<std::vector<pat::Jet>>& jets = evt.getHandle(jetsToken_);

  const edm::Handle<std::vector<pat::MET>>& mets = evt.getHandle(metsToken_);

  const edm::Handle<LeptonCollection>& leps = evt.getHandle(lepsToken_);

  const unsigned int nPartons = 4;

  std::vector<int> match;
  bool invalidMatch = false;
  if (useOnlyMatch_) {
    *pJetsConsidered = nPartons;
    const edm::Handle<std::vector<std::vector<int>>>& matchHandle = evt.getHandle(matchToken_);
    match = *(matchHandle->begin());
    // check if match is valid
    if (match.size() != nPartons)
      invalidMatch = true;
    else {
      for (unsigned int idx = 0; idx < match.size(); ++idx) {
        if (match[idx] < 0 || match[idx] >= (int)jets->size()) {
          invalidMatch = true;
          break;
        }
      }
    }
  }

  // -----------------------------------------------------
  // skip events with no appropriate lepton candidate in
  // or empty MET or less jets than partons or invalid match
  // -----------------------------------------------------

  if (leps->empty() || mets->empty() || jets->size() < nPartons || invalidMatch) {
    // the kinFit getters return empty objects here
    pPartonsHadP->push_back(fitter->fittedHadP());
    pPartonsHadQ->push_back(fitter->fittedHadQ());
    pPartonsHadB->push_back(fitter->fittedHadB());
    pPartonsLepB->push_back(fitter->fittedLepB());
    pLeptons->push_back(fitter->fittedLepton());
    pNeutrinos->push_back(fitter->fittedNeutrino());
    // indices referring to the jet combination
    std::vector<int> invalidCombi;
    for (unsigned int i = 0; i < nPartons; ++i)
      invalidCombi.push_back(-1);
    pCombi->push_back(invalidCombi);
    // chi2
    pChi2->push_back(-1.);
    // chi2 probability
    pProb->push_back(-1.);
    // status of the fitter
    pStatus->push_back(-1);
    // number of jets
    *pJetsConsidered = jets->size();
    // feed out all products
    evt.put(std::move(pCombi));
    evt.put(std::move(pPartonsHadP), "PartonsHadP");
    evt.put(std::move(pPartonsHadQ), "PartonsHadQ");
    evt.put(std::move(pPartonsHadB), "PartonsHadB");
    evt.put(std::move(pPartonsLepB), "PartonsLepB");
    evt.put(std::move(pLeptons), "Leptons");
    evt.put(std::move(pNeutrinos), "Neutrinos");
    evt.put(std::move(pChi2), "Chi2");
    evt.put(std::move(pProb), "Prob");
    evt.put(std::move(pStatus), "Status");
    evt.put(std::move(pJetsConsidered), "NumberOfConsideredJets");
    return;
  }

  // -----------------------------------------------------
  // analyze different jet combinations using the KinFitter
  // (or only a given jet combination if useOnlyMatch=true)
  // -----------------------------------------------------

  std::vector<int> jetIndices;
  if (!useOnlyMatch_) {
    for (unsigned int i = 0; i < jets->size(); ++i) {
      if (maxNJets_ >= (int)nPartons && maxNJets_ == (int)i) {
        *pJetsConsidered = i;
        break;
      }
      jetIndices.push_back(i);
    }
  }

  std::vector<int> combi;
  for (unsigned int i = 0; i < nPartons; ++i) {
    if (useOnlyMatch_)
      combi.push_back(match[i]);
    else
      combi.push_back(i);
  }

  std::list<KinFitResult> FitResultList;

  do {
    for (int cnt = 0; cnt < TMath::Factorial(combi.size()); ++cnt) {
      // take into account indistinguishability of the two jets from the hadr. W decay,
      // reduces combinatorics by a factor of 2
      if ((combi[TtSemiLepEvtPartons::LightQ] < combi[TtSemiLepEvtPartons::LightQBar] || useOnlyMatch_) &&
          doBTagging(useBTag_, jets, combi, bTagAlgo_, minBTagValueBJet_, maxBTagValueNonBJet_)) {
        std::vector<pat::Jet> jetCombi;
        jetCombi.resize(nPartons);
        jetCombi[TtSemiLepEvtPartons::LightQ] = (*jets)[combi[TtSemiLepEvtPartons::LightQ]];
        jetCombi[TtSemiLepEvtPartons::LightQBar] = (*jets)[combi[TtSemiLepEvtPartons::LightQBar]];
        jetCombi[TtSemiLepEvtPartons::HadB] = (*jets)[combi[TtSemiLepEvtPartons::HadB]];
        jetCombi[TtSemiLepEvtPartons::LepB] = (*jets)[combi[TtSemiLepEvtPartons::LepB]];

        // do the kinematic fit
        const int status = fitter->fit(jetCombi, (*leps)[0], (*mets)[0]);

        if (status == 0) {  // only take into account converged fits
          KinFitResult result;
          result.Status = status;
          result.Chi2 = fitter->fitS();
          result.Prob = fitter->fitProb();
          result.HadB = fitter->fittedHadB();
          result.HadP = fitter->fittedHadP();
          result.HadQ = fitter->fittedHadQ();
          result.LepB = fitter->fittedLepB();
          result.LepL = fitter->fittedLepton();
          result.LepN = fitter->fittedNeutrino();
          result.JetCombi = combi;

          FitResultList.push_back(result);
        }
      }
      if (useOnlyMatch_)
        break;  // don't go through combinatorics if useOnlyMatch was chosen
      next_permutation(combi.begin(), combi.end());
    }
    if (useOnlyMatch_)
      break;  // don't go through combinatorics if useOnlyMatch was chosen
  } while (stdcomb::next_combination(jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end()));

  // sort results w.r.t. chi2 values
  FitResultList.sort();

  // -----------------------------------------------------
  // feed out result
  // starting with the JetComb having the smallest chi2
  // -----------------------------------------------------

  if ((unsigned)FitResultList.size() < 1) {  // in case no fit results were stored in the list (all fits aborted)
    pPartonsHadP->push_back(fitter->fittedHadP());
    pPartonsHadQ->push_back(fitter->fittedHadQ());
    pPartonsHadB->push_back(fitter->fittedHadB());
    pPartonsLepB->push_back(fitter->fittedLepB());
    pLeptons->push_back(fitter->fittedLepton());
    pNeutrinos->push_back(fitter->fittedNeutrino());
    // indices referring to the jet combination
    std::vector<int> invalidCombi;
    for (unsigned int i = 0; i < nPartons; ++i)
      invalidCombi.push_back(-1);
    pCombi->push_back(invalidCombi);
    // chi2
    pChi2->push_back(-1.);
    // chi2 probability
    pProb->push_back(-1.);
    // status of the fitter
    pStatus->push_back(-1);
  } else {
    unsigned int iComb = 0;
    for (typename std::list<KinFitResult>::const_iterator result = FitResultList.begin(); result != FitResultList.end();
         ++result) {
      if (maxNComb_ >= 1 && iComb == (unsigned int)maxNComb_)
        break;
      iComb++;
      // partons
      pPartonsHadP->push_back(result->HadP);
      pPartonsHadQ->push_back(result->HadQ);
      pPartonsHadB->push_back(result->HadB);
      pPartonsLepB->push_back(result->LepB);
      // lepton
      pLeptons->push_back(result->LepL);
      // neutrino
      pNeutrinos->push_back(result->LepN);
      // indices referring to the jet combination
      pCombi->push_back(result->JetCombi);
      // chi2
      pChi2->push_back(result->Chi2);
      // chi2 probability
      pProb->push_back(result->Prob);
      // status of the fitter
      pStatus->push_back(result->Status);
    }
  }
  evt.put(std::move(pCombi));
  evt.put(std::move(pPartonsHadP), "PartonsHadP");
  evt.put(std::move(pPartonsHadQ), "PartonsHadQ");
  evt.put(std::move(pPartonsHadB), "PartonsHadB");
  evt.put(std::move(pPartonsLepB), "PartonsLepB");
  evt.put(std::move(pLeptons), "Leptons");
  evt.put(std::move(pNeutrinos), "Neutrinos");
  evt.put(std::move(pChi2), "Chi2");
  evt.put(std::move(pProb), "Prob");
  evt.put(std::move(pStatus), "Status");
  evt.put(std::move(pJetsConsidered), "NumberOfConsideredJets");
}

template <typename LeptonCollection>
TtSemiLepKinFitter::Param TtSemiLepKinFitProducer<LeptonCollection>::param(unsigned val) {
  TtSemiLepKinFitter::Param result;
  switch (val) {
    case TtSemiLepKinFitter::kEMom:
      result = TtSemiLepKinFitter::kEMom;
      break;
    case TtSemiLepKinFitter::kEtEtaPhi:
      result = TtSemiLepKinFitter::kEtEtaPhi;
      break;
    case TtSemiLepKinFitter::kEtThetaPhi:
      result = TtSemiLepKinFitter::kEtThetaPhi;
      break;
    default:
      throw cms::Exception("Configuration") << "Chosen jet parametrization is not supported: " << val << "\n";
      break;
  }
  return result;
}

template <typename LeptonCollection>
TtSemiLepKinFitter::Constraint TtSemiLepKinFitProducer<LeptonCollection>::constraint(unsigned val) {
  TtSemiLepKinFitter::Constraint result;
  switch (val) {
    case TtSemiLepKinFitter::kWHadMass:
      result = TtSemiLepKinFitter::kWHadMass;
      break;
    case TtSemiLepKinFitter::kWLepMass:
      result = TtSemiLepKinFitter::kWLepMass;
      break;
    case TtSemiLepKinFitter::kTopHadMass:
      result = TtSemiLepKinFitter::kTopHadMass;
      break;
    case TtSemiLepKinFitter::kTopLepMass:
      result = TtSemiLepKinFitter::kTopLepMass;
      break;
    case TtSemiLepKinFitter::kNeutrinoMass:
      result = TtSemiLepKinFitter::kNeutrinoMass;
      break;
    case TtSemiLepKinFitter::kEqualTopMasses:
      result = TtSemiLepKinFitter::kEqualTopMasses;
      break;
    case TtSemiLepKinFitter::kSumPt:
      result = TtSemiLepKinFitter::kSumPt;
      break;
    default:
      throw cms::Exception("Configuration") << "Chosen fit constraint is not supported: " << val << "\n";
      break;
  }
  return result;
}

template <typename LeptonCollection>
std::vector<TtSemiLepKinFitter::Constraint> TtSemiLepKinFitProducer<LeptonCollection>::constraints(
    std::vector<unsigned>& val) {
  std::vector<TtSemiLepKinFitter::Constraint> result;
  for (unsigned i = 0; i < val.size(); ++i) {
    result.push_back(constraint(val[i]));
  }
  return result;
}

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
using TtSemiLepKinFitProducerMuon = TtSemiLepKinFitProducer<std::vector<pat::Muon>>;
using TtSemiLepKinFitProducerElectron = TtSemiLepKinFitProducer<std::vector<pat::Electron>>;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtSemiLepKinFitProducerMuon);
DEFINE_FWK_MODULE(TtSemiLepKinFitProducerElectron);
