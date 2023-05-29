#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/TtFullHadKinFitter.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

/*
  \class   TtFullHadKinFitProducer TtFullHadKinFitProducer.h "TopQuarkAnalysis/TopKinFitter/plugins/TtFullHadKinFitProducer.h"

  \brief   Retrieve kinFit result from TtFullHadKinFitter and put it into the event

  Get jet collection and if wanted match from the event content and do the kinematic fit
  of the event with this objects using the kinFit class from TtFullHadKinFitter and put
  the result into the event content

**/

class TtFullHadKinFitProducer : public edm::stream::EDProducer<> {
public:
  /// default constructor
  explicit TtFullHadKinFitProducer(const edm::ParameterSet& cfg);

private:
  /// produce fitted object collections and meta data describing fit quality
  void produce(edm::Event& event, const edm::EventSetup& setup) override;

private:
  /// input tag for jets
  edm::EDGetTokenT<std::vector<pat::Jet> > jetsToken_;
  /// input tag for matches (in case the fit should be performed on certain matches)
  edm::EDGetTokenT<std::vector<std::vector<int> > > matchToken_;
  /// switch to tell whether all possible combinations should be used for the fit
  /// or only a certain combination
  bool useOnlyMatch_;
  /// input tag for b-tagging algorithm
  std::string bTagAlgo_;
  /// min value of bTag for a b-jet
  double minBTagValueBJet_;
  /// max value of bTag for a non-b-jet
  double maxBTagValueNonBJet_;
  /// switch to tell whether to use b-tagging or not
  bool useBTagging_;
  /// minimal number of b-jets
  unsigned int bTags_;
  /// correction level for jets
  std::string jetCorrectionLevel_;
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
  /// numbering of different possible jet parametrizations
  unsigned int jetParam_;
  /// numbering of different possible kinematic constraints
  std::vector<unsigned> constraints_;
  /// W mass value used for constraints
  double mW_;
  /// top mass value used for constraints
  double mTop_;
  /// store the resolutions for the jets
  std::vector<edm::ParameterSet> udscResolutions_, bResolutions_;
  /// scale factors for jet energy resolution
  std::vector<double> jetEnergyResolutionScaleFactors_;
  std::vector<double> jetEnergyResolutionEtaBinning_;

public:
  /// kinematic fit interface
  std::unique_ptr<TtFullHadKinFitter::KinFit> kinFitter;
};

static const unsigned int nPartons = 6;

/// default constructor
TtFullHadKinFitProducer::TtFullHadKinFitProducer(const edm::ParameterSet& cfg)
    : jetsToken_(consumes<std::vector<pat::Jet> >(cfg.getParameter<edm::InputTag>("jets"))),
      matchToken_(mayConsume<std::vector<std::vector<int> > >(cfg.getParameter<edm::InputTag>("match"))),
      useOnlyMatch_(cfg.getParameter<bool>("useOnlyMatch")),
      bTagAlgo_(cfg.getParameter<std::string>("bTagAlgo")),
      minBTagValueBJet_(cfg.getParameter<double>("minBTagValueBJet")),
      maxBTagValueNonBJet_(cfg.getParameter<double>("maxBTagValueNonBJet")),
      useBTagging_(cfg.getParameter<bool>("useBTagging")),
      bTags_(cfg.getParameter<unsigned int>("bTags")),
      jetCorrectionLevel_(cfg.getParameter<std::string>("jetCorrectionLevel")),
      maxNJets_(cfg.getParameter<int>("maxNJets")),
      maxNComb_(cfg.getParameter<int>("maxNComb")),
      maxNrIter_(cfg.getParameter<unsigned int>("maxNrIter")),
      maxDeltaS_(cfg.getParameter<double>("maxDeltaS")),
      maxF_(cfg.getParameter<double>("maxF")),
      jetParam_(cfg.getParameter<unsigned>("jetParametrisation")),
      constraints_(cfg.getParameter<std::vector<unsigned> >("constraints")),
      mW_(cfg.getParameter<double>("mW")),
      mTop_(cfg.getParameter<double>("mTop")),
      jetEnergyResolutionScaleFactors_(cfg.getParameter<std::vector<double> >("jetEnergyResolutionScaleFactors")),
      jetEnergyResolutionEtaBinning_(cfg.getParameter<std::vector<double> >("jetEnergyResolutionEtaBinning")) {
  if (cfg.exists("udscResolutions") && cfg.exists("bResolutions")) {
    udscResolutions_ = cfg.getParameter<std::vector<edm::ParameterSet> >("udscResolutions");
    bResolutions_ = cfg.getParameter<std::vector<edm::ParameterSet> >("bResolutions");
  } else if (cfg.exists("udscResolutions") || cfg.exists("bResolutions")) {
    if (cfg.exists("udscResolutions"))
      throw cms::Exception("Configuration")
          << "Parameter 'bResolutions' is needed if parameter 'udscResolutions' is defined!\n";
    else
      throw cms::Exception("Configuration")
          << "Parameter 'udscResolutions' is needed if parameter 'bResolutions' is defined!\n";
  }

  // define kinematic fit interface
  kinFitter = std::make_unique<TtFullHadKinFitter::KinFit>(useBTagging_,
                                                           bTags_,
                                                           bTagAlgo_,
                                                           minBTagValueBJet_,
                                                           maxBTagValueNonBJet_,
                                                           udscResolutions_,
                                                           bResolutions_,
                                                           jetEnergyResolutionScaleFactors_,
                                                           jetEnergyResolutionEtaBinning_,
                                                           jetCorrectionLevel_,
                                                           maxNJets_,
                                                           maxNComb_,
                                                           maxNrIter_,
                                                           maxDeltaS_,
                                                           maxF_,
                                                           jetParam_,
                                                           constraints_,
                                                           mW_,
                                                           mTop_);

  // produces the following collections
  produces<std::vector<pat::Particle> >("PartonsB");
  produces<std::vector<pat::Particle> >("PartonsBBar");
  produces<std::vector<pat::Particle> >("PartonsLightQ");
  produces<std::vector<pat::Particle> >("PartonsLightQBar");
  produces<std::vector<pat::Particle> >("PartonsLightP");
  produces<std::vector<pat::Particle> >("PartonsLightPBar");

  produces<std::vector<std::vector<int> > >();
  produces<std::vector<double> >("Chi2");
  produces<std::vector<double> >("Prob");
  produces<std::vector<int> >("Status");
}

/// produce fitted object collections and meta data describing fit quality
void TtFullHadKinFitProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  // get jet collection
  const edm::Handle<std::vector<pat::Jet> >& jets = event.getHandle(jetsToken_);

  // get match in case that useOnlyMatch_ is true
  std::vector<int> match;
  bool invalidMatch = false;
  if (useOnlyMatch_) {
    kinFitter->setUseOnlyMatch(true);
    // in case that only a ceratin match should be used, get match here
    const edm::Handle<std::vector<std::vector<int> > >& matches = event.getHandle(matchToken_);
    match = *(matches->begin());
    // check if match is valid
    if (match.size() != nPartons) {
      invalidMatch = true;
    } else {
      for (unsigned int idx = 0; idx < match.size(); ++idx) {
        if (match[idx] < 0 || match[idx] >= (int)jets->size()) {
          invalidMatch = true;
          break;
        }
      }
    }
    /// set match to be used
    kinFitter->setMatch(match);
  }

  /// set the validity of a match
  kinFitter->setMatchInvalidity(invalidMatch);

  std::list<TtFullHadKinFitter::KinFitResult> fitResults = kinFitter->fit(*jets);

  // pointer for output collections
  std::unique_ptr<std::vector<pat::Particle> > pPartonsB(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle> > pPartonsBBar(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle> > pPartonsLightQ(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle> > pPartonsLightQBar(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle> > pPartonsLightP(new std::vector<pat::Particle>);
  std::unique_ptr<std::vector<pat::Particle> > pPartonsLightPBar(new std::vector<pat::Particle>);
  // pointer for meta information
  std::unique_ptr<std::vector<std::vector<int> > > pCombi(new std::vector<std::vector<int> >);
  std::unique_ptr<std::vector<double> > pChi2(new std::vector<double>);
  std::unique_ptr<std::vector<double> > pProb(new std::vector<double>);
  std::unique_ptr<std::vector<int> > pStatus(new std::vector<int>);

  unsigned int iComb = 0;
  for (std::list<TtFullHadKinFitter::KinFitResult>::const_iterator res = fitResults.begin(); res != fitResults.end();
       ++res) {
    if (maxNComb_ >= 1 && iComb == (unsigned int)maxNComb_) {
      break;
    }
    ++iComb;

    pPartonsB->push_back(res->B);
    pPartonsBBar->push_back(res->BBar);
    pPartonsLightQ->push_back(res->LightQ);
    pPartonsLightQBar->push_back(res->LightQBar);
    pPartonsLightP->push_back(res->LightP);
    pPartonsLightPBar->push_back(res->LightPBar);

    pCombi->push_back(res->JetCombi);
    pChi2->push_back(res->Chi2);
    pProb->push_back(res->Prob);
    pStatus->push_back(res->Status);
  }

  event.put(std::move(pCombi));
  event.put(std::move(pPartonsB), "PartonsB");
  event.put(std::move(pPartonsBBar), "PartonsBBar");
  event.put(std::move(pPartonsLightQ), "PartonsLightQ");
  event.put(std::move(pPartonsLightQBar), "PartonsLightQBar");
  event.put(std::move(pPartonsLightP), "PartonsLightP");
  event.put(std::move(pPartonsLightPBar), "PartonsLightPBar");
  event.put(std::move(pChi2), "Chi2");
  event.put(std::move(pProb), "Prob");
  event.put(std::move(pStatus), "Status");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtFullHadKinFitProducer);
