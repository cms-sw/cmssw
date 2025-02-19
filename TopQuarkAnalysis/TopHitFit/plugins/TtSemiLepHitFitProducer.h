#ifndef TtSemiLepHitFitProducer_h
#define TtSemiLepHitFitProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/JetMCUtils/interface/combination.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "TopQuarkAnalysis/TopHitFit/interface/RunHitFit.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Top_Decaykin.h"
#include "TopQuarkAnalysis/TopHitFit/interface/LeptonTranslatorBase.h"
#include "TopQuarkAnalysis/TopHitFit/interface/JetTranslatorBase.h"
#include "TopQuarkAnalysis/TopHitFit/interface/METTranslatorBase.h"

template <typename LeptonCollection>
class TtSemiLepHitFitProducer : public edm::EDProducer {

 public:

  explicit TtSemiLepHitFitProducer(const edm::ParameterSet&);
  ~TtSemiLepHitFitProducer();

 private:
  // produce
  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag jets_;
  edm::InputTag leps_;
  edm::InputTag mets_;

  /// maximal number of jets (-1 possible to indicate 'all')
  int maxNJets_;
  /// maximal number of combinations to be written to the event
  int maxNComb_;

  /// maximum eta value for muons, needed to limited range in which resolutions are provided
  double maxEtaMu_;
  /// maximum eta value for electrons, needed to limited range in which resolutions are provided
  double maxEtaEle_;
  /// maximum eta value for jets, needed to limited range in which resolutions are provided
  double maxEtaJet_;

  /// input tag for b-tagging algorithm
  std::string bTagAlgo_;
  /// min value of bTag for a b-jet
  double minBTagValueBJet_;
  /// max value of bTag for a non-b-jet
  double maxBTagValueNonBJet_;
  /// switch to tell whether to use b-tagging or not
  bool useBTag_;

  /// constraints
  double mW_;
  double mTop_;

  /// jet correction level
  std::string jetCorrectionLevel_;

  /// jet energy scale
  double jes_;
  double jesB_;

  struct FitResult {
    int Status;
    double Chi2;
    double Prob;
    double MT;
    double SigMT;
    pat::Particle HadB;
    pat::Particle HadP;
    pat::Particle HadQ;
    pat::Particle LepB;
    pat::Particle LepL;
    pat::Particle LepN;
    std::vector<int> JetCombi;
    bool operator< (const FitResult& rhs) { return Chi2 < rhs.Chi2; };
  };

  typedef hitfit::RunHitFit<pat::Electron,pat::Muon,pat::Jet,pat::MET> PatHitFit;

  edm::FileInPath hitfitDefault_;
  edm::FileInPath hitfitElectronResolution_;
  edm::FileInPath hitfitMuonResolution_;
  edm::FileInPath hitfitUdscJetResolution_;
  edm::FileInPath hitfitBJetResolution_;
  edm::FileInPath hitfitMETResolution_;

  hitfit::LeptonTranslatorBase<pat::Electron> electronTranslator_;
  hitfit::LeptonTranslatorBase<pat::Muon>     muonTranslator_;
  hitfit::JetTranslatorBase<pat::Jet>         jetTranslator_;
  hitfit::METTranslatorBase<pat::MET>         metTranslator_;

  PatHitFit* HitFit;
};

template<typename LeptonCollection>
TtSemiLepHitFitProducer<LeptonCollection>::TtSemiLepHitFitProducer(const edm::ParameterSet& cfg):
  jets_                    (cfg.getParameter<edm::InputTag>("jets")),
  leps_                    (cfg.getParameter<edm::InputTag>("leps")),
  mets_                    (cfg.getParameter<edm::InputTag>("mets")),
  maxNJets_                (cfg.getParameter<int>          ("maxNJets"            )),
  maxNComb_                (cfg.getParameter<int>          ("maxNComb"            )),
  bTagAlgo_                (cfg.getParameter<std::string>  ("bTagAlgo"            )),
  minBTagValueBJet_        (cfg.getParameter<double>       ("minBDiscBJets"       )),
  maxBTagValueNonBJet_     (cfg.getParameter<double>       ("maxBDiscLightJets"   )),
  useBTag_                 (cfg.getParameter<bool>         ("useBTagging"         )),
  mW_                      (cfg.getParameter<double>       ("mW"                  )),
  mTop_                    (cfg.getParameter<double>       ("mTop"                )),
  jetCorrectionLevel_      (cfg.getParameter<std::string>  ("jetCorrectionLevel"  )),
  jes_                     (cfg.getParameter<double>       ("jes"                 )),
  jesB_                    (cfg.getParameter<double>       ("jesB"                )),

  // The following five initializers read the config parameters for the
  // ASCII text files which contains the physics object resolutions.
  hitfitDefault_           (cfg.getUntrackedParameter<edm::FileInPath>(std::string("hitfitDefault"),
                            edm::FileInPath(std::string("TopQuarkAnalysis/TopHitFit/data/setting/RunHitFitConfiguration.txt")))),
  hitfitElectronResolution_(cfg.getUntrackedParameter<edm::FileInPath>(std::string("hitfitElectronResolution"),
                            edm::FileInPath(std::string("TopQuarkAnalysis/TopHitFit/data/resolution/tqafElectronResolution.txt")))),
  hitfitMuonResolution_    (cfg.getUntrackedParameter<edm::FileInPath>(std::string("hitfitMuonResolution"),
                            edm::FileInPath(std::string("TopQuarkAnalysis/TopHitFit/data/resolution/tqafMuonResolution.txt")))),
  hitfitUdscJetResolution_ (cfg.getUntrackedParameter<edm::FileInPath>(std::string("hitfitUdscJetResolution"),
                            edm::FileInPath(std::string("TopQuarkAnalysis/TopHitFit/data/resolution/tqafUdscJetResolution.txt")))),
  hitfitBJetResolution_    (cfg.getUntrackedParameter<edm::FileInPath>(std::string("hitfitBJetResolution"),
                            edm::FileInPath(std::string("TopQuarkAnalysis/TopHitFit/data/resolution/tqafBJetResolution.txt")))),
  hitfitMETResolution_     (cfg.getUntrackedParameter<edm::FileInPath>(std::string("hitfitMETResolution"),
                            edm::FileInPath(std::string("TopQuarkAnalysis/TopHitFit/data/resolution/tqafKtResolution.txt")))),

  // The following four initializers instantiate the translator between PAT objects
  // and HitFit objects using the ASCII text files which contains the resolutions.
  electronTranslator_(hitfitElectronResolution_.fullPath()),
  muonTranslator_    (hitfitMuonResolution_.fullPath()),
  jetTranslator_     (hitfitUdscJetResolution_.fullPath(), hitfitBJetResolution_.fullPath(), jetCorrectionLevel_, jes_, jesB_),
  metTranslator_     (hitfitMETResolution_.fullPath())

{
  // Create an instance of RunHitFit and initialize it.
  HitFit = new PatHitFit(electronTranslator_,
                         muonTranslator_,
                         jetTranslator_,
                         metTranslator_,
                         hitfitDefault_.fullPath(),
                         mW_,
                         mW_,
                         mTop_);

  maxEtaMu_  = 2.4;
  maxEtaEle_ = 2.5;
  maxEtaJet_ = 2.5;

  edm::LogVerbatim( "TopHitFit" )
    << "\n"
    << "+++++++++++ TtSemiLepHitFitProducer ++++++++++++ \n"
    << " Due to the eta ranges for which resolutions     \n"
    << " are provided in                                 \n"
    << " TopQuarkAnalysis/TopHitFit/data/resolution/     \n"
    << " so far, the following cuts are currently        \n"
    << " implemented in the TtSemiLepHitFitProducer:     \n"
    << " |eta(muons    )| <= " << maxEtaMu_  <<        " \n"
    << " |eta(electrons)| <= " << maxEtaEle_ <<        " \n"
    << " |eta(jets     )| <= " << maxEtaJet_ <<        " \n"
    << "++++++++++++++++++++++++++++++++++++++++++++++++ \n";

  produces< std::vector<pat::Particle> >("PartonsHadP");
  produces< std::vector<pat::Particle> >("PartonsHadQ");
  produces< std::vector<pat::Particle> >("PartonsHadB");
  produces< std::vector<pat::Particle> >("PartonsLepB");
  produces< std::vector<pat::Particle> >("Leptons");
  produces< std::vector<pat::Particle> >("Neutrinos");

  produces< std::vector<std::vector<int> > >();
  produces< std::vector<double> >("Chi2");
  produces< std::vector<double> >("Prob");
  produces< std::vector<double> >("MT");
  produces< std::vector<double> >("SigMT");
  produces< std::vector<int> >("Status");
  produces< int >("NumberOfConsideredJets");
}

template<typename LeptonCollection>
TtSemiLepHitFitProducer<LeptonCollection>::~TtSemiLepHitFitProducer()
{
  delete HitFit;
}

template<typename LeptonCollection>
void TtSemiLepHitFitProducer<LeptonCollection>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr< std::vector<pat::Particle> > pPartonsHadP( new std::vector<pat::Particle> );
  std::auto_ptr< std::vector<pat::Particle> > pPartonsHadQ( new std::vector<pat::Particle> );
  std::auto_ptr< std::vector<pat::Particle> > pPartonsHadB( new std::vector<pat::Particle> );
  std::auto_ptr< std::vector<pat::Particle> > pPartonsLepB( new std::vector<pat::Particle> );
  std::auto_ptr< std::vector<pat::Particle> > pLeptons    ( new std::vector<pat::Particle> );
  std::auto_ptr< std::vector<pat::Particle> > pNeutrinos  ( new std::vector<pat::Particle> );

  std::auto_ptr< std::vector<std::vector<int> > > pCombi ( new std::vector<std::vector<int> > );
  std::auto_ptr< std::vector<double>            > pChi2  ( new std::vector<double> );
  std::auto_ptr< std::vector<double>            > pProb  ( new std::vector<double> );
  std::auto_ptr< std::vector<double>            > pMT    ( new std::vector<double> );
  std::auto_ptr< std::vector<double>            > pSigMT ( new std::vector<double> );
  std::auto_ptr< std::vector<int>               > pStatus( new std::vector<int> );
  std::auto_ptr< int > pJetsConsidered( new int );

  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  edm::Handle<std::vector<pat::MET> > mets;
  evt.getByLabel(mets_, mets);

  edm::Handle<LeptonCollection> leps;
  evt.getByLabel(leps_, leps);

  // -----------------------------------------------------
  // skip events with no appropriate lepton candidate in
  // or empty MET or less jets than partons
  // -----------------------------------------------------

  const unsigned int nPartons = 4;

  // Clear the internal state
  HitFit->clear();

  // Add lepton into HitFit
  bool foundLepton = false;
  if(!leps->empty()) {
    double maxEtaLep = maxEtaMu_;
    if( !dynamic_cast<const reco::Muon*>(&((*leps)[0])) ) // assume electron if it is not a muon
      maxEtaLep = maxEtaEle_;
    for(unsigned iLep=0; iLep<(*leps).size() && !foundLepton; ++iLep) {
      if(std::abs((*leps)[iLep].eta()) <= maxEtaLep) {
	HitFit->AddLepton((*leps)[iLep]);
	foundLepton = true;
      }
    }
  }

  // Add jets into HitFit
  unsigned int nJetsFound = 0;
  for(unsigned iJet=0; iJet<(*jets).size() && (int)nJetsFound!=maxNJets_; ++iJet) {
    if(std::abs((*jets)[iJet].eta()) <= maxEtaJet_) {
      HitFit->AddJet((*jets)[iJet]);
      nJetsFound++;
    }
  }
  *pJetsConsidered = nJetsFound;

  // Add missing transverse energy into HitFit
  if(!mets->empty())
    HitFit->SetMet((*mets)[0]);

  if( !foundLepton || mets->empty() || nJetsFound<nPartons ) {
    // the kinFit getters return empty objects here
    pPartonsHadP->push_back( pat::Particle() );
    pPartonsHadQ->push_back( pat::Particle() );
    pPartonsHadB->push_back( pat::Particle() );
    pPartonsLepB->push_back( pat::Particle() );
    pLeptons    ->push_back( pat::Particle() );
    pNeutrinos  ->push_back( pat::Particle() );
    // indices referring to the jet combination
    std::vector<int> invalidCombi;
    for(unsigned int i = 0; i < nPartons; ++i)
      invalidCombi.push_back( -1 );
    pCombi->push_back( invalidCombi );
    // chi2
    pChi2->push_back( -1. );
    // chi2 probability
    pProb->push_back( -1. );
    // fitted top mass
    pMT->push_back( -1. );
    pSigMT->push_back( -1. );
    // status of the fitter
    pStatus->push_back( -1 );
    // feed out all products
    evt.put(pCombi);
    evt.put(pPartonsHadP, "PartonsHadP");
    evt.put(pPartonsHadQ, "PartonsHadQ");
    evt.put(pPartonsHadB, "PartonsHadB");
    evt.put(pPartonsLepB, "PartonsLepB");
    evt.put(pLeptons    , "Leptons"    );
    evt.put(pNeutrinos  , "Neutrinos"  );
    evt.put(pChi2       , "Chi2"       );
    evt.put(pProb       , "Prob"       );
    evt.put(pMT         , "MT"         );
    evt.put(pSigMT      , "SigMT"      );
    evt.put(pStatus     , "Status"     );
    evt.put(pJetsConsidered, "NumberOfConsideredJets");
    return;
  }

  std::list<FitResult> FitResultList;

  //
  // BEGIN DECLARATION OF VARIABLES FROM KINEMATIC FIT
  //

  // In this part are variables from the
  // kinematic fit procedure

  // Number of all permutations of the event
  size_t nHitFit    = 0 ;

  // Number of jets in the event
  size_t nHitFitJet = 0 ;

  // Results of the fit for all jet permutations of the event
  std::vector<hitfit::Fit_Result> hitFitResult;

  //
  // R U N   H I T F I T
  //
  // Run the kinematic fit and get how many permutations are possible
  // in the fit

  nHitFit         = HitFit->FitAllPermutation();

  //
  // BEGIN PART WHICH EXTRACTS INFORMATION FROM HITFIT
  //

  // Get the number of jets
  nHitFitJet = HitFit->GetUnfittedEvent()[0].njets();

  // Get the fit results for all permutations
  hitFitResult = HitFit->GetFitAllPermutation();

  // Loop over all permutations and extract the information
  for (size_t fit = 0 ; fit != nHitFit ; ++fit) {

      // Get the event after the fit
      hitfit::Lepjets_Event fittedEvent = hitFitResult[fit].ev();

      /*
        Get jet permutation according to TQAF convention
        11 : leptonic b
        12 : hadronic b
        13 : hadronic W
        14 : hadronic W
      */
      std::vector<int> hitCombi(4);
      for (size_t jet = 0 ; jet != nHitFitJet ; ++jet) {
          int jet_type = fittedEvent.jet(jet).type();

          switch(jet_type) {
            case 11: hitCombi[TtSemiLepEvtPartons::LepB     ] = jet;
              break;
            case 12: hitCombi[TtSemiLepEvtPartons::HadB     ] = jet;
              break;
            case 13: hitCombi[TtSemiLepEvtPartons::LightQ   ] = jet;
              break;
            case 14: hitCombi[TtSemiLepEvtPartons::LightQBar] = jet;
              break;
          }
      }

      // Store the kinematic quantities in the corresponding containers.

      hitfit::Lepjets_Event_Jet hadP_ = fittedEvent.jet(hitCombi[TtSemiLepEvtPartons::LightQ   ]);
      hitfit::Lepjets_Event_Jet hadQ_ = fittedEvent.jet(hitCombi[TtSemiLepEvtPartons::LightQBar]);
      hitfit::Lepjets_Event_Jet hadB_ = fittedEvent.jet(hitCombi[TtSemiLepEvtPartons::HadB     ]);
      hitfit::Lepjets_Event_Jet lepB_ = fittedEvent.jet(hitCombi[TtSemiLepEvtPartons::LepB     ]);
      hitfit::Lepjets_Event_Lep lepL_ = fittedEvent.lep(0);

      /*
  /// input tag for b-tagging algorithm
  std::string bTagAlgo_;
  /// min value of bTag for a b-jet
  double minBTagValueBJet_;
  /// max value of bTag for a non-b-jet
  double maxBTagValueNonBJet_;
  /// switch to tell whether to use b-tagging or not
  bool useBTag_;
      */

      if (   hitFitResult[fit].chisq() > 0    // only take into account converged fits
          && (!useBTag_ || (   useBTag_       // use btag information if chosen
                            && jets->at(hitCombi[TtSemiLepEvtPartons::LightQ   ]).bDiscriminator(bTagAlgo_) < maxBTagValueNonBJet_
                            && jets->at(hitCombi[TtSemiLepEvtPartons::LightQBar]).bDiscriminator(bTagAlgo_) < maxBTagValueNonBJet_
                            && jets->at(hitCombi[TtSemiLepEvtPartons::HadB     ]).bDiscriminator(bTagAlgo_) > minBTagValueBJet_
                            && jets->at(hitCombi[TtSemiLepEvtPartons::LepB     ]).bDiscriminator(bTagAlgo_) > minBTagValueBJet_
                            )
             )
          ) {
        FitResult result;
        result.Status = 0;
        result.Chi2 = hitFitResult[fit].chisq();
        result.Prob = exp(-1.0*(hitFitResult[fit].chisq())/2.0);
        result.MT   = hitFitResult[fit].mt();
        result.SigMT= hitFitResult[fit].sigmt();
        result.HadB = pat::Particle(reco::LeafCandidate(0,
                                       math::XYZTLorentzVector(hadB_.p().x(), hadB_.p().y(),
                                       hadB_.p().z(), hadB_.p().t()), math::XYZPoint()));
        result.HadP = pat::Particle(reco::LeafCandidate(0,
                                       math::XYZTLorentzVector(hadP_.p().x(), hadP_.p().y(),
                                       hadP_.p().z(), hadP_.p().t()), math::XYZPoint()));
        result.HadQ = pat::Particle(reco::LeafCandidate(0,
                                       math::XYZTLorentzVector(hadQ_.p().x(), hadQ_.p().y(),
                                       hadQ_.p().z(), hadQ_.p().t()), math::XYZPoint()));
        result.LepB = pat::Particle(reco::LeafCandidate(0,
                                       math::XYZTLorentzVector(lepB_.p().x(), lepB_.p().y(),
                                       lepB_.p().z(), lepB_.p().t()), math::XYZPoint()));
        result.LepL = pat::Particle(reco::LeafCandidate(0,
                                       math::XYZTLorentzVector(lepL_.p().x(), lepL_.p().y(),
                                       lepL_.p().z(), lepL_.p().t()), math::XYZPoint()));
        result.LepN = pat::Particle(reco::LeafCandidate(0,
                                       math::XYZTLorentzVector(fittedEvent.met().x(), fittedEvent.met().y(),
                                       fittedEvent.met().z(), fittedEvent.met().t()), math::XYZPoint()));
        result.JetCombi = hitCombi;

        FitResultList.push_back(result);
      }

  }

  // sort results w.r.t. chi2 values
  FitResultList.sort();

  // -----------------------------------------------------
  // feed out result
  // starting with the JetComb having the smallest chi2
  // -----------------------------------------------------

  if( ((unsigned)FitResultList.size())<1 ) { // in case no fit results were stored in the list (all fits aborted)
    pPartonsHadP->push_back( pat::Particle() );
    pPartonsHadQ->push_back( pat::Particle() );
    pPartonsHadB->push_back( pat::Particle() );
    pPartonsLepB->push_back( pat::Particle() );
    pLeptons    ->push_back( pat::Particle() );
    pNeutrinos  ->push_back( pat::Particle() );
    // indices referring to the jet combination
    std::vector<int> invalidCombi;
    for(unsigned int i = 0; i < nPartons; ++i)
      invalidCombi.push_back( -1 );
    pCombi->push_back( invalidCombi );
    // chi2
    pChi2->push_back( -1. );
    // chi2 probability
    pProb->push_back( -1. );
    // fitted top mass
    pMT->push_back( -1. );
    pSigMT->push_back( -1. );
    // status of the fitter
    pStatus->push_back( -1 );
  }
  else {
    unsigned int iComb = 0;
    for(typename std::list<FitResult>::const_iterator result = FitResultList.begin(); result != FitResultList.end(); ++result) {
      if(maxNComb_ >= 1 && iComb == (unsigned int) maxNComb_) break;
      iComb++;
      // partons
      pPartonsHadP->push_back( result->HadP );
      pPartonsHadQ->push_back( result->HadQ );
      pPartonsHadB->push_back( result->HadB );
      pPartonsLepB->push_back( result->LepB );
      // lepton
      pLeptons->push_back( result->LepL );
      // neutrino
      pNeutrinos->push_back( result->LepN );
      // indices referring to the jet combination
      pCombi->push_back( result->JetCombi );
      // chi2
      pChi2->push_back( result->Chi2 );
      // chi2 probability
      pProb->push_back( result->Prob );
      // fitted top mass
      pMT->push_back( result->MT );
      pSigMT->push_back( result->SigMT );
      // status of the fitter
      pStatus->push_back( result->Status );
    }
  }
  evt.put(pCombi);
  evt.put(pPartonsHadP, "PartonsHadP");
  evt.put(pPartonsHadQ, "PartonsHadQ");
  evt.put(pPartonsHadB, "PartonsHadB");
  evt.put(pPartonsLepB, "PartonsLepB");
  evt.put(pLeptons    , "Leptons"    );
  evt.put(pNeutrinos  , "Neutrinos"  );
  evt.put(pChi2       , "Chi2"       );
  evt.put(pProb       , "Prob"       );
  evt.put(pMT         , "MT"         );
  evt.put(pSigMT      , "SigMT"      );
  evt.put(pStatus     , "Status"     );
  evt.put(pJetsConsidered, "NumberOfConsideredJets");
}

#endif
