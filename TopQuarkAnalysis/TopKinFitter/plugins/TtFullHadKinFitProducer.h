#ifndef TtFullHadKinFitProducer_h
#define TtFullHadKinFitProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/JetMCUtils/interface/combination.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtFullHadKinFitter.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

/*
  \class   TtFullHadKinFitProducer TtFullHadKinFitProducer.h "TopQuarkAnalysis/TopKinFitter/plugins/TtFullHadKinFitProducer.h"
  
  \brief   one line description to be added here...

  text to be added here...
  
**/

class TtFullHadKinFitProducer : public edm::EDProducer {
  
 public:
  /// default constructor  
  explicit TtFullHadKinFitProducer(const edm::ParameterSet& cfg);
  /// default destructor
  ~TtFullHadKinFitProducer();
  
 private:
  /// produce fitted object collections and meta data describing fit quality
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);
  // convert unsigned to Param
  TtFullHadKinFitter::Param param(unsigned int configParameter);
  // convert unsigned int to Constraint
  TtFullHadKinFitter::Constraint constraint(unsigned int configParameter);
  // convert vector of unsigned int's to vector of Contraint's
  std::vector<TtFullHadKinFitter::Constraint> constraints(std::vector<unsigned int>& configParameters);
  // helper function for b-tagging
  bool doBTagging(unsigned int& bTags_, unsigned int& bJetCounter,
		  edm::Handle<std::vector<pat::Jet> >& jets, std::vector<int>& combi,
		  std::string& bTagAlgo_, double& minBTagValueBJets_, double& maxBTagValueNonBJets_);
  /// helper function to construct the proper corrected jet for its corresponding quarkType
    pat::Jet corJet(const pat::Jet& jet, const std::string& quarkType);

 private:
  /// input tag for jets
  edm::InputTag jets_;
  /// input tag for matches (in case the fit should be performed on certain matches)
  edm::InputTag match_;
  /// switch to tell whether all possible combinations should be used for the fit 
  /// or only a certain combination
  bool useOnlyMatch_;
  /// input tag for b-tagging algorithm
  std::string bTagAlgo_;
  /// min value of bTag for a b-jet
  double minBTagValueBJet_;
  /// max value of bTag for a non-b-jet
  double maxBTagValueNonBJet_;
  /// switch to tell whether to use b-tagging with 1 or 2 b-jets or not (=0)
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

  /// kinematic fit interface
  TtFullHadKinFitter* fitter;
  /// struct for fit results
  struct KinFitResult {
    int Status;
    double Chi2;
    double Prob;
    pat::Particle B;
    pat::Particle BBar;
    pat::Particle LightQ;
    pat::Particle LightQBar;
    pat::Particle LightP;
    pat::Particle LightPBar;
    std::vector<int> JetCombi;
    bool operator< (const KinFitResult& rhs) { return Chi2 < rhs.Chi2; };
  };
};

#endif
