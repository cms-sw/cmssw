#ifndef TtFullHadKinFitProducer_h
#define TtFullHadKinFitProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
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

class TtFullHadKinFitProducer : public edm::EDProducer {
  
 public:
  /// default constructor  
  explicit TtFullHadKinFitProducer(const edm::ParameterSet& cfg);
  /// default destructor
  ~TtFullHadKinFitProducer();
  
 private:
  /// produce fitted object collections and meta data describing fit quality
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);

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
  /// smearing factor for jet energy resolutions
  double energyResolutionSmearFactor_;
  std::vector<double> etaDependentResSmearFactor_;
  std::vector<double> etaBinningForSmearFactor_;

 public:

  /// kinematic fit interface
  TtFullHadKinFitter::KinFit* kinFitter;
 
};

#endif
