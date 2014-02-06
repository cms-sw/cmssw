#ifndef TauAnalysis_MCEmbeddingTools_ZmumuEvtSelEffCorrWeightProducer_h
#define TauAnalysis_MCEmbeddingTools_ZmumuEvtSelEffCorrWeightProducer_h

/** \class ZmumuEvtSelEffCorrWeightProducer
 *
 * Compute weight factor and uncertainty for correcting Embedded samples 
 * for efficiency with which Zmumu events used as input for Embedding production were selected
 *
 * \authors Christian Veelken
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ZmumuEvtSelEffCorrWeightProducer.h,v 1.1 2013/02/21 14:08:41 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include <TH2.h>
#include <TAxis.h>

#include <string>

class ZmumuEvtSelEffCorrWeightProducer : public edm::EDProducer 
{
 public:
  explicit ZmumuEvtSelEffCorrWeightProducer(const edm::ParameterSet&);
  ~ZmumuEvtSelEffCorrWeightProducer();

  void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag srcSelectedMuons_;
    
  TH2* lutEfficiencyPt_;
  TAxis* xAxisEfficiencyPt_;
  TAxis* yAxisEfficiencyPt_;
  TH2* lutEffCorrEta_;
  TAxis* xAxisEffCorrEta_;
  TAxis* yAxisEffCorrEta_;

  double minWeight_;
  double maxWeight_;

  int verbosity_;
};

#endif

