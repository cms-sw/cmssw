#ifndef TauAnalysis_MCEmbeddingTools_AcceptanceHistoProducer_h
#define TauAnalysis_MCEmbeddingTools_AcceptanceHistoProducer_h

/** \class AcceptanceHistoProducer
 *
 * Produces di-muon acceptance histograms and stores them in lumi section.
 * 
 * \author Armin Burgmeier, DESY
 *
 * \version $Revision: 1.1 $
 *
 * $Id: AcceptanceHistoProducer.h,v 1.1 2012/10/14 12:22:24 aburgmei Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TH2D.h>

class AcceptanceHistoProducer : public edm::EDProducer
{
 public:

  // constructor 
  explicit AcceptanceHistoProducer(const edm::ParameterSet& cfg);

  // destructor
  virtual ~AcceptanceHistoProducer();

 private:
  void beginLuminosityBlock(edm::LuminosityBlock& lumi, const edm::EventSetup&);
  void produce(edm::Event&, const edm::EventSetup&);
  void endLuminosityBlock(edm::LuminosityBlock& lumi, const edm::EventSetup&);

  edm::InputTag srcGenParticles_;

  TH2D* hPtPosPtNeg_;
  TH2D* hEtaPosEtaNeg_;
  TH2D* hPtPosEtaPos_;
  TH2D* hPtNegEtaNeg_;
};

#endif    
