#ifndef TauAnalysis_MCEmbeddingTools_AcceptanceHistoProducer_h
#define TauAnalysis_MCEmbeddingTools_AcceptanceHistoProducer_h

/** \class AcceptanceHistoProducer
 *
 * Produces di-muon acceptance histograms and stores them in lumi section.
 * 
 * \author Armin Burgmeier, DESY
 *
 * \version $Revision: 1.2 $
 *
 * $Id: AcceptanceHistoProducer.h,v 1.2 2012/11/07 17:03:01 aburgmei Exp $
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

class AcceptanceHistoProducer : public edm::EDAnalyzer
{
 public:

  // constructor 
  explicit AcceptanceHistoProducer(const edm::ParameterSet& cfg);

  // destructor
  virtual ~AcceptanceHistoProducer();

 private:
  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);

  std::string dqmDir_;
  edm::InputTag srcGenParticles_;

  DQMStore* dbe_;

  MonitorElement* hPtPosPtNeg_;
  MonitorElement* hEtaPosEtaNeg_;
  MonitorElement* hPtPosEtaPos_;
  MonitorElement* hPtNegEtaNeg_;
};

#endif    
