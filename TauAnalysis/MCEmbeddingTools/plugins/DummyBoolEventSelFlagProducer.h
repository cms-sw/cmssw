#ifndef TauAnalysis_MCEmbeddingTools_DummyBoolEventSelFlagProducer_h
#define TauAnalysis_MCEmbeddingTools_DummyBoolEventSelFlagProducer_h

/** \class DummyBoolEventSelFlagProducer
 *
 * Produce boolean flag that is always set to true
 * (indicating that a whole path has been run up to the end, 
 *  passing all EDFilter modules which are in this path)
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: DummyBoolEventSelFlagProducer.h,v 1.1 2012/10/14 12:22:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DummyBoolEventSelFlagProducer : public edm::EDProducer
{
 public:

  // constructor 
  explicit DummyBoolEventSelFlagProducer(const edm::ParameterSet& cfg);

  // destructor
  virtual ~DummyBoolEventSelFlagProducer();

 private:

  // method for adding boolean flag to the event
  void produce(edm::Event&, const edm::EventSetup&);
};

#endif    
