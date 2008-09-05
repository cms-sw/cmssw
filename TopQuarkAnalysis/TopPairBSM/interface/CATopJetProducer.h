#ifndef TopQuarkAnalysis_TopPairBSM_CATopJetProducer_h
#define TopQuarkAnalysis_TopPairBSM_CATopJetProducer_h


/* *********************************************************
 * \class CATopJetProducer
 * Jet producer to produce top jets using the C-A algorithm to break
 * jets into subjets as described here:
 * "Top-tagging: A Method for Identifying Boosted Hadronic Tops"
 * David E. Kaplan, Keith Rehermann, Matthew D. Schwartz, Brock Tweedie
 * arXiv:0806.0848v1 [hep-ph] 
 *
 ************************************************************/


#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoJets/JetProducers/src/BaseJetProducer.h"
#include "TopQuarkAnalysis/TopPairBSM/interface/CATopJetAlgorithm.h"

namespace cms
{
  class CATopJetProducer : public edm::EDProducer
  {
  public:

    CATopJetProducer(const edm::ParameterSet& ps);

    virtual ~CATopJetProducer() {}

    //Produces the EDM products
    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    CATopJetAlgorithm        alg_;
  };
}


#endif
