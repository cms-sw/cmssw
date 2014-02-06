#ifndef TauAnalysis_MCEmbeddingTools_RochesterCorrMuonProducerT_h
#define TauAnalysis_MCEmbeddingTools_RochesterCorrMuonProducerT_h

/** \class RochesterCorrMuonProducerT
 *
 * Apply muon momentum corrections
 *
 * NOTE: The muon momentum corrections account for residual differences in alignment between MC and data.
 *       The corrections are documented on the wiki
 *         http://www-cdf.fnal.gov/~jyhan/cms_momscl/cms_rochcor_manual.html
 *       and in CMS analysis notes AN-12/062 and AN-12/298.
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: RochesterCorrMuonProducerT.h,v 1.2 2013/01/15 14:53:15 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "TauAnalysis/MCEmbeddingTools/interface/rochcor.h"

#include <TLorentzVector.h>

#include <vector>

template<typename T>
class RochesterCorrMuonProducerT : public edm::EDProducer  
{
  typedef std::vector<T> MuonCollection;

 public:

  explicit RochesterCorrMuonProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
  {
    src_ = cfg.getParameter<edm::InputTag>("src");
    isMC_ = cfg.getParameter<bool>("isMC");

    produces<MuonCollection>("");
  }
  ~RochesterCorrMuonProducerT() {}
    
 private:

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    //std::cout << "<RochesterCorrMuonProducer>:" << std::endl;

    std::auto_ptr<MuonCollection> correctedMuonCollection(new MuonCollection);

    edm::Handle<MuonCollection> uncorrectedMuonCollection;
    evt.getByLabel(src_, uncorrectedMuonCollection);

    int idx = 0;
    for ( typename MuonCollection::const_iterator uncorrectedMuon = uncorrectedMuonCollection->begin();
	  uncorrectedMuon != uncorrectedMuonCollection->end(); ++uncorrectedMuon ) {
      //std::cout << " uncorrected Muon #" << idx << ": Pt = " << uncorrectedMuon->pt() << "," 
      //	  << " eta = " << uncorrectedMuon->eta() << ", phi = " << uncorrectedMuon->phi() << ", mass = " << uncorrectedMuon->mass() << std::endl;

      TLorentzVector muonP4(uncorrectedMuon->px(), uncorrectedMuon->py(), uncorrectedMuon->pz(), uncorrectedMuon->energy());

      float error;
      if ( isMC_ ) algorithm_.momcor_mc(muonP4, uncorrectedMuon->charge(), 0, 0, error);
      else algorithm_.momcor_data(muonP4, uncorrectedMuon->charge(), 0, 0, error);

      T correctedMuon(*uncorrectedMuon);
      // CV: use PolarLorentzVector to preserve muon mass (Rochester muon corrections change muon mass)
      correctedMuon.setP4(reco::Candidate::PolarLorentzVector(muonP4.Pt(), muonP4.Eta(), muonP4.Phi(), uncorrectedMuon->mass()));
      //std::cout << " corrected Muon #" << idx << ": Pt = " << correctedMuon.pt() << "," 
      //	  << " eta = " << correctedMuon.eta() << ", phi = " << correctedMuon.phi() << ", mass = " << correctedMuon.mass() << std::endl;

      correctedMuonCollection->push_back(correctedMuon);

      ++idx;
    }

    evt.put(correctedMuonCollection);
  }

  std::string moduleLabel_;

  edm::InputTag src_; 
  bool isMC_;

  rochcor algorithm_;
};

#endif

 
