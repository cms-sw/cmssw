#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"


class TtDilepEvtSolutionMaker : public edm::EDProducer {

  public:

    explicit TtDilepEvtSolutionMaker(const edm::ParameterSet & iConfig);
    ~TtDilepEvtSolutionMaker();
  
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:  

    // next methods are avoidable but they make the code legible
    bool PTComp(const TopElectron * e, const TopMuon * m) const;
    bool LepDiffCharge(const TopElectron * e, const TopMuon * m) const;
    bool LepDiffCharge(const TopElectron * e1, const TopElectron * e2) const;
    bool LepDiffCharge(const TopMuon * m1, const TopMuon * m2) const;
    bool HasPositiveCharge(const TopMuon * m) const;
    bool HasPositiveCharge(const TopElectron * e) const;

  private:

    edm::InputTag electronSource_;
    edm::InputTag muonSource_;
    edm::InputTag metSource_;
    edm::InputTag jetSource_;
    bool matchToGenEvt_, calcTopMass_;
    bool eeChannel_, emuChannel_, mumuChannel_;
    double tmassbegin_, tmassend_, tmassstep_;

};
