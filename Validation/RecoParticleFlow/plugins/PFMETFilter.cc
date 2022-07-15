// author: Florent Lacroix (UIC)
// date: 07/14/2009

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/MET.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class PFMETFilter : public edm::global::EDFilter<> {
public:
  explicit PFMETFilter(const edm::ParameterSet &);
  ~PFMETFilter() override;

  bool filter(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  void beginJob() override;
  void endJob() override;
  bool checkInput() const;

private:
  std::vector<std::string> collections_;
  std::vector<std::string> variables_;
  std::vector<double> min_;
  std::vector<double> max_;
  std::vector<int> doMin_;
  std::vector<int> doMax_;
  // parameters for the cut:
  // sqrt(DeltaMEX**2+DeltaMEY**2)>DeltaMEXsigma*sigma
  // with sigma=sigma_a+sigma_b*sqrt(SET)+sigma_c*SET
  std::string TrueMET_;
  double DeltaMEXsigma_;
  double sigma_a_;
  double sigma_b_;
  double sigma_c_;
  bool verbose_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFMETFilter);

PFMETFilter::PFMETFilter(const edm::ParameterSet &iConfig) {
  collections_ = iConfig.getParameter<std::vector<std::string>>("Collections");
  variables_ = iConfig.getParameter<std::vector<std::string>>("Variables");
  min_ = iConfig.getParameter<std::vector<double>>("Mins");
  max_ = iConfig.getParameter<std::vector<double>>("Maxs");
  doMin_ = iConfig.getParameter<std::vector<int>>("DoMin");
  doMax_ = iConfig.getParameter<std::vector<int>>("DoMax");
  TrueMET_ = iConfig.getParameter<std::string>("TrueMET");
  DeltaMEXsigma_ = iConfig.getParameter<double>("DeltaMEXsigma");
  sigma_a_ = iConfig.getParameter<double>("sigma_a");
  sigma_b_ = iConfig.getParameter<double>("sigma_b");
  sigma_c_ = iConfig.getParameter<double>("sigma_c");
  verbose_ = iConfig.getParameter<bool>("verbose");
}

PFMETFilter::~PFMETFilter() {}

bool PFMETFilter::checkInput() const {
  if (collections_.size() != min_.size()) {
    std::cout << "Error: in PFMETFilter: collections_.size()!=min_.size()" << std::endl;
    std::cout << "collections_.size() = " << collections_.size() << std::endl;
    std::cout << "min_.size() = " << min_.size() << std::endl;
    return false;
  }
  if (collections_.size() != max_.size()) {
    std::cout << "Error: in PFMETFilter: collections_.size()!=max_.size()" << std::endl;
    std::cout << "collections_.size() = " << collections_.size() << std::endl;
    std::cout << "max_.size() = " << max_.size() << std::endl;
    return false;
  }
  if (collections_.size() != doMin_.size()) {
    std::cout << "Error: in PFMETFilter: collections_.size()!=min_.size()" << std::endl;
    std::cout << "collections_.size() = " << collections_.size() << std::endl;
    std::cout << "doMin_.size() = " << doMin_.size() << std::endl;
    return false;
  }
  if (collections_.size() != doMax_.size()) {
    std::cout << "Error: in PFMETFilter: collections_.size()!=min_.size()" << std::endl;
    std::cout << "collections_.size() = " << collections_.size() << std::endl;
    std::cout << "doMax_.size() = " << doMax_.size() << std::endl;
    return false;
  }
  if (collections_.size() != variables_.size()) {
    std::cout << "Error: in PFMETFilter: collections_.size()!=variables_.size()" << std::endl;
    std::cout << "collections_.size() = " << collections_.size() << std::endl;
    std::cout << "variables_.size() = " << variables_.size() << std::endl;
    return false;
  }
  return true;
}

void PFMETFilter::beginJob() {
  // std::cout << "FL: beginJob" << std::endl;
}

bool PFMETFilter::filter(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  // std::cout << "FL: filter" << std::endl;
  // std::cout << "FL: Mins = " << min_ << std::endl;

  if (!checkInput())
    return true;  // no filtering !

  bool skip = false;

  for (unsigned int varc = 0; varc < collections_.size(); ++varc) {
    // std::cout << "FL: var[" << varc << "] = " << collections_[varc] <<
    // std::endl; std::cout << "FL: var[0] = " << collections_[0] << std::endl;

    // if the collection is collection1-collection2:
    const unsigned int minuspos = collections_[varc].find('-');
    if (minuspos < collections_[varc].size()) {
      std::string collection1;
      collection1.assign(collections_[varc], 0, minuspos);
      // std::cout << "collection1 = " << collection1 << std::endl;
      std::string collection2;
      collection2.assign(collections_[varc], minuspos + 1, collections_[varc].size());
      // std::cout << "collection2 = " << collection2 << std::endl;

      const edm::View<reco::Candidate> *var1;
      edm::Handle<edm::View<reco::Candidate>> var1_hnd;
      bool isVar1 = iEvent.getByLabel(collection1, var1_hnd);
      if (!isVar1) {
        std::cout << "Warning : no " << collection1 << " in input !" << std::endl;
        return false;
      }
      var1 = var1_hnd.product();
      const reco::Candidate *var10 = &(*var1)[0];
      // std::cout << "FL: var10.pt = " << var10->et() << std::endl;
      double coll_var1;
      if (variables_[varc] == "et")
        coll_var1 = var10->et();
      else if (variables_[varc] == "phi")
        coll_var1 = var10->phi();
      else if (variables_[varc] == "eta")
        coll_var1 = var10->eta();
      else {
        std::cout << "Error: PFMETFilter: variable unknown: " << variables_[varc] << std::endl;
        return true;
      }
      // std::cout << "FL: coll_var1[" << variables_[varc] << "] = " <<
      // coll_var1 << std::endl;

      const edm::View<reco::Candidate> *var2;
      edm::Handle<edm::View<reco::Candidate>> var2_hnd;
      bool isVar2 = iEvent.getByLabel(collection2, var2_hnd);
      if (!isVar2) {
        std::cout << "Warning : no " << collection2 << " in input !" << std::endl;
        return false;
      }
      var2 = var2_hnd.product();
      const reco::Candidate *var20 = &(*var2)[0];
      // std::cout << "FL: var20.pt = " << var20->et() << std::endl;
      double coll_var2;
      if (variables_[varc] == "et")
        coll_var2 = var20->et();
      else if (variables_[varc] == "phi")
        coll_var2 = var20->phi();
      else if (variables_[varc] == "eta")
        coll_var2 = var20->eta();
      else {
        std::cout << "Error: PFMETFilter: variable unknown: " << variables_[varc] << std::endl;
        return true;
      }
      // std::cout << "FL: coll_var2[" << variables_[varc] << "] = " <<
      // coll_var2 << std::endl; std::cout << "FL: max_[varc] = " << max_[varc]
      // << std::endl; std::cout << "FL: min_[varc] = " << min_[varc] <<
      // std::endl;

      // Delta computation
      double delta = coll_var1 - coll_var2;
      if (variables_[varc] == "phi") {
        if (coll_var1 > M_PI)
          coll_var1 -= ceil((coll_var1 - M_PI) / (2 * M_PI)) * 2 * M_PI;
        if (coll_var1 <= -M_PI)
          coll_var1 += ceil((coll_var1 + M_PI) / (-2. * M_PI)) * 2. * M_PI;
        if (coll_var2 > M_PI)
          coll_var2 -= ceil((coll_var2 - M_PI) / (2 * M_PI)) * 2 * M_PI;
        if (coll_var2 <= -M_PI)
          coll_var2 += ceil((coll_var2 + M_PI) / (-2. * M_PI)) * 2 * M_PI;

        double deltaphi = -999.0;
        if (fabs(coll_var1 - coll_var2) < M_PI) {
          deltaphi = (coll_var1 - coll_var2);
        } else {
          if ((coll_var1 - coll_var2) > 0.0) {
            deltaphi = (2 * M_PI - fabs(coll_var1 - coll_var2));
          } else {
            deltaphi = -(2 * M_PI - fabs(coll_var1 - coll_var2));
          }
        }
        delta = deltaphi;
      }

      // cuts
      if (doMin_[varc] && doMax_[varc] && max_[varc] < min_[varc]) {
        if (delta > max_[varc] && delta < min_[varc])
          skip = true;
      } else {
        if (doMin_[varc] && delta < min_[varc])
          skip = true;
        if (doMax_[varc] && delta > max_[varc])
          skip = true;
      }
      // std::cout << "skip = " << skip << std::endl;
    } else {
      // get the variable:
      const edm::View<reco::Candidate> *var0;
      edm::Handle<edm::View<reco::Candidate>> var0_hnd;
      bool isVar0 = iEvent.getByLabel(collections_[varc], var0_hnd);
      if (!isVar0) {
        std::cout << "Warning : no " << collections_[varc] << " in input !" << std::endl;
        return false;
      }
      var0 = var0_hnd.product();
      const reco::Candidate *var00 = &(*var0)[0];
      // std::cout << "FL: var00.pt = " << var00->et() << std::endl;
      double coll_var;
      if (variables_[varc] == "et")
        coll_var = var00->et();
      else if (variables_[varc] == "phi")
        coll_var = var00->phi();
      else if (variables_[varc] == "eta")
        coll_var = var00->eta();
      else if (variables_[varc] == "DeltaMEXcut") {
        const edm::View<reco::Candidate> *truevar0;
        edm::Handle<edm::View<reco::Candidate>> truevar0_hnd;
        bool istrueVar0 = iEvent.getByLabel(TrueMET_, truevar0_hnd);
        if (!istrueVar0) {
          std::cout << "Warning : no " << TrueMET_ << " in input !" << std::endl;
          return false;
        }
        truevar0 = truevar0_hnd.product();
        const reco::Candidate *truevar00 = &(*truevar0)[0];

        const double DeltaMEX = var00->px() - truevar00->px();
        const double DeltaMEY = var00->py() - truevar00->py();
        const double cutvalc = sqrt(DeltaMEX * DeltaMEX + DeltaMEY * DeltaMEY);
        const reco::MET *met = static_cast<const reco::MET *>(truevar00);
        const double SETc = met->sumEt();
        // std::cout << "FL: SETc = " << SETc << std::endl;
        const double sigmac = sigma_a_ + sigma_b_ * sqrt(SETc) + sigma_c_ * SETc;
        if (cutvalc > DeltaMEXsigma_ * sigmac) {
          if (verbose_) {
            std::cout << "DeltaMET = " << var00->et() - truevar00->et() << std::endl;
            std::cout << "trueSET = " << SETc << std::endl;
            std::cout << "pfMET = " << var00->et() << std::endl;
            std::cout << "trueMET = " << truevar00->et() << std::endl;
            std::cout << "DeltaMEX = " << DeltaMEX << std::endl;
            std::cout << "DeltaMEY = " << DeltaMEY << std::endl;
            std::cout << "cutvalc = " << cutvalc << std::endl;
            std::cout << "sigmac = " << sigmac << std::endl;
            std::cout << "cutvalc/sigmac = " << cutvalc / sigmac << std::endl;
          }
          return true;
        } else {
          if (verbose_ && (var00->et() - truevar00->et()) > 300.0) {
            std::cout << "EVENT NOT KEPT:" << std::endl;
            std::cout << "DeltaMET = " << var00->et() - truevar00->et() << std::endl;
            std::cout << "SETc = " << SETc << std::endl;
            std::cout << "pfMET = " << var00->et() << std::endl;
            std::cout << "trueMET = " << truevar00->et() << std::endl;
            std::cout << "DeltaMEX = " << DeltaMEX << std::endl;
            std::cout << "DeltaMEY = " << DeltaMEY << std::endl;
            std::cout << "cutvalc = " << cutvalc << std::endl;
            std::cout << "sigmac = " << sigmac << std::endl;
            std::cout << "cutvalc/sigmac = " << cutvalc / sigmac << std::endl;
          }
          return false;
        }
      } else {
        std::cout << "Error: PFMETFilter: variable unknown: " << variables_[varc] << std::endl;
        return true;
      }
      // std::cout << "FL: coll_var[" << variables_[varc] << "] = " << coll_var
      // << std::endl; std::cout << "FL: max_[varc] = " << max_[varc] <<
      // std::endl; std::cout << "FL: min_[varc] = " << min_[varc] << std::endl;

      // cuts
      if (doMin_[varc] && doMax_[varc] && max_[varc] < min_[varc]) {
        if (coll_var > max_[varc] && coll_var < min_[varc])
          skip = true;
      } else {
        if (doMin_[varc] && coll_var < min_[varc])
          skip = true;
        if (doMax_[varc] && coll_var > max_[varc])
          skip = true;
      }
      // std::cout << "skip = " << skip << std::endl;
    }
  }
  // std::cout << "final skip = " << skip << std::endl;
  return !skip;
}

void PFMETFilter::endJob() {
  // std::cout << "FL: endJob" << std::endl;
}
