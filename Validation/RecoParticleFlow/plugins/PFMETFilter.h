#ifndef PFMETFILTER_H
#define PFMETFILTER_H

// author: Florent Lacroix (UIC)
// date: 07/14/2009

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

class PFMETFilter : public edm::EDFilter {
public:
  explicit PFMETFilter(const edm::ParameterSet &);
  ~PFMETFilter() override;

  bool filter(edm::Event &, const edm::EventSetup &) override;
  void beginJob() override;
  void endJob() override;
  bool checkInput();

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

#endif  // PFMETFILTER_H
