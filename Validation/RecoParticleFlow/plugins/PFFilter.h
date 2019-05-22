#ifndef PFFILTER_H
#define PFFILTER_H

// author: Florent Lacroix (UIC)
// date: 07/14/2009

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

class PFFilter : public edm::EDFilter {
public:
  explicit PFFilter(const edm::ParameterSet &);
  ~PFFilter() override;

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
};

#endif  // PFFILTER_H
