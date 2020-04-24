#ifndef DIGI2RAW2DIGI_H
#define DIGI2RAW2DIGI_H

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include <map>

class Digi2Raw2Digi : public DQMEDAnalyzer {
public:
  explicit Digi2Raw2Digi(const edm::ParameterSet&);
  ~Digi2Raw2Digi();

  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const & );

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  template<class Digi>  void compare(const edm::Event&, const edm::EventSetup&, const edm::EDGetTokenT<edm::SortedCollection<Digi> >& tok1, const edm::EDGetTokenT<edm::SortedCollection<Digi> >& tok2);  


 private:

  edm::InputTag inputTag1_;
  edm::InputTag inputTag2_;

  edm::EDGetTokenT<edm::SortedCollection<HBHEDataFrame> > tok_hbhe1_;
  edm::EDGetTokenT<edm::SortedCollection<HBHEDataFrame> > tok_hbhe2_;
  edm::EDGetTokenT<edm::SortedCollection<HODataFrame> > tok_ho1_;
  edm::EDGetTokenT<edm::SortedCollection<HODataFrame> > tok_ho2_;
  edm::EDGetTokenT<edm::SortedCollection<HFDataFrame> > tok_hf1_;
  edm::EDGetTokenT<edm::SortedCollection<HFDataFrame> > tok_hf2_;
  edm::EDGetTokenT<edm::SortedCollection<ZDCDataFrame> > tok_zdc1_;
  edm::EDGetTokenT<edm::SortedCollection<ZDCDataFrame> > tok_zdc2_;

  std::string outputFile_;

  MonitorElement* meStatus;

  int unsuppressed; // flag for ZSC unsuppressedDigis picking up

};

#endif
