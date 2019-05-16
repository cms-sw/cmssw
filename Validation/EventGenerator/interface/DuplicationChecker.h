#ifndef DUPLICATIONCHECKER_H
#define DUPLICATIONCHECKER_H

/*class DuplicationChecker
 *  
 *  Class to monitor duplication of events
 *
 *
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <vector>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "Validation/EventGenerator/interface/WeightManager.h"

class DuplicationChecker : public DQMEDAnalyzer {
  
 public:

  typedef std::multimap<double,edm::EventID> associationMap;
  typedef std::vector<associationMap::iterator> itemList;

  explicit DuplicationChecker(const edm::ParameterSet&);
  ~DuplicationChecker() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;

  void findValuesAssociatedWithKey(associationMap &mMap, double &key, itemList &theObjects);  
  
 private:
  WeightManager wmanager_;
  
  edm::InputTag generatedCollection_;
  edm::InputTag lheEventProduct_;
  
  bool searchForLHE_;

  associationMap xBjorkenHistory;
	
  MonitorElement* xBjorkenME;

  edm::EDGetTokenT<LHEEventProduct> lheEventProductToken_;
  edm::EDGetTokenT<edm::HepMCProduct> generatedCollectionToken_;

};

#endif
