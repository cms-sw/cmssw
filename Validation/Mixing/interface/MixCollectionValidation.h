#ifndef MixCollectionValidation_H
#define MixCollectionValidation_H

// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

//
// class declaration
//

class MixCollectionValidation : public edm::EDAnalyzer {
public:
  explicit MixCollectionValidation(const edm::ParameterSet&);
  ~MixCollectionValidation();

  void beginJob();
  void endJob();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:

  template<class T1, class T2> void fillMultiplicity(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillGenParticleMulti(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillSimHitTime(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillCaloHitTime(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  std::string outputFile_;
  int minbunch_;
  int maxbunch_;

  bool verbose_;

  MonitorElement * nrHepMCProductH_;
  MonitorElement * nrSimTrackH_;
  MonitorElement * nrSimVertexH_;

  std::map<std::string,MonitorElement *> SimHitNrmap_;
  std::map<std::string,MonitorElement *> SimHitTimemap_;

  std::map<std::string,MonitorElement *> CaloHitNrmap_;
  std::map<std::string,MonitorElement *> CaloHitTimemap_;

  DQMStore* dbe_;
  
  std::vector<std::string> names_;
  std::vector<edm::InputTag> HepMCProductTags_;
  std::vector<edm::InputTag> SimTrackTags_;
  std::vector<edm::InputTag> SimVertexTags_;
  std::vector<edm::InputTag> PSimHitTags_;
  std::vector<edm::InputTag> PCaloHitTags_;

  int nbin_;
  
};

#endif
