#ifndef MixCollectionValidation_H
#define MixCollectionValidation_H

// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

//DQM services for histogram
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <string>

//
// class declaration
//

class MixCollectionValidation : public DQMEDAnalyzer {
public:
  explicit MixCollectionValidation(const edm::ParameterSet&);
  ~MixCollectionValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  
  edm::ParameterSet mixObjextsSet_;

  template<class T1, class T2> void fillMultiplicity(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillGenParticleMulti(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillSimHitTime(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  template<class T1, class T2> void fillCaloHitTime(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_);

  /* N.B. I see vector<InputTag> as private members of this class, but
     in the corresponding C++ only the first element, if present, is
     used to get products from the event. Hence I did implement a
     single Token for each kind of objects, not a vector of
     Tokens. For all but PSimHitTags_ and PCaloHitTags_, which have a
     corresponding vector of Tokens. */

  edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct> > crossingFrame_Hep_Token_;
  edm::EDGetTokenT<CrossingFrame<SimTrack> > crossingFrame_SimTr_Token_;
  edm::EDGetTokenT<CrossingFrame<SimVertex> > crossingFrame_SimVtx_Token_;
  std::vector< edm::EDGetTokenT<CrossingFrame<PSimHit> > > crossingFrame_PSimHit_Tokens_;
  std::vector< edm::EDGetTokenT<CrossingFrame<PCaloHit> > > crossingFrame_PCaloHit_Tokens_;

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

  std::vector<edm::InputTag> HepMCProductTags_;
  std::vector<edm::InputTag> SimTrackTags_;
  std::vector<edm::InputTag> SimVertexTags_;
  std::vector<edm::InputTag> PSimHitTags_;
  std::vector<edm::InputTag> PCaloHitTags_;

  int nbin_;

};

#endif
