////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <memory>
#include <vector>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
template<typename T1>
class ZllArbitrator : public edm::EDProducer
{
public:
  // construction/destruction
  ZllArbitrator(const edm::ParameterSet& iConfig);
  virtual ~ZllArbitrator();
  
  // member functions
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup);
  void endJob();

private:  
  // member data
  edm::InputTag              srcZCand_;  
};



////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T1>
ZllArbitrator<T1>::ZllArbitrator(const edm::ParameterSet& iConfig)
  : srcZCand_(iConfig.getParameter<edm::InputTag>("ZCandidateCollection"))
{
  produces<std::vector<T1> >();
}


//______________________________________________________________________________
template<typename T1>
ZllArbitrator<T1>::~ZllArbitrator(){}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T1>
void ZllArbitrator<T1>::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  std::stringstream ss ;
  
  std::auto_ptr<std::vector<T1> > TheBestZ(new std::vector<T1 >);
  
  edm::Handle< std::vector<T1> > ZCandidatesHandle;
  iEvent.getByLabel(srcZCand_,ZCandidatesHandle);
  
  if( ZCandidatesHandle->size() == 0 ) 
  {
    iEvent.put(TheBestZ);
    return ;
  }
  
  double ZmassPDG  = 91.18;
  double DeltaMass = 10000;
  
  typename std::vector<T1>::const_iterator ZCandIt   ;
  typename std::vector<T1>::const_iterator bestZCand ;

  for (ZCandIt = ZCandidatesHandle->begin(); ZCandIt != ZCandidatesHandle->end(); ++ZCandIt) {

	if( fabs(ZCandIt->mass()-ZmassPDG) < DeltaMass ){
	  DeltaMass = fabs(ZCandIt->mass()-ZmassPDG) ;
	  bestZCand = ZCandIt; 
	}
  }

  TheBestZ->push_back( *bestZCand );  
  iEvent.put(TheBestZ);
  
}

template<typename T1>
void ZllArbitrator<T1>::endJob()
{
}

typedef ZllArbitrator<reco::CompositeCandidate>      BestMassZArbitrationProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BestMassZArbitrationProducer);
