/*class DuplicationChecker
 *  
 *  Class to monitor duplication of events
 *
 *  $Date: 2012/10/16 14:49:03 $
 *  $Revision: 1.4 $
 *
 */
 
#include "Validation/EventGenerator/interface/DuplicationChecker.h"

using namespace edm;

DuplicationChecker::DuplicationChecker(const edm::ParameterSet& iPSet):
  _wmanager(iPSet),
  generatedCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection")),
  searchForLHE_(iPSet.getParameter<bool>("searchForLHE"))
{ 
  if (searchForLHE_) {
    lheEventProduct_ = iPSet.getParameter<edm::InputTag>("lheEventProduct");
  }
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

  xBjorkenHistory.clear();
}

DuplicationChecker::~DuplicationChecker() 
{
  xBjorkenHistory.clear();
}

void DuplicationChecker::beginJob()
{
  if(dbe){
	///Setting the DQM top directories
	dbe->setCurrentFolder("Generator/DuplicationCheck");
	
	///Booking the ME's
	xBjorkenME = dbe->book1D("xBjorkenME", "x Bjorken ratio", 1000000, 0., 1.);
  }
}

void DuplicationChecker::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
    
  double bjorken = 0;
 
  double weight = 1.;

  if (searchForLHE_) {

    Handle<LHEEventProduct> evt;
    iEvent.getByLabel(lheEventProduct_, evt);

    const lhef::HEPEUP hepeup_ = evt->hepeup();

    const std::vector<lhef::HEPEUP::FiveVector> pup_ = hepeup_.PUP;

    double pz1=(pup_[0])[3];
    double pz2=(pup_[1])[3];
    bjorken+=(pz1/(pz1+pz2));
  }
  else {
    //change teh weight in this case
    weight = _wmanager.weight(iEvent);

    edm::Handle<HepMCProduct> evt;
    iEvent.getByLabel(generatedCollection_, evt);

    const HepMC::PdfInfo *pdf = evt->GetEvent()->pdf_info();    
    if(pdf){
      bjorken = ((pdf->x1())/((pdf->x1())+(pdf->x2())));
    }

  }

  xBjorkenHistory.insert(std::pair<double,edm::EventID>(bjorken,iEvent.id()));

  xBjorkenME->Fill(bjorken,weight);

}//analyze

void DuplicationChecker::findValuesAssociatedWithKey(associationMap &mMap, double &key, itemList &theObjects)
{
  associationMap::iterator itr;
  associationMap::iterator lastElement;
        
  theObjects.clear();

  // locate an iterator to the first pair object associated with key
  itr = mMap.find(key);
  if (itr == mMap.end())
    return; // no elements associated with key, so return immediately

  // get an iterator to the element that is one past the last element associated with key
  lastElement = mMap.upper_bound(key);

  // for each element in the sequence [itr, lastElement)
  for ( ; itr != lastElement; ++itr)
    theObjects.push_back(itr);
}  

void DuplicationChecker::endJob()
{

  itemList theObjects;
  theObjects.reserve(10);

  for (associationMap::iterator it = xBjorkenHistory.begin(); it != xBjorkenHistory.end(); it++) {
    double theKey = (*it).first;

    findValuesAssociatedWithKey(xBjorkenHistory, theKey, theObjects);

    if (theObjects.size() > 1) {
      edm::LogWarning("DuplicatedEventFound") << "Duplicated events found with xBjorken = " << std::fixed << std::setw(16) << std::setprecision(14) << theKey; 
      for (unsigned int i = 0; i < theObjects.size(); i++) {
        edm::LogPrint("DuplicatedEventList") << "Event = " << (*theObjects[i]).second;
      }
    }

    theObjects.clear();
 
  }

}
