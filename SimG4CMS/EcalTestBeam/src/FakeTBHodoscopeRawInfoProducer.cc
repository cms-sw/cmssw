/*
 * \file FakeTBHodoscopeRawInfoProducer.cc
 *
 * $Id: FakeTBHodoscopeRawInfoProducer.cc,v 1.3 2006/10/26 08:01:06 fabiocos Exp $
 *
 */

#include "SimG4CMS/EcalTestBeam/interface/FakeTBHodoscopeRawInfoProducer.h"

using namespace cms;
using namespace std;


FakeTBHodoscopeRawInfoProducer::FakeTBHodoscopeRawInfoProducer(const edm::ParameterSet& ps) {
  
  produces<EcalTBHodoscopeRawInfo>();

  ecalTBInfoLabel_ = ps.getUntrackedParameter<string>("EcalTBInfoLabel","SimEcalTBG4Object");

  theTBHodoGeom_ = new EcalTBHodoscopeGeometry();

}

 
FakeTBHodoscopeRawInfoProducer::~FakeTBHodoscopeRawInfoProducer() {

  delete theTBHodoGeom_;

}

 void FakeTBHodoscopeRawInfoProducer::produce(edm::Event & event, const edm::EventSetup& eventSetup)
{
  auto_ptr<EcalTBHodoscopeRawInfo> product(new EcalTBHodoscopeRawInfo());

  // get the vertex information from the event

  edm::Handle<PEcalTBInfo> theEcalTBInfo;
  event.getByLabel(ecalTBInfoLabel_,theEcalTBInfo);

  double partXhodo = theEcalTBInfo->evXbeam();
  double partYhodo = theEcalTBInfo->evYbeam();

  LogDebug("EcalTBHodo") << "TB frame vertex (X,Y) for hodoscope simulation: \n" 
                         << "x = " << partXhodo << " y = " << partYhodo;
  
  // for each hodoscope plane determine the fibre number corresponding 
  // to the event vertex coordinates in the TB reference frame
  // plane 0/2 = x plane 1/3 = y
  
  int nPlanes = (int)theTBHodoGeom_->getNPlanes();
  product->setPlanes(nPlanes);
  
  for ( int iPlane = 0 ; iPlane < nPlanes ; ++iPlane) {
    
    float theCoord = (float)partXhodo;
    if (iPlane == 1 || iPlane == 3) theCoord = (float)partYhodo;
    
    vector<int> firedChannels = theTBHodoGeom_->getFiredFibresInPlane(theCoord, iPlane);
    unsigned int nChannels = firedChannels.size();
    
    EcalTBHodoscopePlaneRawHits planeHit(nChannels);
    for ( unsigned int i = 0 ; i < nChannels ; ++i ) {
      planeHit.addHit(firedChannels[i]);
    }
    
    product->setPlane((unsigned int)iPlane, planeHit);
    
  }

  LogDebug("EcalTBHodo") << (*product);
  
  event.put(product);
  
}
