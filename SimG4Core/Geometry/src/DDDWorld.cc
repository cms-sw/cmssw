#include "FWCore/CoreFramework/interface/EventSetupProvider.h"
#include "FWCore/CoreFramework/interface/recordGetImplementation.icc"
 
#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include "SimG4Core/Geometry/interface/DDCompactViewXMLRetriever.h"
#include "SimG4Core/Geometry/interface/DDG4Builder.h"

#include "G4RunManagerKernel.hh"
#include "G4PVPlacement.hh"
 
using namespace edm;

DDDWorld::DDDWorld(const edm::ParameterSet & p) 
{
    EventSetupProvider provider;
  
    boost::shared_ptr<DDCompactViewXMLRetriever> 
	pRetriever(new DDCompactViewXMLRetriever(p));
    boost::shared_ptr<DataProxyProvider> pProxyProv(pRetriever);
    provider.add(pProxyProv);
  
    provider.add(boost::shared_ptr<EventSetupRecordIntervalFinder>(pRetriever));
    const EventSetup & eventsetup = 
	provider.eventSetupForInstance(Timestamp(1));
  
    std::auto_ptr<DDG4Builder> theBuilder(new DDG4Builder(eventsetup));

    G4LogicalVolume * world = theBuilder->BuildGeometry();
    G4VPhysicalVolume * pv = 
	new G4PVPlacement(0,G4ThreeVector(),world,"DDDWorld",0,false,0);
    SetAsWorld(pv);
}

DDDWorld::~DDDWorld() {}

void DDDWorld::SetAsWorld(G4VPhysicalVolume * pv)
{
    G4RunManagerKernel * kernel = G4RunManagerKernel::GetRunManagerKernel();
    if (kernel != 0) kernel->DefineWorldVolume(pv);
    std::cout << " World volume defined " << std::endl;
}

