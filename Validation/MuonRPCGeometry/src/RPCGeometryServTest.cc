/**
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

using namespace std;

class RPCGeometryServTest : public edm::EDAnalyzer {

 public: 
  RPCGeometryServTest( const edm::ParameterSet& pset);

  ~RPCGeometryServTest();

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
 
  const std::string& myName() { return myName_;}

 private: 

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
};

RPCGeometryServTest::RPCGeometryServTest( const edm::ParameterSet& iConfig )
  : dashedLineWidth_(104), dashedLine_( std::string(dashedLineWidth_, '-') ), 
    myName_( "RPCGeometryServTest" ) 
{ 
  std::cout <<"======================== Opening output file"<< std::endl;
}


RPCGeometryServTest::~RPCGeometryServTest() 
{
}

void
RPCGeometryServTest::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  edm::ESHandle<RPCGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get( pDD );     

  std::cout << myName() << ": Analyzer..." << std::endl;
  std::cout << "start " << dashedLine_ << std::endl;


  std::cout << " Geometry node for RPCGeom is  " << &(*pDD) << std::endl;   
  cout << " I have "<<pDD->detTypes().size()    << " detTypes" << endl;
  cout << " I have "<<pDD->detUnits().size()    << " detUnits" << endl;
  cout << " I have "<<pDD->dets().size()        << " dets" << endl;
  cout << " I have "<<pDD->rolls().size()       << " rolls" << endl;
  cout << " I have "<<pDD->chambers().size()    << " chambers" << endl;

  std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
  std::cout << "iter " << dashedLine_ << std::endl;

  

  int iRPCCHcount = 0;
  LocalPoint a(0.,0.,0.);
  for(TrackingGeometry::DetContainer::const_iterator it = 
	pDD->dets().begin(); it != pDD->dets().end(); it++){
    
    //----------------------- RPCCHAMBER TEST ---------------------------

    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
      ++iRPCCHcount;
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      RPCDetId detId=ch->id();
      
      std::vector< const RPCRoll*> rollsRaf = (ch->rolls());
      for(std::vector<const RPCRoll*>::iterator r = rollsRaf.begin();
	  r != rollsRaf.end(); ++r){
	std::cout << dashedLine_ << " NEW ROLL" << std::endl;
	std::cout<<"Region = "<<(*r)->id().region()
		 <<"  Ring = "<<(*r)->id().ring()
		 <<"  Station = "<<(*r)->id().station()
		 <<"  Sector = "<<(*r)->id().sector()
		 <<"  Layer = "<<(*r)->id().layer()
		 <<"  Subsector = "<<(*r)->id().subsector()
		 <<"  Roll = "<<(*r)->id().roll()<<std::endl;
	RPCGeomServ s( (*r)->id());
	GlobalPoint g= (*r)->toGlobal(a);
	std::cout <<s.name()<<" eta partition "<<s.eta_partition()<< " nroll="<<ch->nrolls()
		  <<" z="<<g.z()<<" phi="<<g.phi()<<" R="<<g.perp()
		  <<std::endl;
      }
    }
  }
  std::cout <<std::endl;
  std::cout << dashedLine_ << " end" << std::endl;
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(RPCGeometryServTest);
