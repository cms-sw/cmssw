#include <cppunit/extensions/HelperMacros.h>
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include "FWCore/Utilities/interface/EDMException.h" 
#include <iostream>

class testPdtEntry : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testPdtEntry);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION( testPdtEntry );

void testPdtEntry::checkAll() {
  using namespace std;
  PdtEntry e( 13 );
  CPPUNIT_ASSERT_THROW( e.name(), cms::Exception ); 
  PdtEntry n( "mu-" );
  CPPUNIT_ASSERT_THROW( n.pdgId(), cms::Exception ); 

  edm::ParameterSet cfg;
  cfg.addParameter<int>( "pdt1", 13 );
  cfg.addParameter<string>( "pdt2", "mu-" );

  int pdt1 = cfg.getParameter<int>( "pdt1" );
  string pdt2 = cfg.getParameter<string>( "pdt2" );
  PdtEntry e1  = cfg.getParameter<PdtEntry>( "pdt1" );
  PdtEntry e2  = cfg.getParameter<PdtEntry>( "pdt2" );
  CPPUNIT_ASSERT( pdt1 == e1.pdgId() );
  CPPUNIT_ASSERT( pdt2 == e2.name() );
}
