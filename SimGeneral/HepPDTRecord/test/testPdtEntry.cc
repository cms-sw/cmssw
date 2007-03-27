#include <cppunit/extensions/HelperMacros.h>
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include "FWCore/Utilities/interface/EDMException.h" 
#include <iostream>
using namespace std;
using namespace edm;

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

  {
    ParameterSet cfg;
    cfg.addParameter<int>( "pdt1", 13 );
    cfg.addParameter<string>( "pdt2", "mu-" );
    int pdt1 = cfg.getParameter<int>( "pdt1" );
    string pdt2 = cfg.getParameter<string>( "pdt2" );
    PdtEntry e1  = cfg.getParameter<PdtEntry>( "pdt1" );
    PdtEntry e2  = cfg.getParameter<PdtEntry>( "pdt2" );
    CPPUNIT_ASSERT( pdt1 == e1.pdgId() );
    CPPUNIT_ASSERT( pdt2 == e2.name() );
    vector<string> p = cfg.getParameterNamesForType<PdtEntry>();
    CPPUNIT_ASSERT( p.size() == 2 );
  }
  {
    ParameterSet cfg;
    cfg.addUntrackedParameter<int>( "pdt1", 13 );
    cfg.addUntrackedParameter<string>( "pdt2", "mu-" );
    int pdt1 = cfg.getUntrackedParameter<int>( "pdt1" );
    string pdt2 = cfg.getUntrackedParameter<string>( "pdt2" );
    PdtEntry e1  = cfg.getUntrackedParameter<PdtEntry>( "pdt1" );
    PdtEntry e2  = cfg.getUntrackedParameter<PdtEntry>( "pdt2" );
    CPPUNIT_ASSERT( pdt1 == e1.pdgId() );
    CPPUNIT_ASSERT( pdt2 == e2.name() );
  }
  {
    ParameterSet cfg;
    int pdt1 = cfg.getUntrackedParameter<int>( "pdt1", 13 );
    string pdt2 = cfg.getUntrackedParameter<string>( "pdt2", "mu-" );
    PdtEntry e1  = cfg.getUntrackedParameter<PdtEntry>( "pdt1", PdtEntry( 13 ) );
    PdtEntry e2  = cfg.getUntrackedParameter<PdtEntry>( "pdt2", PdtEntry( "mu-" ) );
    CPPUNIT_ASSERT( pdt1 == e1.pdgId() );
    CPPUNIT_ASSERT( pdt2 == e2.name() );
  }
  {
    ParameterSet cfg;
    cfg.addParameter<vector<int> >( "pdt1", vector<int>( 1, 13 ) );
    cfg.addParameter<vector<string> >( "pdt2", vector<string>( 1, "mu-" ) );
    vector<int> pdt1 = cfg.getParameter<vector<int> >( "pdt1" );
    vector<string> pdt2 = cfg.getParameter<vector<string> >( "pdt2" );
    vector<PdtEntry> e1  = cfg.getParameter<vector<PdtEntry> >( "pdt1" );
    vector<PdtEntry> e2  = cfg.getParameter<vector<PdtEntry> >( "pdt2" );
    CPPUNIT_ASSERT( e1.size() == pdt1.size() );
    CPPUNIT_ASSERT( e2.size() == pdt2.size() );
    CPPUNIT_ASSERT( pdt1.front() == e1.front().pdgId() );
    CPPUNIT_ASSERT( pdt2.front() == e2.front().name() );
  }
  {
    ParameterSet cfg;
    cfg.addUntrackedParameter<vector<int> >( "pdt1", vector<int>( 1, 13 ) );
    cfg.addUntrackedParameter<vector<string> >( "pdt2", vector<string>( 1, "mu-" ) );
    vector<int> pdt1 = cfg.getUntrackedParameter<vector<int> >( "pdt1" );
    vector<string> pdt2 = cfg.getUntrackedParameter<vector<string> >( "pdt2" );
    vector<PdtEntry> e1  = cfg.getUntrackedParameter<vector<PdtEntry> >( "pdt1" );
    vector<PdtEntry> e2  = cfg.getUntrackedParameter<vector<PdtEntry> >( "pdt2" );
    CPPUNIT_ASSERT( e1.size() == pdt1.size() );
    CPPUNIT_ASSERT( e2.size() == pdt2.size() );
    CPPUNIT_ASSERT( pdt1.front() == e1.front().pdgId() );
    CPPUNIT_ASSERT( pdt2.front() == e2.front().name() );
  }
  {
    ParameterSet cfg;
    vector<int> pdt1 = cfg.getUntrackedParameter<vector<int> >( "pdt1", vector<int>( 1, 13 ) );
    vector<string> pdt2 = cfg.getUntrackedParameter<vector<string> >( "pdt2", vector<string>( 1, "mu-" ) );
    vector<PdtEntry> e1  = cfg.getUntrackedParameter<vector<PdtEntry> >( "pdt1", vector<PdtEntry>( 1, PdtEntry( 13 ) ) );
    vector<PdtEntry> e2  = cfg.getUntrackedParameter<vector<PdtEntry> >( "pdt2", vector<PdtEntry>( 1, PdtEntry( "mu-" ) ) );
    CPPUNIT_ASSERT( e1.size() == pdt1.size() );
    CPPUNIT_ASSERT( e2.size() == pdt2.size() );
    CPPUNIT_ASSERT( pdt1.front() == e1.front().pdgId() );
    CPPUNIT_ASSERT( pdt2.front() == e2.front().name() );
  }
}
