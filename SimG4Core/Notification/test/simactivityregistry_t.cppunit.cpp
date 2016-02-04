/*
 *  serviceregistry_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 9/7/05.
 *
 */

//need to open a 'back door' to be able to setup the SimActivityRegistry
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include <cppunit/extensions/HelperMacros.h>

class testSimActivityRegistry: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testSimActivityRegistry);
   
   CPPUNIT_TEST(signalTest);
   CPPUNIT_TEST(signalForwardingTest);
   CPPUNIT_TEST(enrollerTest);
   
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void signalTest();
   void signalForwardingTest();
   void enrollerTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testSimActivityRegistry);

namespace {
   template<class T> struct MyObserver : public Observer<const T*> {
      mutable bool saw_;
      MyObserver() : saw_(false) {}     
      void update(const T*) {
         saw_=true;
      }
   };
}

#define TEST(signal, SIGNAL)    MyObserver<SIGNAL> watch ## SIGNAL; \
registry.connect(&watch ## SIGNAL);\
const SIGNAL* p ## SIGNAL=0; \
registry. signal ## Signal_(p ## SIGNAL); \
CPPUNIT_ASSERT(watch ## SIGNAL .saw_);

void
testSimActivityRegistry::signalTest()
{
   SimActivityRegistry registry;

   for(int i=0; i<1000; i++) {
   TEST(beginOfRun,BeginOfRun);
   TEST(beginOfJob,BeginOfJob);
   TEST(beginOfEvent,BeginOfEvent);
   TEST(beginOfTrack,BeginOfTrack);
   TEST(dddWorld,DDDWorld);
   TEST(g4Step,G4Step);

   TEST(endOfRun,EndOfRun);
   TEST(endOfEvent,EndOfEvent);
   TEST(endOfTrack,EndOfTrack);   
   }
}

#define TESTF(signal, SIGNAL)    MyObserver<SIGNAL> watch ## SIGNAL; \
registry2.connect(&watch ## SIGNAL);\
const SIGNAL* p ## SIGNAL=0; \
registry. signal ## Signal_(p ## SIGNAL); \
CPPUNIT_ASSERT(watch ## SIGNAL .saw_);

void
testSimActivityRegistry::signalForwardingTest()
{
   SimActivityRegistry registry;
   SimActivityRegistry registry2;
   registry.connect(registry2);

   for(int i=0; i<1000; i++) {
   TESTF(beginOfRun,BeginOfRun);
   TESTF(beginOfJob,BeginOfJob);
   TESTF(beginOfEvent,BeginOfEvent);
   TESTF(beginOfTrack,BeginOfTrack);
   TESTF(dddWorld,DDDWorld);
   TESTF(g4Step,G4Step);

   TESTF(endOfRun,EndOfRun);
   TESTF(endOfEvent,EndOfEvent);
   TESTF(endOfTrack,EndOfTrack);   
   }
}

namespace {
   
   template<class T> struct Counting : public Observer<const T*> {
      mutable int& count_;
      Counting(int& iCount) : count_(iCount) {}     
      void update(const T*) {
         ++count_;
      }
   };
   struct NoSignal { };
   
   struct OneSignal : public Counting<BeginOfEvent> { 
      OneSignal(int& iCount):Counting<BeginOfEvent>(iCount) {} 
   };
   
   struct TwoSignals : public Counting<BeginOfEvent>, public Counting<EndOfEvent> {
      TwoSignals(int& iCount):Counting<BeginOfEvent>(iCount),Counting<EndOfEvent>(iCount) {} 
   };

}

#define TESTREG(signal, SIGNAL)    int count ## SIGNAL = 0; \
Counting<SIGNAL> watch ## SIGNAL (count ## SIGNAL ); \
enroller.enroll(registry, &watch ## SIGNAL);\
const SIGNAL* p ## SIGNAL=0; \
registry. signal ## Signal_(p ## SIGNAL); \
CPPUNIT_ASSERT(1==watch ## SIGNAL .count_);

void
testSimActivityRegistry::enrollerTest()
{
   SimActivityRegistry registry;

   NoSignal noSignal;
   SimActivityRegistryEnroller enroller;
   enroller.enroll(registry, &noSignal);
   
   int int1Signal=0;
   OneSignal oneSignal(int1Signal);
   enroller.enroll(registry, &oneSignal);

   int int2Signals=0;
   TwoSignals twoSignals(int2Signals);
   enroller.enroll(registry, &twoSignals);

   const BeginOfEvent* pBegin=0;
   registry.beginOfEventSignal_(pBegin);
   
   const EndOfEvent* pEnd=0;
   registry.endOfEventSignal_(pEnd);
   
   CPPUNIT_ASSERT(1==int1Signal);
   CPPUNIT_ASSERT(2==int2Signals);

   TESTREG(beginOfRun,BeginOfRun);
   TESTREG(beginOfJob,BeginOfJob);
   TESTREG(beginOfEvent,BeginOfEvent);
   TESTREG(beginOfTrack,BeginOfTrack);
   TESTREG(dddWorld,DDDWorld);
   TESTREG(g4Step,G4Step);
   
   TESTREG(endOfRun,EndOfRun);
   TESTREG(endOfEvent,EndOfEvent);
   TESTREG(endOfTrack,EndOfTrack);   
   
}
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
