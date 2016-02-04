#ifndef Notification_SimActivityRegistryEnroller_h
#define Notification_SimActivityRegistryEnroller_h
// -*- C++ -*-
//
// Package:     Notification
// Class  :     SimActivityRegistryEnroller
// 
/**\class SimActivityRegistryEnroller SimActivityRegistryEnroller.h SimG4Core/Notification/interface/SimActivityRegistryEnroller.h

 Description: Based on what classes an object inherts, enrolls that object with the proper signal

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 13 15:08:12 EST 2005
// $Id: SimActivityRegistryEnroller.h,v 1.4 2005/11/24 19:59:52 chrjones Exp $
//

// system include files
#include "boost/mpl/pop_back.hpp"
#include "boost/mpl/begin_end.hpp"
#include "boost/mpl/copy_if.hpp"
#include "boost/mpl/deref.hpp"
#include "boost/mpl/prior.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/mpl/eval_if.hpp"
#include "boost/mpl/empty.hpp"

// user include files
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

// forward declarations
namespace enroller_helper {
  template <class T>
    struct Enrollment {
      static void enroll(SimActivityRegistry& iReg, Observer<const T*>* iObs){
	iReg.connect(iObs);
      }
      static void enroll(SimActivityRegistry&, void*) {}
    };
   
  //this class is used to terminate our recursion
  template <class T>
    struct LastEnrollerHelper {
      static void enroll(SimActivityRegistry&, T*) {
      }
    };

  template< class T, class TVector>
    struct EnrollerHelper  {
      typedef typename boost::mpl::pop_back<TVector>::type RemainingVector;
      static void enroll(SimActivityRegistry& iReg, T* iT) {
	//Try to enroll the object if it inherits from the class at the 
	// end of TVector
	Enrollment< typename boost::mpl::deref< typename boost::mpl::prior< typename boost::mpl::end< TVector >::type >::type >::type >::enroll(iReg, iT );
	
	//If TVector is not at its end, call EnrollerHelper with a vector
	// that had our last type 'popped off' the end
	typedef typename boost::mpl::eval_if<boost::mpl::empty<TVector>,
	  boost::mpl::identity<LastEnrollerHelper<T> >,
	  boost::mpl::identity<EnrollerHelper<T, typename boost::mpl::pop_back< TVector >::type > >
	  >::type NextEnroller;
	NextEnroller::enroll(iReg,iT);
      }
    };
  
}

class SimActivityRegistryEnroller
{
  
 public:
  SimActivityRegistryEnroller() {}
      //virtual ~SimActivityRegistryEnroller();
      typedef boost::mpl::vector<BeginOfJob,DDDWorld,BeginOfRun,BeginOfEvent,BeginOfTrack,BeginOfStep,G4Step,EndOfTrack,EndOfEvent,EndOfRun> Signals;
   
      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      template<class T>
      static void enroll(SimActivityRegistry& iReg, T* iObj){
         enroller_helper::EnrollerHelper<T,Signals>::enroll(iReg,iObj);
      }
      // ---------- member functions ---------------------------

   private:
      SimActivityRegistryEnroller(const SimActivityRegistryEnroller&); // stop default

      const SimActivityRegistryEnroller& operator=(const SimActivityRegistryEnroller&); // stop default

      // ---------- member data --------------------------------

};


#endif
