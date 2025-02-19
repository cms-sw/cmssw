//
// $Id: Refcount.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: Refcount.h
// Purpose: A base class for a simple reference-counted object.
// Created: Aug 2000, sss, from the version that used to be in d0om.
//
// To make a reference counted type, derive from class d0_util::Refcount.
// When the object is created, its refcount is initially set to 0;
// the first action taken by the creator should be to call incref()
// to bump the refcount to 1.  Thereafter, the refcount may be incremented
// by incref() and decremented by decref().  If the reference count reaches
// 0, the object calls delete on itself.
//
// If the object is deleted explicitly, the reference count must be 0 or 1.
// Otherwise, an assertion violation will be reported.
//
// CMSSW File      : interface/Refcount.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Refcount.h

    @brief A base class for simple reference-counted object.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    Aug 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent for the original author (Scott Snyder).

 */


#ifndef D0_UTIL_REFCOUNT_H
#define D0_UTIL_REFCOUNT_H

#include<stdexcept>
#include<sstream>
#include<cassert>
namespace hitfit {


/**
    @brief Simple reference-counted object.

    To make a reference-counted type, derive from this class.
    When the object is first created, its refcount is initialy set to 0.
    The first action taken by the creator should be to call incref()
    to increase the refcount to 1.  Thereafter, the refcount may be incremented
    by incref() and decremented by decref().  If the reference count reaches
    0, the object calls delete on itself.

    If the object is deleted explicitly, the reference count must be 0 or 1.
    Otherwise, an assertion violation will be reported.
 */
class Refcount
//
// Purpose: Simple reference-counted object.
//
{
public:
  // Constructor, destructor.

  /**
     @brief Constructor, initialize the reference count to 0.
   */
  Refcount ();

  /**
     @brief Destructor, it is an error to try to delete an object if the
     reference count is not 0.
   */
  virtual ~Refcount ();

  // Increment and decrement reference count.
  /**
     @brief Increment the reference count.
   */
  void incref () const;

  /**
     @brief Decrease the reference count.
   */
  void decref () const;

  // True if calling decref() will delete the object.
  /**
     @brief Return <b>true</b> if calling decref() will delete the object.
     Otherwise return <b>FALSE</b>.
   */
  bool decref_will_delete () const;

  // True if incref() has never been called, or if the object is being
  // deleted.
  /**
     @brief Return <b>true</b> if incref() has never been called or if the
     object is being deleted.  Otherwise return <b>FALSE</b>.
   */
  bool unowned () const;


protected:
  // Reset the refcount to zero.
  // This should only be used in the context of a dtor of a derived
  // class that wants to throw an exception.
  // It may also be used to implement a `release' function.
  /**
     @brief Reset the reference count to zero. This should only be used in the
     context of a destructor of a derived class that wants to throw an
     exception.  It may also be used to implement a "release" function.
   */
  void nuke_refcount ();


private:
  // The reference count itself.
  /**
     The reference count itself.
   */
  mutable unsigned _refcount;
};


} // namespace hitfit


#include "TopQuarkAnalysis/TopHitFit/interface/Refcount.i"


#endif // not D0_UTIL_REFCOUNT_H
