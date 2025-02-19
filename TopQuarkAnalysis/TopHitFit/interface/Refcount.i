//
// $Id: Refcount.i,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: private/Refcount.i
// Purpose: Inline implementations for Refcount.
// Created: Aug 2000, sss, from the version that used to be in d0om.
//
// CMSSW File      : interface/Refcount.i
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Refcount.i

    @brief Inline source file for a base class for simple
    reference-counted object.  See the documentation for the header file
    Refcount.h for details.

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

namespace hitfit {


inline
Refcount::Refcount ()
//
// Purpose: Constructor.  Initializes the reference count to 0.
//
  : _refcount (0)
{
}


inline
Refcount::~Refcount ()
//
// Purpose: Destructor.  It is an error to try to delete the object if the
//          refcount is not 0.
{
  assert (_refcount == 0);
}


inline
void Refcount::incref () const
//
// Purpose: Increment the refcount.
// 
{
  ++_refcount;
}


inline
void Refcount::decref () const
//
// Purpose: Decrement the refcount; delete the object when it reaches zero.
//
{
  // Actually, we do the test first, to avoid triing the assertion
  // in the destructor.
  if (_refcount == 1) {
    _refcount = 0;
    delete this;
  }
  else {
    assert (_refcount > 0);
    --_refcount;
  }
}


inline
bool Refcount::decref_will_delete () const
//
// Purpose: Will the next decref() delete the object?
//
// Returns:
//   True if the next decref() will delete the object.
//
{
  return (_refcount == 1);
}


inline
bool Refcount::unowned () const
//
// Purpose: True if incref() has never been called, or if the object
//          is being deleted.
//
// Returns:
//   True if incref() has never been called, or if the object
//   is being deleted.
//
{
  return (_refcount == 0);
}


} // namespace hitfit
