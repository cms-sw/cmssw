//
// $Id: Refcount.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: Refcount.cc
// Purpose: Reference count implementation.
// Created: Aug 2000, sss, from the version that used to be in d0om.
//
// CMSSW File      : src/Refcount.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Refcount.cc

    @brief A base class for simple reference-counted object.  See the
    documentation for the header file Refcount.h for details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/Refcount.h"


namespace hitfit {


void Refcount::nuke_refcount ()
//
// Purpose: Reset the refcount to zero.
//          This should only be used in the context of a dtor of a derived
//          class which wants to throw an exception.
//
{
  _refcount = 0;
}


} // namespace hitfit
