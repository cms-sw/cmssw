//
//
// File: src/Fit_Result_Vec.cc
// Purpose: Hold a set of Fit_Result structures.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Fit_Result_Vec.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file Fit_Result_Vec.cc

    @brief Hold a list of pointers to a set of
    Fit_Result objects, resulting from different jet permutation with
    some consistent selection.  See the documentation for the header file
    Fit_Result_Vec.h for details.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    Jul 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).

 */

#include "TopQuarkAnalysis/TopHitFit/interface/Fit_Result_Vec.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Fit_Result.h"
#include <cassert>
#include <ostream>
#include <algorithm>


using std::ostream;
using std::vector;
using std::lower_bound;


namespace hitfit {


Fit_Result_Vec::Fit_Result_Vec (std::vector<std::shared_ptr<Fit_Result>>::size_type max_len)
//
// Purpose Constructor.
//
// Inputs:
//   max_len -     The maximum length of the vector.
//
  : _max_len (max_len)
{
  assert (max_len > 0);
  _v.reserve (max_len + 1);
}


Fit_Result_Vec::Fit_Result_Vec (const Fit_Result_Vec& vec)
//
// Purpose: Copy constructor.
//
// Inputs:
//   vec -         The vector to copy.
//
  : _v (vec._v),
    _max_len (vec._max_len)
{
}


Fit_Result_Vec::~Fit_Result_Vec ()
//
// Purpose: Destructor.
//
{
}


Fit_Result_Vec& Fit_Result_Vec::operator= (const Fit_Result_Vec& vec)
//
// Purpose: Assignment.
//
// Inputs:
//   vec -         The vector to copy.
//
// Returns:
//   This object.
//
{
  _v = vec._v;
  _max_len = vec._max_len;
  return *this;
}


std::vector<std::shared_ptr<Fit_Result>>::size_type Fit_Result_Vec::size () const
//
// Purpose: Get back the number of results in the vector.
//
// Returns:
//   The number of results in the vector.
//
{
  return _v.size ();
}


const Fit_Result& Fit_Result_Vec::operator[] (std::vector<std::shared_ptr<Fit_Result>>::size_type i) const
//
// Purpose: Get back the Ith result in the vector.
//
// Inputs:
//   i -           The index of the desired result.
//
// Returns:
//   The Ith result.
//
{
  assert (i < _v.size());
  return *_v[i];
}


namespace {


struct Compare_Fitresptr
//
// Purpose: Helper for push().
//
{
  bool operator() (std::shared_ptr<const Fit_Result> a, std::shared_ptr<const Fit_Result> b) const
  {
    return *a < *b;
  }
};


} // unnamed namespace


void Fit_Result_Vec::push (std::shared_ptr<Fit_Result> res)
//
// Purpose: Add a new result to the vector.
//
// Inputs:
//   res -         The result to add.
//
{
  // Find where to add it.
  vector<std::shared_ptr<Fit_Result>>::iterator it = lower_bound (_v.begin(),
                                                  _v.end(),
                                                  res,
                                                  Compare_Fitresptr());

  // Insert it.
  _v.insert (it, res);

  // Knock off the guy at the end if we've exceeded our maximum size.
  if (_v.size() > _max_len) {
    _v.erase (_v.end()-1);
  }
}


/**
    @brief Output stream operator, print the content of this
    Fit_Result_Vec to an output stream.

    @param s The output stream to which to write.

    @param resvec The instance of Fit_Result_Vec to be printed.
*/
std::ostream& operator<< (std::ostream& s, const Fit_Result_Vec& resvec)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   resvec -      The object to write.
//
// Returns:
//   The stream S.
//
{
  for (std::vector<std::shared_ptr<Fit_Result>>::size_type i=0; i < resvec._v.size(); i++)
    s << "Entry " << i << "\n" << *resvec._v[i];
  return s;
}


} // namespace hitfit
