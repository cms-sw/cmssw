//
// $Id: Fit_Result_Vec.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Fit_Result_Vec.h
// Purpose: Hold a set of Fit_Result structures.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// This class holds pointers to a set of Fit_Result's, resulting from
// different jet permutation with some consistent selection.
// The results are ordered by increasing chisq values.  A maximum
// length for the vector may be specified; when new results are added,
// those with the largest chisq fall off the end.
//
// The Fit_Result objects are reference counted, in order to allow them
// to be entered in multiple vectors.
//
// CMSSW File      : interface/Fit_Result_Vec.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file Fit_Result_Vec.h

    @brief Hold a list of pointers to a set of
    Fit_Result objects, resulting from different jet permutation with
    some consistent selection.

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

#ifndef HITFIT_FIT_RESULT_VEC_H
#define HITFIT_FIT_RESULT_VEC_H


#include <vector>
#include <iosfwd>


namespace hitfit {


class Fit_Result;


/**
    @class Fit_Result_Vec

    @brief Holds pointers to a set of Fit_Result objects, resulting from
    different jet permutation with some consistent selection.  The results
    are ordered by increasing \f$\chi^{2}\f$ values. A maximum length
    for the list of Fit_Result objects may be specified; when new results,
    those with the largest \f$\chi^{2}\f$ value fall off the end.

    The Fit_Result object are reference counted, in order to allow then
    to be entered in multiple vectors.
 */
class Fit_Result_Vec
//
// Purpose: Hold a set of Fit_Result structures.
//
{
public:
  // Constructor, destructor.  The maximum length of the vector
  // is specified here.

  /**
     @brief Constructor.
     @param max_len The maximum length of the list.  Must be a positive
     integer.
   */
  Fit_Result_Vec (std::vector<Fit_Result*>::size_type max_len);

  /**
     @brief Destructor.
   */
  ~Fit_Result_Vec ();

  // Copy constructor.
  /**
     @brief Copy constructor.
     @param vec The list to copy.
   */
  Fit_Result_Vec (const Fit_Result_Vec& vec);

  // Assignment.
  /**
     @brief Assignment operator.
     @param vec The list to copy.
   */
  Fit_Result_Vec& operator= (const Fit_Result_Vec& vec);

  // Get back the number of results in the vector.
  /**
     @brief Return the number of Fit_Result objects in the list.
   */
  std::vector<Fit_Result*>::size_type size () const;

  // Get back the Ith result in the vector.
  /**
     @brief Return the <i>i-</i>th element of the Fit_Result
     objects in the vector.
     @param i The index of the desired Fit_Result object.
   */
  const Fit_Result& operator[] (std::vector<Fit_Result*>::size_type i) const;

  // Add a new result to the list.
  /**
     @brief Add a new Fit_Result to the list.
     @param res The new Fit_Result object to be added into the list of
     Fit_Result object.
   */
  void push (Fit_Result* res);

  // Dump out the vector.
  friend std::ostream& operator<< (std::ostream& s,
                                   const Fit_Result_Vec& resvec);

private:
  // The object state.

  /**
     The list of Fit_Result pointers.
   */
  std::vector<Fit_Result*> _v;

  /**
     Maximum number of Fit_Result pointers in the list.
   */
  std::vector<Fit_Result*>::size_type _max_len;
};


} // namespace hitfit


#endif // not HITFIT_FIT_RESULT_H
