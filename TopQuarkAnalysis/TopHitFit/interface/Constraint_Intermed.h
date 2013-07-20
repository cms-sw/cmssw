//
// $Id: Constraint_Intermed.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/private/Constraint_Intermed.h
// Purpose: Represent one side of a mass constraint equation.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// Mass constraints come in two varieties, either saying that the sum
// of a set of labels should equal a constant:
//
//     (1 + 2) = 80
//
// or that two such sums should equal each other:
//
//     (1 + 2) = (3 + 4)
//
// These classes represent one side of such an equation.
// There is an abstract base class Constraint_Intermed, and then concrete
// implementations for the two cases, Constraint_Intermed_Constant
// and Constraint_Intermed_Labels.  There is also a free function
// make_constraint_intermed() to parse a string representing one
// side of a constraint and return the appropriate Constraint_Intermed
// instance.
//
// CMSSW File      : interface/Constraint_Intermed.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Constraint_Intermed.h

    @brief Represent one side of a mass constraint equation.
    Contains the abstract base class <i>Constraint_Intermed</i> and concrete
    implementations <i>Constraint_Intermed_Constant</i> and
    <i>Constraint_Intermed_Labels</i>.

    Mass constraints come in two varieties, either saying that the sum
    of a set of labels should equal a constant:<br>

    \f$(1 + 2) = 80\f$.<br>

    or that two such sums should equal each other:<br>

    \f$(1 + 2) = (3 + 4)\f$.<br>

    These classes represent one side of such an equation.
    There is an abstract base class <i>Constraint_Intermed</i>, and then the
    concrete implementations of the two cases, <i>Constraint_Intermed_Constant</i>
    and <i>Constraint_Intermed_Labels</i>.  There is also a free function
    make_constant_intermed() to parse a string representation one
    side of the constraint and return the appropriate <i>Constraint_Intermed</i>
    instance.

    @par Creation date:
    July 2000.

    @author
    Scott Stuart Snyder <snyder@bnl.gov>.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Oct 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).
 */


#ifndef HITFIT_CONSTRAINT_INTERMED_H
#define HITFIT_CONSTRAINT_INTERMED_H


#include <iosfwd>
#include <vector>
#include <string>
#include <memory>


namespace hitfit {


class Fourvec_Event;


//************************************************************************


/**
   @class Constraint_Intermed.

   @brief Abstract base classes for describing one side of a mass constraint.
 */
class Constraint_Intermed
//
// Purpose: Abstract base class for describing one side of a mass constraint.
//
{
public:
  // Constructor, destructor.

  /**
     Constructor.
   */
  Constraint_Intermed () {}

  /**
     Destructor.
   */
  virtual ~Constraint_Intermed () {}

  // Return true if this guy references both labels ILABEL and JLABEL.
  /**

     Check the instance for reference of <i>ilabel</i> and <i>jlabel</i>.

     @param ilabel The first label to test.
     @param jlabel The second label to test.

     @par Return:
     <b>true</b> if this instance references both labels <i>ilabel</i>
     and <i>jlabel</i>.<br>
     <b>false</b> if this instance doesn't reference both labels.

   */
  virtual bool has_labels (int ilabel, int jlabel) const = 0;

  // Evaluate this half of the mass constraint, using the data in EV.
  // Return m^2/2.
  /**
     Evaluate this half of the mass constraint, using the data in <i>ev</i>.

     @param ev The event for which the mass constraint is to be evaluated.

     @par Return:
     \f$\frac{m^{2}}{2}\f$.
   */
  virtual double sum_mass_terms (const Fourvec_Event& ev) const = 0;

  // Print out this object.
  /**
     Print out the instance to the output stream.

     @param s The output stream to which the instance is printed.
   */
  virtual void print (std::ostream& s) const = 0;

  // Copy this object.
  /**
     Clone function to copy the instance.
   */
  virtual std::auto_ptr<Constraint_Intermed> clone () const = 0;
};


//************************************************************************


/**
   @class Constraint_Intermed_Constant
   @brief Concrete class for one side of mass constraint
   equation of the type:<br>
   \f$(1 + 2) = C\f$.
 */
class Constraint_Intermed_Constant
  : public Constraint_Intermed
//
// Purpose: Concrete base class for a constant mass constraint half.
//
{
public:
  // Constructor, destructor.
  /**
     Constructor.

     @param constant The mass constraint of the constraint equation.
   */
  Constraint_Intermed_Constant (double constant);

  /**
     Destructor.
   */
  virtual ~Constraint_Intermed_Constant () {};

  // Copy constructor.
  /**
     Copy constructor.
     @param c The instance to be copied.
   */
  Constraint_Intermed_Constant (const Constraint_Intermed_Constant& c);

  // Return true if this guy references both labels ILABEL and JLABEL.
  /**

     Check the instance for reference of <i>ilabel</i> and <i>jlabel</i>.

     @param ilabel The first label to test.
     @param jlabel The second label to test.

     @par Return:
     <b>true</b> if this instance references both labels <i>ilabel</i>
     and <i>jlabel</i>.<br>
     <b>false</b> if this instance doesn't reference both labels.

   */
  virtual bool has_labels (int ilabel, int jlabel) const;

  // Evaluate this half of the mass constraint, using the data in EV.
  // Return m^2/2.
  /**
     Evaluate this half of the mass constraint, using the data in <i>ev</i>.

     @param ev The event for which the mass constraint is to be evaluated.

     @par Return:
     \f$\frac{m^{2}}{2}\f$.
   */
  virtual double sum_mass_terms (const Fourvec_Event& ev) const;

  // Print out this object.
  /**
     Print out the instance to the output stream.

     @param s The output stream to which the instance is printed.
   */
  virtual void print (std::ostream& s) const;

  // Copy this object.
  /**
     Clone function to copy the instance.
   */
  virtual std::auto_ptr<Constraint_Intermed> clone () const;

private:
  // Store c^2 / 2.
  /**
     The mass constraint value.
   */
  double _c2;
};


//************************************************************************


/**

   @class Constraint_Intermed_Labels

   @brief Concrete class for one side of mass constraint
   equation of the type:<br>
   \f$(1 + 2) =  (3 + 4)\f$.

 */
class Constraint_Intermed_Labels
  : public Constraint_Intermed
{
public:

  /**
     Constructor.
     @param labels The labels used by this side of mass constraint.
   */
  Constraint_Intermed_Labels (const std::vector<int>& labels);

  /**
     Copy constructor.
     @param c The instance of Constraint_Intermed_Labels to be copied.
   */
  Constraint_Intermed_Labels (const Constraint_Intermed_Labels& c);
  /**
     Destructor.
   */
  virtual ~Constraint_Intermed_Labels () {};

  // Return true if this guy references both labels ILABEL and JLABEL.
  /**

     Check the instance for reference of <i>ilabel</i> and <i>jlabel</i>.

     @param ilabel The first label to test.
     @param jlabel The second label to test.

     @par Return:
     <b>true</b> if this instance references both labels <i>ilabel</i>
     and <i>jlabel</i>.<br>
     <b>false</b> if this instance doesn't reference both labels.

   */
  virtual bool has_labels (int ilabel, int jlabel) const;

  // Evaluate this half of the mass constraint, using the data in EV.
  // Return m^2/2.
  /**
     Evaluate this half of the mass constraint, using the data in <i>ev</i>.

     @param ev The event for which the mass constraint is to be evaluated.

     @par Return:
     \f$\frac{m^{2}}{2}\f$.
   */
  virtual double sum_mass_terms (const Fourvec_Event& ev) const;

  // Print out this object.
  /**
     Print out the instance to the output stream.

     @param s The output stream to which the instance is printed.
   */
  virtual void print (std::ostream& s) const;

  // Copy this object.
  /**
     Clone function to copy the instance.
   */
  virtual std::auto_ptr<Constraint_Intermed> clone () const;

private:
  // Test to see if LABEL is used by this constraint half.
  /**
     Test to see if <i>label</i> is used by this side of the mass constraint.
     @param label The label for which to search.
     @par Return:
     <b>true</b> is this constraint use the label.
     <b>false</b> if this constraint doesn't use the label.
   */
  bool has_label (int label) const;

  // List of the labels for this constraint half, kept in sorted order.
  /**
     List of the labels for this side of mass constraint, kept in sorted
     order.
   */
  std::vector<int> _labels;

  // Disallow assignment
  /**
     Disallow assignment by NOT defining the assignment operation.
   */
  Constraint_Intermed& operator= (const Constraint_Intermed&);
};



//************************************************************************


// Print out a Constraint_Intermed object.
std::ostream& operator<< (std::ostream& s, const Constraint_Intermed& ci);


// Parse the string S and construct the appropriate Constraint_Intermed
// instance.
/**
   Helper function to parse input string <i>s</i> and construct the
   appropriate Constraint_Intermed instance.  Returns null if the input
   string cannot be interpreted as a mass constraint.

   The string should be either a numeric constant like

   80.4

   or a list of integers in parenthesis like

   (1 2 4)

   Leading spaces are ignored, as is text in a leading <>
   construction.

   @param s The string to parse which contains information about the mass
   constraint.
   @par Return:
   <b>Valid pointer</b> to the appropriate type of constraint object,
   i.e. a Constraint_Intermed_Constant or a
   Constraint_Intermed_Labels.<br>
   <b>NULL pointer</b> if the input string cannot be interpreted as
   a mass constraint.
 */
std::auto_ptr<Constraint_Intermed> make_constraint_intermed (std::string s);


} // namespace hitfit


#endif // not HITFIT_CONSTRAINT_INTERMED_H
