//
// $Id: Defaults.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Defaults.h
// Purpose: Define an interface for getting parameter settings.
// Created: Nov, 2000, sss.
//
// This defines a very simple abstract interface for retrieving settings
// for named parameters.  Using this ensures that the hitfit code doesn't
// have to depend on something like rcp.  There is a lightweight concrete
// implementation of this interface, Defaults_Text, which can be used
// for standalone applications.  If this code gets used with the D0 framework,
// a Defaults_RCP can be provided too.
//
// CMSSW File      : interface/Defaults.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Defaults.h
    @brief Define an abstract interface for getting parameter settings.

    This defines a very simple abstract interface for retrieving settings
    for named parameters.  Using ensures that the hitft code doesn't
    have to depend on something like rcp.  There is a lightweight concrete
    implementation of this interface, <i>Defaults_Text</i>,
    which can be used for standalone applications.  If this code gets used
    with the D0 framework, a <i>Defaults_RCP</i> can be provided too.

    @par Creation date:
    November 2000.

    @author
    Scott Stuart Snyder <snyder@bnl.gov>

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Oct 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).
 */

#ifndef HITFIT_DEFAULTS_H
#define HITFIT_DEFAULTS_H


#include <string>


namespace hitfit {


/**
    @class Defaults
    @brief Define an interface for getting parameter settings.
 */
class Defaults
//
// Purpose: Define an interface for getting parameter settings.
//
{
public:
  // Constructor, destructor.

  /**
     Constructor.
   */
  Defaults () {}

  /**
     Destructor
   */
  virtual ~Defaults () {}

  // Test to see if parameter NAME exists.
  /**
     Test to see if parameter <i>name</i> exists.
     @param name The parameter name to be checked.
     @par Return:
     The parameter value.
   */
  virtual bool exists (std::string name) const = 0;

  // Get the value of NAME as an integer.
  /**
     Get the value of <i>name</i> as integer.
     @param name The parameter name.
     @par Return:
     The parameter value.
   */
  virtual int get_int (std::string name) const = 0;

  // Get the value of NAME as a boolean.
  /**
     Get the value of <i>name</i> as boolean.
     @param name The parameter name.
     @par Return:
     The parameter value.
   */
  virtual bool get_bool (std::string name) const = 0;

  // Get the value of NAME as a float.
  /**
     Get the value of <i>name</i> as a floating-point of
     type double.
     @param name The parameter name.
     @par Return:
     The parameter value.
   */
  virtual double get_float (std::string name) const = 0;

  // Get the value of NAME as a string.
  /**
     Get the value of <i>name</i> as a string.
     @param name The parameter name.
     @par Return:
     The parameter value.
   */
  virtual std::string get_string (std::string name) const = 0;
};


} // namespace hitfit


#endif // not HITFIT_DEFAULTS_H
