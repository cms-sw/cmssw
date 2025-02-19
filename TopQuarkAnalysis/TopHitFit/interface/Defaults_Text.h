//
// $Id: Defaults_Text.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Defaults_Text.h
// Purpose: A lightweight implementation of the Defaults interface
//          that uses simple text files.
// Created: Jul, 2000, sss.
//
// Create instances of these objects passing in the name of a file.
// Each line of the file should contain a parameter setting like
//
//   NAME = VALUE
//
// Anything following a `;' or `#' is stried off; leading and trailing
// spaces on VALUE are also removed.  Blank lines are ignored.
//
// You can also pass an argument list to the constructor.  After the
// defaults file is read, the argument list will be scanned, to possibly
// override some of the parameter settings.  An argument of the form
//
//   --NAME=VALUE
//
// is equivalent to the parameter setting
//
//     NAME=VALUE
//
// while
//
//   --NAME
//
// is equivalent to
//
//   NAME=1
//
// and
//
//   --noNAME
//
// is equivalent to
//
//   NAME=0
//
// CMSSW File      : interface/Defaults_Text.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Defaults_Text.h
    @brief Define a concrete interface for getting parameter settings from
    an ASCII text file.

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
#ifndef HITFIT_DEFAULTS_TEXT_H
#define HITFIT_DEFAULTS_TEXT_H

#include <string>
#include <iosfwd>
#include "TopQuarkAnalysis/TopHitFit/interface/Defaults.h"


namespace hitfit {


class Defaults_Textrep;


/**
    @brief A lightweight implementation of the Defaults interface that
    uses simple ASCII text files.

    Create instances of these objects passing in the name of a file.
    Each line of the file should contain a parameter setting like

    <i>name</i> = <b>value</b>

    Anything following a `;' or `#' is stried off; leading and trailing
    whitespaces on <b>value</b> are also removed.  Blank lines are ignored.

    User can also pass an argument list to the constructor.  After the default
    ASCII input file is read, the argument list will be scanned, to possibly
    override some of the parameter settings.  An argument of the form

    <i>--name=value</i>

    is equivalent to the parameter setting

    <i>name</i> = <b>value</b>

    while

    <i>--name</i>

    is equivalent to

    <i>name</i> = <b>1</b>

    and

    <i>--noname</i>

    is equivalent to

    <i>name</i> = <b>0</b>.

 */
class Defaults_Text
  : public Defaults
//
// Purpose: A lightweight implementation of the Defaults interface
//          that uses simple text files.
//
{
public:
  // Constructor, destructor.

  /**
     @brief Constructor, create a Default_Text object from an ASCII text
     file. Pass an empty string to skip reading a file.
     @param def_file The ASCII text file to read.  Pass an empty string
     to skip reading a file.
   */
  Defaults_Text (std::string def_file);

  /**
     @brief Constructor, create a Default_Text object from an ASCII text
     file and argument list.
     @param def_file The ASCII text file to read.  Pass an empty string
     to skip reading a file.
     @param argc The length of the argument list.
     @param argv The argument list.
   */
  Defaults_Text (std::string def_file, int argc, char** argv);

  /**
    @brief Destructor.
  */
  ~Defaults_Text ();

  // Test to see if parameter NAME exists.
  /**
     Test to see if parameter <i>name</i> exists.
     @param name The name of the parameter.
     @par Return:
     <b>true</b> if the parameter exists.<br>
     <b>false</b> if the parameter does not exist.<br>
   */
  virtual bool exists (std::string name) const;

  // Get the value of NAME as an integer.
  /**
     Get the value of <i>name</i> as integer.
     @param name The name of the parameter.
     @par Return:
     The value of the parameter an integer (C/C++ int).
   */
  virtual int get_int (std::string name) const;

  // Get the value of NAME as a boolean.
  /**
     Get the value of <i>name</i> as boolean.
     @param name The name of the parameter.
     @par Return:
     The value of the parameter a C/C++ bool.
   */
  virtual bool get_bool (std::string name) const;

  // Get the value of NAME as a float.
  /**
     Get the value of <i>name</i> as a floating-point of
     type double.
     @param name The name of the parameter.
     @par Return:
     The value of the parameter as a floating-point number (C/C++ double).
  */
  virtual double get_float (std::string name) const;

  // Get the value of NAME as a string.
  /**
     Get the value of <i>name</i> as a string.
     @param name The name of the parameter.
     @par Return:
     The value of the parameter as a string.
   */
  virtual std::string get_string (std::string name) const;

  // Dump out all parameters.
  /**
     Output stream operator.  Print out all parameters' names and their
     values.
     @param s The output stream to write.
     @param def The instance to print.
     @par Return:
     The output stream <i>s</i>
   */
  friend std::ostream& operator<< (std::ostream& s, const Defaults_Text& def);

private:
  // The internal representation.
  /**
     The internal representation.
   */
  Defaults_Textrep* _rep;
};


} // namespace hitfit


#endif // not HITFIT_DEFAULTS_TEXT_H
