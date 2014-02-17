//
// $Id: Defaults_Text.cc,v 1.1 2011/05/26 09:46:59 mseidel Exp $
//
// File: src/Defaults_Text.cc
// Purpose: A lightweight implementation of the Defaults interface
//          that uses simple text files.
// Created: Jul, 2000, sss.
//
// CMSSW File      : src/Defaults_Text.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file Defaults_Text.cc

    @brief A lightweight implementation of the Defaults
    interface that uses simple text files.  See the documentation for
    the header file Defaults_Text.h for details.

    @par Creation date:
    Jul 2000

    @author
    Scott Stuart Snyder <snyder@bnl.gov> for D0.

    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Oct 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).

 */

#include "TopQuarkAnalysis/TopHitFit/interface/Defaults_Text.h"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cctype>
#include <cstring>
#include <map>

using std::cerr;
using std::string;
using std::ifstream;
using std::getline;
using std::isspace;
using std::tolower;
using std::atoi;
using std::atof;
using std::abort;
using std::strchr;
using std::map;

namespace {

/**
    Helper function to remove comments (text starting with `;' or `#') and
    leading and trailing spaces from string <i>s</i>.
    @param s The string to strip.
    @par Return:
    The stripped string.
 */
string strip (string s)
//
// Purpose: Remove comments (text starting with `;' or `#') and leading
//          and trailing spaces from S.
//
// Inputs:
//   s -           The string to strip.
//
// Returns:
//  The stripped string.
//
{
  string::size_type j = s.find_first_of (";#");
  if (j == string::npos)
    j = s.size();

  while (j > 0 && isspace (s[j-1]))
    --j;

  string::size_type i = 0;
  while (i < j && isspace (s[i]))
    ++i;

  return string (s, i, j-i);
}


} // unnamed namespace


namespace hitfit {


//***************************************************************************


/**
    @class Defaults_Textrep
    @brief The internal representation for a Defaults_Text object.
 */
class Defaults_Textrep
//
// Purpose: The internal representation for a Defaults_Text object.
//
{
public:
  // Constructor.

  /**
     @brief Constructor, construct a Defaults_Textrep instance from an
     ASCII text-file and command line arguments.
     @param file The name of the input ASCII file to read.  See the
     header file for a description of the format for this and the argument
     list. Pass an empty string to skip reading a file.
   */
  Defaults_Textrep (string file);
  /**
     @brief Constructor, construct a Defaults_Textrep instance from an
     ASCII text-file.
     @param file The name of the input ASCII file to read.  See the
     header file for a description of the format for this and the argument
     list. Pass an empty string to skip reading a file.
     @param argc The length of the argument list.
     @param argv The argument list.
   */
  Defaults_Textrep (string file, int argc, char** argv);

  // The data.  Maps from parameter names to values (which are stored
  // as strings).
  /**
     The data, maps from parameter names to values (which are stored
     as strings)
   */
  std::map<std::string,std::string> _map;

  // Look up parameter NAME and return its value.
  /**
     Look up parameter <i>name</i> and return its value.
     @param name The name of the parameter.
     @par Return:
     The value of the parameter in C++ string format.
   */
  string get_val (string name) const;


private:
  // Read parameters from FILE and add them to our data.
  /**
     Read parameters from ASCII text file <i>file</i> and add the content
     to data.
     @param file The ASCII text file to read.
   */
  void read_file (string file);

  // Look for additional parameter settings in the argument list
  // ARGC, ARGV and add them to our data.
  /**
     Look for additional parameter settings in the argument list
     <i>argc</i> and <i>argv</i> and add the content to data.
     @param argc The length of the argument list.
     @param argv The argument list.
   */
  void process_args (int argc, char** argv);

  // Helper to process a line defining a single parameter.
  /**
     Helper function to process a line defining a single parameter.
     @param l The line to process.
   */
  void doline (string l);
};


Defaults_Textrep::Defaults_Textrep (string file)
//
// Purpose: Constructor.
//
// Inputs:
//   file -        The name of the defaults file to read.
//                 See the comments in the header for a description
//                 of the format for this and for the argument list.
//                 Pass an empty string to skip reading a file.
//
{
  read_file (file);
}


Defaults_Textrep::Defaults_Textrep (string file, int argc, char** argv)
//
// Purpose: Constructor.
//
// Inputs:
//   file -        The name of the defaults file to read.
//                 See the comments in the header for a description
//                 of the format for this and for the argument list.
//                 Pass an empty string to skip reading a file.
//   argc -        The arglist length.
//   argv -        The arglist.
//
{
  read_file (file);
  process_args (argc, argv);
}


void Defaults_Textrep::read_file (string file)
//
// Purpose: Read parameters from FILE and add them to our data.
//
// Inputs:
//   s -           The name of the file to read.
//
{
  // Just return if we weren't given a file.
  if (file.size() == 0)
    return;

  ifstream f (file.c_str());
  if (!f.good()) {
    cerr << "Can't open " << file << "\n";
    abort ();
  }

  string l;
  while (getline (f, l)) {
    doline (l);
  }

  f.close ();
}


void Defaults_Textrep::process_args (int argc, char** argv)
//
// Purpose: Process the argument list ARGC, ARGV and add additional
//          parameters from it to our data (possibly overriding
//          existing settings).  See the header file for more details.
//
// Inputs:
//   argc -        The arglist length.
//   argv -        The arglist.
//
{
  // Look for arguments starting with `--'.
  for (int i=1; i < argc; i++) {
    if (argv[i][0] == '-' && argv[i][1] == '-') {

      // Found one. 
      string l;
      if (strchr (argv[i], '=') != 0)
        // It was of the form `--NAME=VALUE'.  Change to `NAME=VALUE'.
        l = argv[i] + 2;
      else if (argv[i][2] == 'n' && argv[i][3] == 'o') {
        // It was of the form `--noNAME'.  Change to `NAME=0'.
        l = argv[i] + 4;
        l += "=0";
      }
      else {
        // It was of the form `--NAME'.  Change to `NAME=1'. 
        l = argv[i] + 2;
        l += "=1";
      }

      // Process it like a line we read from a file.
      doline (l);
    }
  }
}


string Defaults_Textrep::get_val (string name) const
//
// Purpose: Look up parameter NAME and return its value.
//          The parameter must exist.
//
// Inputs:
//   name -        The name of the parameter.
//
// Returns:
//   The value of the parameter.
//
{

    std::string val;

    if (_map.find(name) == _map.end()) {
    cerr << "can't find default for " << name << "\n";
    abort ();
    } else {
    std::map<string,string>::const_iterator it = _map.find(name);
    val = it->second;
    }

    return val;
}


void Defaults_Textrep::doline (string l)
//
// Purpose: Helper to process a line defining a single parameter.
//
// Inputs:
//   l -           The line to process.
//
{
  // Strip spaces from the line and ignore it if it's blank.
  l = strip (l);
  if (l.size() == 0)
    return;

  // It must contain a `=' character.
  string::size_type pos = l.find ('=');
  if (pos == string::npos) {
    cerr << "bad defaults line " << l << "\n";
    abort ();
  }

  // Split off name and value parts.
  std::string name = strip (l.substr (0, pos));
  std::string val = strip (l.substr (pos+1));

  // Add it to the map.
  _map[name] = val;

}


//***************************************************************************


Defaults_Text::Defaults_Text (std::string def_file)
//
// Purpose: Constructor.
//
// Inputs:
//   def_file -    The name of the defaults file to read.
//                 See the comments in the header for a description
//                 of the format for this and for the argument list.
//                 Pass an empty string to skip reading a file.
//
  : _rep (new Defaults_Textrep (def_file))
{
}


	Defaults_Text::Defaults_Text (std::string def_file, int argc, char** argv)
//
// Purpose: Constructor.
//
// Inputs:
//   def_file -    The name of the defaults file to read.
//                 See the comments in the header for a description
//                 of the format for this and for the argument list.
//                 Pass an empty string to skip reading a file.
//   argc -        The arglist length.
//   argv -        The arglist.
//
  : _rep (new Defaults_Textrep (def_file, argc, argv))
{
}

Defaults_Text::~Defaults_Text ()
//
// Purpose: Destructor.
//
{
  delete _rep;
}


bool Defaults_Text::exists (std::string name) const
//
// Purpose: Test to see if parameter NAME exists.
//
// Inputs:
//   name -        The name of the parameter to look up.
//
// Returns:
//   True if NAME exists.
//
{
  std::string val;
  return (_rep->_map.find(name) != _rep->_map.end());

}


int Defaults_Text::get_int (std::string name) const
//
// Purpose: Get the value of NAME as an integer.
//
// Inputs:
//   name -        The name of the parameter to look up.
//
// Returns:
//   The parameter's value as an integer.
//
{
  return atoi (_rep->get_val (name).c_str());
}


double Defaults_Text::get_float (std::string name) const
//
// Purpose: Get the value of NAME as a float.
//
// Inputs:
//   name -        The name of the parameter to look up.
//
// Returns:
//   The parameter's value as a float.
//
{
  return atof (_rep->get_val (name).c_str());
}


bool Defaults_Text::get_bool (std::string name) const
//
// Purpose: Get the value of NAME as a bool.
//
// Inputs:
//   name -        The name of the parameter to look up.
//
// Returns:
//   The parameter's value as a bool.
//
{
  string val = _rep->get_val (name);
  if (tolower (val[0]) == 't' || tolower (val[0]) == 'y')
    return true;
  else if (tolower (val[0]) == 'f' || tolower (val[0]) == 'n')
    return false;
  return !!get_int (name);
}


string Defaults_Text::get_string (std::string name) const
//
// Purpose: Get the value of NAME as a string.
//
// Inputs:
//   name -        The name of the parameter to look up.
//
// Returns:
//   The parameter's value as a string.
//
{
  return _rep->get_val (name);
}


std::ostream& operator<< (std::ostream& s, const Defaults_Text& def)
//
// Purpose: Dump out all parameter settings.
//
// Inputs:
//   s -           The stream to which we're writing.
//   def -         The instance to dump.
//
// Returns:
//   The stream S.
//
{

    for (std::map<std::string,std::string>::const_iterator it = def._rep->_map.begin() ;
     it != def._rep->_map.end() ;
     it++) {
	 s << "[" << it->first << "] = [" << it->second << "]\n";
    }

  return s;
}


} // namespace hitfit
