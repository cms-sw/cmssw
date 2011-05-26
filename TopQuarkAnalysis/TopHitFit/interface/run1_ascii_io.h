//
// $Id: run1_ascii_io.h,v 1.3 2009/11/19 17:53:20 haryo Exp $
//
// File: hitfit/run1_ascii_io.h
// Purpose: Read and write the run 1 ntuple dump files.
// Created: Dec, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : interface/run1_ascii_io.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file run1_ascii_io.h

    @brief Header file for legacy code to read and write D0 Run I
    ntuple file. No detailed doxygen documentation will
    be provided.

 */

#ifndef HITFIT_RUN1_ASCII_IO_H
#define HITFIT_RUN1_ASCII_IO_H


#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event.h"
#include "CLHEP/Random/Random.h"
#include <iosfwd>



namespace hitfit {


class Defaults;
class Vector_Resolution;


/**
    @brief Legacy code to read and write D0 Run I
    ntuple file, no detailed doxygen documentation will
    be provided.

 */
class Run1_Ascii_IO_Args
//
// Purpose: Hold on to parameters for run1_ascii_io.
//
//   bool use_e         - If true, then when rescaling the 4-vectors
//   string jet_type_wanted-Character specifying which jet type algorithm
//                        is to be used.
//   string ele_res     - Electron resolutions, for Vector_Resolution.
//   string muo_res     - Muon resolutions, for Vector_Resolution.
//   string jet_res     - Jet resolutions, for Vector_Resolution.
//   string kt_res      - Kt resolution, for Resolution.
//
{
public:
  // Constructor.  Initialize from a Defaults object.
  Run1_Ascii_IO_Args (const Defaults& defs);

  // Retrieve parameter values.
  char jet_type_wanted () const;
  const Vector_Resolution& ele_res () const;
  const Vector_Resolution& muo_res () const;
  const Vector_Resolution& jet_res () const;
  const Resolution& kt_res () const;

private:
  // Hold on to parameter values.
  char _jet_type_wanted;
  Vector_Resolution _ele_res;
  Vector_Resolution _muo_res;
  Vector_Resolution _jet_res;
  Resolution _kt_res;
};


//***************************************************************************


// Read an event from stream S.
Lepjets_Event read_run1_ascii (std::istream& s,
                               const Run1_Ascii_IO_Args& args);

// Write an event to stream S.
std::ostream& write_run1_ascii (std::ostream& s, const Lepjets_Event& ev);


} // namespace hitfit


#endif // not HITFIT_RUN1_ASCII_IO_H
