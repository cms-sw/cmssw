// This file's extension implies that it's C, but it's really -*- C++ -*-.
/**
 * @file CheckerGccPlugins/src/checker_gccplugins.h
 * @author scott snyder <snyder@bnl.gov>
 * @date Aug, 2014
 * @brief Framework for running checker plugins in gcc.
 *
 * Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
 */


#ifndef CHECKERGCCPLUGINS_CHECKER_GCCPLUGINS_H
#define CHECKERGCCPLUGINS_CHECKER_GCCPLUGINS_H


#ifdef PACKAGE_VERSION
# undef PACKAGE_VERSION
#endif
#include "config.h"
#ifdef HAVE_DECL_GETOPT
# undef HAVE_DECL_GETOPT
#endif
#define HAVE_DECL_GETOPT 1

#include "gcc-plugin.h"
#include "input.h"
#include <string.h>

#define CHECKER_GCCPLUGINS_VERSION_FULL "0.1"
#define CHECKER_GCCPLUGINS_C_VERSION "Atlas gcc checker plugins version: " CHECKER_GCCPLUGINS_VERSION_FULL


namespace CheckerGccPlugins {


/// Has DECL been declared thread-safe?
bool check_thread_safety_p (tree decl);

bool check_thread_safety_location_p (location_t loc);

void handle_check_thread_safety_pragma (cpp_reader*);
void handle_no_check_thread_safety_pragma (cpp_reader*);

void inform_url (location_t loc, const char* url);

inline
bool startswith (const char* s, const char* prefix)
{
  return strncmp (s, prefix, strlen(prefix)) == 0;
}


typedef gimple* gimplePtr;


} // namespace CheckerGccPlugins


// Declare prototypes for the checker initialization functions.
#define CHECKER(NAME, FLAG) void init_##NAME##_checker (plugin_name_args* plugin_info);
#include "checkers.def"
#undef CHECKER


#endif // not CHECKERGCCPLUGINS_CHECKER_GCCPLUGINS_H
