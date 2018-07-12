/**
 * @file CheckerGccPlugins/src/checker_gccplugins.cxx
 * @author scott snyder <snyder@bnl.gov>
 * @date Aug, 2014
 * @brief Framework for running checker plugins in gcc.
 *
 * Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
 */


#include <string>
#include <iostream>
#include "checker_gccplugins.h"

#include "plugin-version.h"
#include "options.h"
#include "tree-core.h"
#include "tree-pass.h"
#include "plugin.h"
#include "c-family/c-pragma.h"
#include "diagnostic.h"


/* Declare GPL compatible license. */
int plugin_is_GPL_compatible;


/// Table of defined checkers.  All on by default.
struct Checker
{
  std::string name;
  bool enabled;
  void (*initfunc) (plugin_name_args*);
};

static Checker checkers[] = {
#define CHECKER(NAME, FLAG) { #NAME, FLAG, init_##NAME##_checker },
#include "checkers.def"
#undef CHECKER
};


extern "C" {


static
void print_plugin_version(struct plugin_name_args *plugin_info)
{
  printf("ATLAS gcc checker plugins info:\n"
         "\tVersion:  %s\n"
         "\tbase_name: %s\n"
         "\tfull_name: %s\n",
          plugin_info->version,
          plugin_info->base_name,
          plugin_info->full_name);

  printf("GCC info:\n"
         "\tVersion: %s\n",
          gcc_version.basever);
}


static
bool handle_checker_arg (const std::string& arg)
{
  if (arg == "all") {
    for (Checker& c : checkers)
      c.enabled = true;
    return true;
  }

  if (arg == "none") {
    for (Checker& c : checkers)
      c.enabled = false;
    return true;
  }

  for (Checker& c : checkers) {
    if (arg == c.name) {
      c.enabled = true;
      return true;
    }
    else if (arg == "no-" + c.name) {
      c.enabled = false;
      return true;
    }
  }

  return false;
}


/** Collect command-line arguments from `plugin_init(..)`
 *
 * If more options are needed, do a couple of things:-
 *   1. Add the option to `allowed_args`, making sure to increase
 *      its size by one.
 *   2. Add a new `else if` clause, to deal with the option.
 *
 * Not very elegant cf. GCC, but does the job.
 */
static
int
collect_plugin_args (plugin_name_args *plugin_info)
{
  for (int i=0; i < plugin_info->argc; i++)
  {
    plugin_argument& arg = plugin_info->argv[i];
    std::string key = arg.key;
    std::string val_orig = arg.value ? arg.value : "";

    bool bad = false;

    if (key == "version")
      print_plugin_version (plugin_info);

    else if (key == "checkers" && arg.value) {
      char* ptr;
      char* str = arg.value;
      while (char* val = strtok_r (str, ",", &ptr)) {
        str = 0;
        if (!handle_checker_arg (val)) {
          bad = true;
          break;
        }
      }
    }

    else
      bad = true;

    if (bad) {
      // Print a pretty error message, showing in bold where the option
      // went wrong.
      std::cerr << "Unrecognised plugin argument: " << key;
      if (arg.value)
        std::cerr << "=" << val_orig;
      std::cerr << "\n";
      std::cerr << "Allowed arguments:\n";
      std::cerr << "\t-fplugin-arg-" << plugin_info->base_name << "-version\n";
      std::cerr << "\t-fplugin-arg-" << plugin_info->base_name << "-checkers=LIST\n";
      std::cerr << "\n";
      std::cerr << "\tLIST is a comma-separated list of checker names.\n";
      std::cerr << "\tPrefix a name with `no-' to disable it; use `all' or `none'\n";
      std::cerr << "\tto enable/disable all checkers.|n";
      std::cerr << "\n";
      std::cerr << "Available checkers:\n";
      for (Checker& c : checkers)
        std::cerr << "\t" << c.name << "\n";
      return 1;
    }
  }
  return 0;
}

// Plugin info Structs
static struct plugin_info
checker_plugin_info = {
  /*.version =*/ CHECKER_GCCPLUGINS_VERSION_FULL,
  /*.help =*/ "Collection of ATLAS checker plugins for gcc."
};


#if GCC_VERSION >= 8000
# define ATTRIB(A) {A, 0, 0, false, false, false, false, nullptr, nullptr}
#else
# define ATTRIB(A) {A, 0, 0, false, false, false, nullptr, false}
#endif  
static struct
attribute_spec attribs[] =
                     
  { ATTRIB("thread_safe"),
    ATTRIB("not_thread_safe"),
    ATTRIB("not_reentrant"),
    ATTRIB("not_const_thread_safe"),
    ATTRIB("argument_not_const_thread_safe"),
    ATTRIB("check_thread_safety"),
    ATTRIB("check_thread_safety_debug"),
    ATTRIB(nullptr),
  };
#undef ATTRIB


static
void
register_checker_attributes (void* /*event_data*/, void* /*data*/)
{
  //fprintf (stderr, "register_attributes %s\n", thread_safe_attr.name);
  // Through gcc6, the prefix can only be `gnu'.
  // This was fixed in gcc7.
  register_scoped_attributes (attribs, "gnu");
}


static
void
register_checker_pragmas (void* /*event_data*/, void* /*data*/)
{
  c_register_pragma ("ATLAS", "check_thread_safety",
                     CheckerGccPlugins::handle_check_thread_safety_pragma);
  c_register_pragma ("ATLAS", "no_check_thread_safety",
                     CheckerGccPlugins::handle_no_check_thread_safety_pragma);

  cpp_define (parse_in, "ATLAS_GCC_CHECKERS");
}


int
plugin_init(struct plugin_name_args   *plugin_info,
            struct plugin_gcc_version *version)
{
  plugin_info->version = CHECKER_GCCPLUGINS_VERSION_FULL;


  // Complain if built for a different major, or minor version.
  // Ignore patchlevel and other `plugin_gcc_version` attributes.
  if (strncmp(version->basever, gcc_version.basever, 3))
  {
    fprintf(stderr, "Incompatible version (%s). Compiled for %s\n",
        version->basever, gcc_version.basever);
    return 1;
  }

  if (collect_plugin_args(plugin_info)!=0)
      return 1;

  /* Populate the plugin_info struct */
  register_callback(plugin_info->base_name, PLUGIN_INFO,
                    NULL, &checker_plugin_info);

  for (Checker& c : checkers) {
    if (c.enabled) c.initfunc (plugin_info);
  }

  // Register callback to set up common attributes and pragmas.
  register_callback (plugin_info->base_name,
                     PLUGIN_ATTRIBUTES,
                     register_checker_attributes,
                     NULL);
  register_callback (plugin_info->base_name,
                     PLUGIN_PRAGMAS,
                     register_checker_pragmas,
                     NULL);

  return 0;
}


} // extern "C"



namespace CheckerGccPlugins {

void inform_url (location_t loc, const char* url)
{
  if (!in_system_header_at(loc) && !global_dc->dc_warn_system_headers)
    inform (loc, "See %s.", url);
}

}
