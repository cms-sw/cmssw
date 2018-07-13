/**
 * @file CheckerGccPlugins/src/check_thread_safety_p.cxx
 * @author scott snyder <snyder@bnl.gov>
 * @date Sep, 2015
 * @brief Test to see if something should be checked for thread-safety.
 *
 * check_thread_safety_p(decl) returns false if DECL has a not_thread_safe
 * attribute.  Otherwise, it returns true in the following cases:
 *
 *  - DECL directly has a check_thread_safety attribute.
 *  - DECL is a function or type and a containing context has thread-safety
 *    checking on.
 *  - The file contains #pragma ATLAS check_thread_safety.
 *  - The package contains a file ATLAS_CHECK_THREAD_SAFETY.
 *
 * If a decl has the attribute check_thread_safety_debug, then a diagnostic
 * will be printed saying if that decl has thread-safety checking enabled.
 *
 * Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
 */


#include <unordered_set>
#include <unordered_map>
#include "checker_gccplugins.h"
#include "tree.h"
#include "diagnostic.h"
#include "print-tree.h"
#include "stringpool.h"
#include "attribs.h"


namespace {


std::unordered_set<location_t> seen_loc;
void thread_safe_debug_finishdecl_callback (void* gcc_data, void* /*user_data*/)
{
  tree decl = (tree)gcc_data;
  if (!DECL_P (decl)) return;
  location_t loc = DECL_SOURCE_LOCATION (decl);
  if (!loc) return;
  if (!seen_loc.insert (loc).second)
    return;

  tree attribs = DECL_ATTRIBUTES (decl);
  if (attribs == NULL_TREE && TREE_CODE (decl) == TYPE_DECL)
    attribs = TYPE_ATTRIBUTES (TREE_TYPE (decl));
  if (lookup_attribute ("check_thread_safety_debug", attribs))
  {
    const char* flag = " not";
    if (CheckerGccPlugins::check_thread_safety_p (decl))
      flag = "";
    warning_at (DECL_SOURCE_LOCATION (decl), 0,
                "%<%D%> is%s marked thread-safe.", decl, flag);
    
  }
}


void thread_safe_debug_pregen_callback (void* gcc_data, void* user_data)
{
  tree ast = (tree)gcc_data;
  if (TREE_CODE (ast) == FUNCTION_DECL)
    thread_safe_debug_finishdecl_callback (gcc_data, user_data);
}


typedef std::unordered_map<std::string, bool> thread_safe_files_t;
thread_safe_files_t thread_safe_files;


const std::unordered_set<std::string> stop_walk_dirs
{
  "InstallArea",

    "AtlasTest",
    "Calorimeter",
    "Commission",
    "Control",
    "Database",
    "DataQuality",
    "DetectorDescription",
    "Event",
    "External",
    "ForwardDetectors",
    "Generators",
    "graphics",
    "InnerDetector",
    "LArCalorimeter",
    "LumiBlock",
    "MagneticField",
    "MuonSpectrometer",
    "PhysicsAnalysis",
    "Reconstruction",
    "Simulation",
    "TestBeam",
    "TileCalorimeter",
    "Tools",
    "Tracking",
    "Trigger",

    "boost",
    "root",
    "Eigen",
    "fastjet",
    "clhep",
    "rootsys",
    "ROOT",
    "gcc",
    "usr",
    };


} // anonymous namespace


namespace CheckerGccPlugins {


bool is_thread_safe_dir (const std::string& dir, int nwalk = 5);
bool is_thread_safe_dir1 (const std::string& dir, int nwalk = 5)
{
  const std::string flagname = "/ATLAS_CHECK_THREAD_SAFETY";
  std::string flagfile = dir + flagname;
  if (access (flagfile.c_str(), R_OK) == 0) {
    return true;
  }

  std::string::size_type dpos = dir.rfind ('/');
  std::string dir2 = "..";
  std::string dname = dir;
  if (dpos != std::string::npos) {
    dir2 = dir.substr (0, dpos);
    dname = dir.substr (dpos+1, std::string::npos);
  }

  std::string cmakefile = dir + "/CMakeLists.txt";
  if (access (cmakefile.c_str(), R_OK) == 0) {
    // Check for a flag file in an include dir.
    flagfile = dir + "/" + dname + flagname;
    if (access (flagfile.c_str(), R_OK) == 0) {
      return true;
    }

    return false;
  }

  if (nwalk <= 0 || stop_walk_dirs.count (dname) > 0)
    return false;

  if (dir2.empty())
    return false;

  if (dname == "..")
    dir2 += "/../..";

  return is_thread_safe_dir (dir2, nwalk-1);
}


bool is_thread_safe_dir (const std::string& dir, int nwalk /*= 5*/)
{
  thread_safe_files_t::iterator it = thread_safe_files.find (dir);
  if (it != thread_safe_files.end()) return it->second;
  bool ret = is_thread_safe_dir1 (dir, nwalk);
  thread_safe_files[dir] = ret;
  return ret;
}


bool check_thread_safety_location_p (location_t loc)
{
  std::string file = LOCATION_FILE(loc);
  thread_safe_files_t::iterator it = thread_safe_files.find (file);
  if (it != thread_safe_files.end()) return it->second;

  std::string::size_type dpos = file.rfind ('/');
  std::string dir = ".";
  if (dpos != std::string::npos)
    dir = file.substr (0, dpos);

  bool ret = is_thread_safe_dir (dir);
  thread_safe_files[file] = ret;
  return ret;
}


/// Has DECL been declared for thread-safety checking?
bool check_thread_safety_p (tree decl)
{
  // Shut off checking entirely if we're processing a root dictionary.
  static enum { UNCHECKED, OK, SKIP } checkedMain = UNCHECKED;
  if (checkedMain == UNCHECKED) {
    if (strstr (main_input_filename, "ReflexDict.cxx") != 0) {
      checkedMain = SKIP;
    }
    else {
      checkedMain = OK;
    }
  }
  if (checkedMain == SKIP) {
    return false;
  }

  tree attribs = DECL_ATTRIBUTES (decl);
  if (attribs == NULL_TREE && TREE_CODE (decl) == TYPE_DECL)
    attribs = TYPE_ATTRIBUTES (TREE_TYPE (decl));

  // Check if attributes are present directly.
  if (lookup_attribute ("not_thread_safe", attribs))
    return false;
  if (lookup_attribute ("check_thread_safety", attribs))
    return true;

  // If it's a function or class, check the containing function or class.
  if (TREE_CODE (decl) == FUNCTION_DECL  ||
      TREE_CODE (decl) == TYPE_DECL)
  {
    tree ctx = DECL_CONTEXT (decl);
    while (ctx && !SCOPE_FILE_SCOPE_P (ctx)) {
      if (TREE_CODE (ctx) == RECORD_TYPE) {
        if (lookup_attribute ("check_thread_safety", TYPE_ATTRIBUTES (ctx)))
          return true;
        if (lookup_attribute ("not_thread_safe", TYPE_ATTRIBUTES (ctx)))
          return false;
        if (check_thread_safety_location_p (DECL_SOURCE_LOCATION (TYPE_NAME (ctx))))
          return true;
        ctx = TYPE_CONTEXT (ctx);
      }

      else if (TREE_CODE (ctx) == FUNCTION_DECL) {
        if (lookup_attribute ("check_thread_safety", DECL_ATTRIBUTES (ctx)))
          return true;
        if (check_thread_safety_location_p (DECL_SOURCE_LOCATION (ctx)))
          return true;
        ctx = DECL_CONTEXT (ctx);
      }

      else
        break;
    }
  }

  // Check the file in which it was declared.
  if (check_thread_safety_location_p (DECL_SOURCE_LOCATION (decl)))
    return true;

  return false;
}


void handle_check_thread_safety_pragma (cpp_reader*)
{
  thread_safe_files[LOCATION_FILE (input_location)] = true;
}


void handle_no_check_thread_safety_pragma (cpp_reader*)
{
  thread_safe_files[LOCATION_FILE (input_location)] = false;
}


} // namespace CheckerGccPlugins


void init_thread_safe_debug_checker (plugin_name_args* plugin_info)
{
  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_DECL,
                     thread_safe_debug_finishdecl_callback,
                     NULL);
  register_callback (plugin_info->base_name,
                     PLUGIN_PRE_GENERICIZE,
                     thread_safe_debug_pregen_callback,
                     NULL);
}
