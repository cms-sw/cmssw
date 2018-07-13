/**
 * @file CheckerGccPlugins/src/usingns_plugin.cxx
 * @author scott snyder <snyder@bnl.gov>
 * @date Jul, 2015
 * @brief Check for bad uses of `using namespace' and `using'.
 *
 * Complain about `using namespace' in a header file, or in the main file
 * followed by an #include.
 *
 * Complain about `using' in the global namespace in a header file, or in the
 * main file followed by an #include.
 *
 * Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
 *
 * NOTE: This currently requires a patch to gcc to work correctly.
 */


#include <unordered_set>
#include <vector>

#include "checker_gccplugins.h"
#include "tree.h"
#include "function.h"
#include "basic-block.h"
#include "coretypes.h"
#include "is-a.h"
#include "predict.h"
#include "internal-fn.h"
#include "tree-ssa-alias.h"
#include "gimple-expr.h"
#include "gimple.h"
#include "gimple-iterator.h"
#include "tree-ssa-loop.h"
#include "cp/cp-tree.h"
#include "diagnostic.h"
#include "context.h"
#include "tree-pass.h"
#include "gimple-pretty-print.h"
#include "print-tree.h"
#include "tree-cfg.h"
#include "cfgloop.h"
#include "tree-ssa-operands.h"
#include "tree-phinodes.h"
#include "gimple-ssa.h"
#include "ssa-iterators.h"


using CheckerGccPlugins::startswith;


namespace {


const char* url = "<https://twiki.cern.ch/twiki/bin/view/AtlasComputing/CheckerGccPlugins#usingns_plugin>";


/// Does HAYSTACK end with NEEDLE?
bool strendswith_p (const char* haystack, const char* needle)
{
  size_t len_haystack = strlen(haystack);
  size_t len_needle = strlen(needle);
  return strcmp (haystack + len_haystack - len_needle, needle) == 0;
}


bool ignore_p (tree ns)
{
  while (ns && TREE_CODE(ns) == NAMESPACE_DECL && !SCOPE_FILE_SCOPE_P(ns)) {
    tree nsname_node = DECL_NAME (ns);
    if (nsname_node) {
      const char* nsname = IDENTIFIER_POINTER (nsname_node);
      if (strcmp (nsname, "boost") == 0) return true;
    }
    else {
      // Anonymous namespace.
      return true;
    }
    ns = DECL_CONTEXT(ns);
  }
  return false;
}


#if GCC_VERSION < 8000
// Called at the end of compilation (with FROM_INCLUDE false) and after seeing
// the first decl in a header (with FROM_INCLUDE true).  On the initial call,
// NS should be the global namespace.  Walk all namespaces reachable from NS
// and check if any of them were used in a `using namespace' directive; warn
// if appropriate.
std::unordered_set<location_t> seen_loc;
void scan_ns_for_using (tree ns, bool from_include)
{
  if (!DECL_NAME (ns)) return;

  // Loop through all namespaces that use NS.
  for (tree t = DECL_NAMESPACE_USERS (ns); t; t = TREE_CHAIN (t)) {
    // Get the location of the using directive.
    tree stmt = TREE_VALUE (t);
    location_t loc = EXPR_LOCATION (stmt);
    if (!loc) continue;

    if (ignore_p (TREE_PURPOSE (t))) continue;

    // Don't warn about a location more than once.
    if (seen_loc.count (loc) > 0) continue;

    const char* file = LOCATION_FILE (loc);

    // boost
    if (strendswith_p (file, "boost/bind.hpp")) return;

    // library
    if (strendswith_p (file, "bits/c++config.h")) return;

    // using namespace std;
    if (strendswith_p (file, "VariableRPlugin.hh"))  return;
    if (strendswith_p (file, "/G4INCLXXInterface.hh")) return;

    // using namespace EventStorage;
    if (strendswith_p (file, "EventStorage/DataReader.h")) return;

    // Warn if the directive wasn't in the main source file
    // (i.e., it was something included).
    if (strcmp (file, main_input_filename) != 0)
    {
      // For now, only warn about names being brought into the global NS.
      if (TREE_PURPOSE(t) == global_namespace) {
        warning_at (loc, 0,
                    "Do not use `using namespace' in an #included file.");
        CheckerGccPlugins::inform_url (loc, url);
        seen_loc.insert (loc);
      }
    }

    // Also warn if we got here while processing a header file; that means
    // that the using directive was given before an #include.
    else if (from_include) {
      warning_at (loc, 0,
                  "Do not use `using namespace' before an #include.");
      seen_loc.insert (loc);
      CheckerGccPlugins::inform_url (loc, url);
    }
  }

  // Recursively walk contained namespaces.
  cp_binding_level* level = NAMESPACE_LEVEL(ns);
  for (tree decl = level->namespaces;
       decl;
       decl = TREE_CHAIN(decl))
  {
    if (decl)
      scan_ns_for_using (decl, from_include);
  }
}


// Called at end of compilation.
void usingns_finish_callback (void* /*gcc_data*/, void* /*user_data*/)
{
  if (global_namespace)
    scan_ns_for_using (global_namespace, false);
}
#endif


// Handle a toplevel `using' declaration.
std::vector<location_t> global_using_decls;
void handle_using_decl (tree /*decl*/)
{
  // Only warn about declarations in the global namespace.
  if (current_namespace == global_namespace)
  {
    // Give a warning now if this decl was from a header file.
    // nb. DECL_SOURCE_LOCATION(decl) is not reliable.
    location_t loc = input_location;

    const char* file = LOCATION_FILE (loc);

    // Rtypeinfo.h contains
    //   using std::type_info;
    // Bleh.  Ignore it.
    if (strendswith_p (file, "/Rtypeinfo.h"))  return;

    // using RooFit::operator<< ;
    if (strendswith_p (file, "/RooPrintable.h")) return;

    // Eigen/Core also has using declarations.
    if (strendswith_p (file, "Eigen/Core"))  return;

    // This has using decls for AIDA.
    if (strendswith_p (file, "GaudiKernel/IHistogramSvc.h"))  return;

    // using std::chrono::system_clock;
    if (strendswith_p (file, "ers/IssueFactory.h"))  return;
    if (strendswith_p (file, "ers/Issue.h"))  return;

    // using decls from g4
    if (strendswith_p (file, "/G4SystemOfUnits.hh")) return;
    if (strendswith_p (file, "/G4PhysicalConstants.hh")) return;
    if (strendswith_p (file, "/G4RadioactivityTable.hh")) return;


    if (strcmp (file, main_input_filename) != 0)
    {
      warning_at (loc, 0,
                  "Do not use a using declaration in the global namespace in an #included file");
      CheckerGccPlugins::inform_url (loc, url);
    }

    else
    {
      // Otherwise, remember it.  We'll generate warnings for these
      // if we later notice that we're in a header file.
      global_using_decls.push_back (loc);
    }
  }
}


// Handle a toplevel using directive.
std::vector<location_t> global_using_stmts;
void handle_using_stmt (tree stmt)
{
  // Only warn if we're in the global namespace.
  if (current_namespace != global_namespace) return;
  if (current_function_decl) return;

  // Give a warning now if this we're in a header file.
  location_t loc = EXPR_LOCATION (stmt);
  if (!loc) return;

  if (ignore_p (USING_STMT_NAMESPACE (stmt))) return;

  const char* file = LOCATION_FILE (loc);

  // boost
  if (strendswith_p (file, "boost/bind.hpp")) return;

  // library
  if (strendswith_p (file, "bits/c++config.h")) return;

  // using namespace std;
  if (strendswith_p (file, "VariableRPlugin.hh"))  return;
  if (strendswith_p (file, "/G4INCLXXInterface.hh")) return;

  // using namespace EventStorage;
  if (strendswith_p (file, "EventStorage/DataReader.h")) return;

  // Warn if the directive wasn't in the main source file
  // (i.e., it was something included).
  if (strcmp (file, main_input_filename) != 0)
  {
    warning_at (loc, 0,
                "Do not use `using namespace' in an #included file.");
    CheckerGccPlugins::inform_url (loc, url);
  }
  else {
    // Otherwise remember it; we'll want to warn later if we find
    // an #include after this.
    global_using_stmts.push_back (loc);
  }
}


// Called after processed a declaration.
// This can be a using declaration, which we need to handle.
// If this is the first decl in a new header, we do a
// `using namespace' scan.  Also, if we've seen any uing declarations
// in the global namespace from the main input file, we warn at this point.
std::unordered_set<std::string> seen_files;
void usingns_finishdecl_callback (void* gcc_data, void* /*user_data*/)
{
  tree decl = (tree)gcc_data;
  if (TREE_CODE (decl) == USING_STMT)
  {
    handle_using_stmt (decl);
    return;
  }
  if (TREE_CODE (decl) == USING_DECL)
  {
    handle_using_decl (decl);
  }

  // Ignore template instantiations.  They'll have a location of the
  // file containing the template, but our input location will be
  // somewhere else.
  {
    tree ctx = decl;
    while (ctx && DECL_P (ctx) && !SCOPE_FILE_SCOPE_P (ctx)) {
      if (DECL_LANG_SPECIFIC (ctx) && DECL_USE_TEMPLATE (ctx)) return;
      ctx = DECL_CONTEXT (ctx);
    }
  }

  // Some internal objects can also appear out of order.
  {
    tree name_node = DECL_NAME (decl);
    if (name_node) {
      const char* name = IDENTIFIER_POINTER (name_node);
      if (startswith (name, "_ZTI")) return; // typeinfo
      if (startswith (name, "_ZGV")) return; // guard var
      if (startswith (name, "_ZTV")) return; // vtable
      if (startswith (name, "_ZTS")) return; // typeinfo name
      if (strcmp (name, "__dso_handle") == 0) return;
    }
  }

  // Is this decl in a header file?
  const char* file = LOCATION_FILE (input_location);

  if (file && strcmp (file, main_input_filename) != 0)
  {
    // Yes.  Is this the first time we've seen this header?
    if (seen_files.insert (file).second)
    {
#if GCC_VERSION < 8000
      // Yes.  Do `using namespace' scan.
      scan_ns_for_using (global_namespace, true);
#endif

      // Emit warnings about previous `using' directives
      // in the global namespace.
      for (const location_t& loc : global_using_stmts)
      {
        warning_at (loc, 0,
                    "Do not use `using namespace' before an #include.");
        CheckerGccPlugins::inform_url (loc, url);
      }
      global_using_stmts.clear();

      // Same about previous declarations.
      for (const location_t& loc : global_using_decls)
      {
        warning_at (loc, 0,
                    "Do not use a using declaration in the global namespace before an #included file");
        CheckerGccPlugins::inform_url (loc, url);
      }
      global_using_decls.clear();
    }
  }
}


} // anonymous namespace


void init_usingns_checker (plugin_name_args* plugin_info)
{
  // Don't run this plugin for generated dictionary files.
  // ROOT puts using directives in the generated code.
  if (strstr (main_input_filename, "_gen.cxx")) return;
  if (strstr (main_input_filename, "_gen.cpp")) return;
  if (strstr (main_input_filename, "_rootcint.cxx")) return;
  if (strstr (main_input_filename, "_rootcint.cpp")) return;

#if GCC_VERSION < 8000
  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_UNIT,
                     usingns_finish_callback,
                     NULL);
#endif
  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_DECL,
                     usingns_finishdecl_callback,
                     NULL);
}
