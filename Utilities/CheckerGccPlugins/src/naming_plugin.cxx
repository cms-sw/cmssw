/**
 * @file CheckerGccPlugins/src/naming_plugin.cxx
 * @author scott snyder <snyder@bnl.gov>
 * @date Aug, 2015
 * @brief Check ATLAS naming conventions.
 *
 * Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
 */


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


const char* url = "<https://twiki.cern.ch/twiki/bin/view/AtlasComputing/CheckerGccPlugins#naming_plugin>";


// Is DECL in the SG:: namespace?
bool in_sg_p (tree decl)
{
  tree ctx = DECL_CONTEXT (decl);
  if (ctx && !SCOPE_FILE_SCOPE_P (ctx) &&
      TREE_CODE (ctx) == NAMESPACE_DECL)
  {
    tree nsname_node = DECL_NAME (ctx);
    if (nsname_node) {
      const char* nsname = IDENTIFIER_POINTER (nsname_node);
      if (strcmp (nsname, "SG") == 0) {
        tree ctx2 = DECL_CONTEXT (ctx);
        if (ctx2 && SCOPE_FILE_SCOPE_P (ctx2)) {
          return true;
        }
      }
    }
  }
  return false;
}


bool handle_p (tree type)
{
  tree decl = TYPE_NAME (type);
  if (decl) {
    tree name_node = DECL_NAME (decl);
    if (name_node) {
      const char* name = IDENTIFIER_POINTER (name_node);
      if (strcmp (name, "ServiceHandle") == 0) return true;
      if (strcmp (name, "ToolHandle") == 0) return true;
    }
  }
  return false;
}


bool cap_p (char c)
{
  return (c >= 'A' && c <= 'Z');
}

bool digit_p (char c)
{
  return (c >= '0' && c <= '9');
}

bool allcaps_p (const char* s)
{
  while (*s) {
    if ( ! (cap_p (*s) || *s == '_' || digit_p (*s)) )
      return false;
    ++s;
  }
  return true;
}


// Check to see if we should ignore the type given by DECL.
bool ignore_decl_p1 (tree decl)
{
  static tree iauxstore_type = NULL_TREE;
  tree type = NULL_TREE;
  if (TYPE_P (decl)) {
    type = decl;
    decl = TYPE_NAME (decl);

    // Ignore classes deriving from SG::IAuxStore.
    if (TREE_CODE (type) == RECORD_TYPE && iauxstore_type) {
      tree base = lookup_base (type, iauxstore_type,
                               ba_any, NULL, tf_none);
      if (base != NULL_TREE && base != error_mark_node)
        return true;
    }
  }
  if (DECL_P (decl)) {
    tree name_node = DECL_NAME (decl);
    if (name_node) {
      const char* name = IDENTIFIER_POINTER (name_node);

      // Is this the decl for SG::IAuxStore?
      if (!iauxstore_type && strcmp (name, "IAuxStore") == 0) {
        if (in_sg_p (decl))
          iauxstore_type = type;
      }

      // Ignore compiler/library types.
      if (name[0] == '_' && name[1] == '_') return true;
      if (strcmp (name, "std") == 0) return true;

      // Ignore ROOT types -- but don't match names starting with TRT!
      if (strncmp (name, "TRT", 3) == 0)
        return false;
      if (name[0] == 'T' && cap_p (name[1]))
        return true;

      // Ignore GEANT types.
      if (name[0] == 'G' && name[1] == '4')
        return true;

      // Ignore some other libraries.
      if (strcmp (name, "HepGeom") == 0) return true;
      if (strcmp (name, "Eigen") == 0) return true;
      if (strcmp (name, "fastjet") == 0) return true;
      if (strcmp (name, "CLHEP") == 0) return true;
      if (strcmp (name, "boost") == 0) return true;
      if (strcmp (name, "pool") == 0) return true;
      if (strcmp (name, "ers") == 0) return true;
      if (strcmp (name, "dqm_core") == 0) return true;
      if (strcmp (name, "CTPdataformatVersion") == 0) return true;
      if (strcmp (name, "rapidjson") == 0) return true;
      if (strcmp (name, "OptimalJetFinder") == 0) return true;
      if (strcmp (name, "EventStorage") == 0) return true;
      if (strcmp (name, "hltinterface") == 0) return true;
      if (strcmp (name, "tbb") == 0) return true;
      if (strcmp (name, "testing") == 0) return true;

      // Ignore generated Qt code.
      if (startswith (name, "qt_")) return true;

      // Don't warn about _pN classes since we can't really change them.
      size_t len = IDENTIFIER_LENGTH (name_node);
      if (len >= 4) {
        const char* p = name + len - 1;
        if (digit_p (*p)) {
          while (p > name && digit_p(*p))
            --p;
          if (p > name && *p == 'p') {
            --p;
            if (p > name && *p == '_') {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}


// Check to see if we should ignore the type given by DECL.
// Checks outer scopes as well.
bool ignore_decl_p (tree decl)
{
  if (ignore_decl_p1 (decl)) return true;
  if (TYPE_P (decl))
    decl = TYPE_NAME (decl);
  tree ctx = DECL_CONTEXT (decl);
  while (ctx && !SCOPE_FILE_SCOPE_P (ctx)) {
    if (ignore_decl_p1 (ctx)) return true;
    if (DECL_P (ctx))
      ctx = DECL_CONTEXT (ctx);
    else if (TYPE_P (ctx))
      ctx = TYPE_CONTEXT (ctx);
    else
      break;
  }

  return false;
}


// Test FILE to see if it's one we should be checking.
bool in_atlas_source1 (const char* file)
{
  // Ignore generated dictionary files.
  if (strstr (file, "_gen.cxx") != 0) return false;
  if (strstr (file, "_gen.cpp") != 0) return false;
  if (strstr (file, "_rootcint.cxx") != 0) return false;
  if (strstr (file, "_rootcint.cpp") != 0) return false;
  if (strstr (file, "ReflexDict.cxx") != 0) return false;

  // Ignore generated qt files.
  if (strstr (file, "moc_") != 0) return false;
  if (strstr (file, "ui_") != 0) return false;

  // Otherwise always check files given by relative path.
  if (file[0] != '/') return true;

  // Ignore geant.
  if (strstr (file, "share/geant") != 0) return false;

  if (strstr (file, "/gtest/") != 0) return false;

  // Check files that appear to be part of ATLAS offline.
  if (strstr (file, "/AtlasAnalysis/") != 0) return true;
  if (strstr (file, "/AtlasConditions/") != 0) return true;
  if (strstr (file, "/AtlasCore/") != 0) return true;
  if (strstr (file, "/AtlasEvent/") != 0) return true;
  if (strstr (file, "/AtlasHLT/") != 0) return true;
  if (strstr (file, "/AtlasOffline/") != 0) return true;
  if (strstr (file, "/AtlasProduction/") != 0) return true;
  if (strstr (file, "/AtlasReconstruction/") != 0) return true;
  if (strstr (file, "/AtlasSimulation/") != 0) return true;
  if (strstr (file, "/AtlasTrigger/") != 0) return true;
  if (strstr (file, "/DetCommon/") != 0) return true;

  if (strstr (file, "/AtlasTest/") != 0) return true;
  if (strstr (file, "/Calorimeter/") != 0) return true;
  if (strstr (file, "/Commission/") != 0) return true;
  if (strstr (file, "/Control/") != 0) return true;
  if (strstr (file, "/Database/") != 0) return true;
  if (strstr (file, "/DataQuality/") != 0) return true;
  if (strstr (file, "/DetectorDescription/") != 0) return true;
  if (strstr (file, "/Event/") != 0) return true;
  if (strstr (file, "/External/CheckerGccPlugins/") != 0) return true;
  if (strstr (file, "/ForwardDetectors/") != 0) return true;
  if (strstr (file, "/Generators/") != 0) return true;
  if (strstr (file, "/graphics/") != 0) return true;
  if (strstr (file, "/InnerDetector/") != 0) return true;
  if (strstr (file, "/LArCalorimeter/") != 0) return true;
  if (strstr (file, "/LumiBlock/") != 0) return true;
  if (strstr (file, "/MagneticField/") != 0) return true;
  if (strstr (file, "/MuonSpectrometer/") != 0) return true;
  if (strstr (file, "/PhysicsAnalysis/") != 0) return true;
  if (strstr (file, "/Reconstruction/") != 0) return true;
  if (strstr (file, "/Simulation/") != 0) return true;
  if (strstr (file, "/TestBeam/") != 0) return true;
  if (strstr (file, "/TileCalorimeter/") != 0) return true;
  if (strstr (file, "/Tools/") != 0) return true;
  if (strstr (file, "/Tracking/") != 0) return true;
  if (strstr (file, "/Trigger/") != 0) return true;

  return false;
}


// Test to see if DECL is in a file we should be checking.
// Memoize the result.
bool in_atlas_source (tree decl)
{
  static hash_map<const char*, bool> seen_files;

  location_t loc = DECL_SOURCE_LOCATION (decl);
  if (loc == UNKNOWN_LOCATION)
    loc = input_location;
  const char* file = LOCATION_FILE (loc);
  bool existed = false;
  bool& flag = seen_files.get_or_insert (file, &existed);
  if (!existed)
    flag = in_atlas_source1 (file);
  return flag;
}


// Called after a declaration.
// Check local/global variable declarations here.
// (Don't check FIELD_DECL since we want to wait until the entire
// type is defined.)
void naming_finishdecl_callback (void* gcc_data, void* /*user_data*/)
{
  tree decl = (tree)gcc_data;
  if (TREE_CODE(decl) == FUNCTION_DECL) {
    if (!in_atlas_source (decl)) return;
    tree fnid = DECL_NAME (decl);
    const char* fnname = IDENTIFIER_POINTER(fnid);
    if (fnname[0] == 'm' && fnname[1] =='_') {
      warning_at (DECL_SOURCE_LOCATION (decl), 0,
                  "ATLAS coding standards require that the name of function %<%D%> not start with %<m_%>.", decl);
      CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (decl), url);
    }
  }
  else if (TREE_CODE(decl) == VAR_DECL) {
    if (!in_atlas_source (decl)) return;
    tree ctx = DECL_CONTEXT (decl);
    if (ctx && TREE_CODE (ctx) == RECORD_TYPE) return;
    if (ctx && DECL_P (ctx) && ignore_decl_p (ctx)) return;
    tree id = DECL_NAME(decl);
    if (id) {
      const char* name = IDENTIFIER_POINTER(id);
      if (name[0] == 'm' && name[1] == '_') {
        warning_at (DECL_SOURCE_LOCATION(decl), 0,
                    "ATLAS coding standards require that variable name %<%D%> not start with %<m_%>.", decl);
        CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (decl), url);
      }
      else if (name[0] == '_') {
        if (startswith (name, "_foreach_")) return;
        if (startswith (name, "_ZTS")) return;
        if (startswith (name, "_ZGV")) return;
        if (strcmp (name, "__dso_handle") == 0) return;
        if (strcmp (name, "_DYNAMIC") == 0) return;
        if (strcmp (name, "__FUNCTION__") == 0) return;
        if (strcmp (name, "__PRETTY_FUNCTION__") == 0) return;
        if (strcmp (name, "__func__") == 0) return;
        if (strcmp (name, "__for_range") == 0) return;
        if (strcmp (name, "__for_begin") == 0) return;
        if (strcmp (name, "__for_end") == 0) return;
        if (strcmp (name, "__tls_guard") == 0) return;

        // root dictionaries.
        if (strcmp (name, "__TheDictionaryInitializer") == 0) return;

        // in bfd header
        if (strcmp (name, "_bfd_std_section") == 0) return;

        // valgrind
        if (startswith (name, "_zzq_")) return;

        // Qt
        if (strcmp (name, "_container_") == 0) return;

        // ers
        if (strcmp (name, "__issue__") == 0) return;

        tree type = TREE_TYPE (decl);
        tree type_decl = TYPE_NAME (type);
        if (type_decl) {
          tree type_id = DECL_NAME (type_decl);
          if (type_id) {
            const char* type_name = IDENTIFIER_POINTER (type_id);
            if (type_name[0] == '_')
              return;
          }
        }
        warning_at (DECL_SOURCE_LOCATION(decl), 0,
                    "ATLAS coding standards require that variable name %<%D%> not start with %<_%>.", decl);
        CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (decl), url);
      }
#if 0
      // Don't enforce this for now
      else if (name[0] == 's' && name[1] == '_') {
        // Allow globals and statics with s_
        if (TREE_CODE (ctx) == FUNCTION_DECL && !TREE_STATIC (decl)) {
          warning_at (DECL_SOURCE_LOCATION(decl), 0,
                      "ATLAS coding standards require that variable name %<%D%> not start with %<s_%>.", decl);
          CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (decl), url);
        }
      }
#endif
    }
  }
}


// Called after a type declaration.
// Check class/struct members here.
void naming_finishtype_callback (void* gcc_data, void* /*user_data*/)
{
  tree type = (tree)gcc_data;
  if (TREE_CODE (type) == RECORD_TYPE) {

    // Skip checking `aggregate' types --- essentially POD types.
    // However, this test is not reliable for template types --- we get called
    // with the template itself.  The template will not have had the
    // not_aggregate flag set, so CP_AGGREGATE_TYPE will always be true.
    if (CP_AGGREGATE_TYPE_P (type) && !CLASSTYPE_TEMPLATE_INFO (type)) return;

    tree decl = TYPE_NAME (type);
    if (!in_atlas_source (decl)) return;
    if (ignore_decl_p (type)) return;
    for (tree f = TYPE_FIELDS (type); f; f = DECL_CHAIN (f)) {
      if (TREE_CODE (f) == FIELD_DECL || TREE_CODE (f) == VAR_DECL) {
        if (TREE_PRIVATE (f) || TREE_PROTECTED (f)) {
          tree id = DECL_NAME (f);
          if (!id) return;
          const char* name = IDENTIFIER_POINTER(id);
          const char* access = TREE_PRIVATE(f) ? "private" : "protected";
          if (name[0] == 'm' && name[1] == '_') {
            // ok
          }
          else {
            if (TREE_STATIC(f)) {
              if (name[0] == 's' && name[1] == '_') {
                // ok
              }
              else if (strcmp (name, "fgIsA") == 0) {
                // ok -- root exceptions.
              }
              else if (strcmp (name, "test_info_") == 0) {
                // ok -- gtest exception.
              }
              else if ((TREE_READONLY (f) || TREE_CONSTANT (f)) &&
                       (allcaps_p (name) ||
                        (name[0] == 'k' && cap_p(name[1]))))
              {
                // allow static const names to be in all-caps
                // (not in the coding guidelines, but common)
              }
              else {
                warning_at (DECL_SOURCE_LOCATION(f), 0,
                            "ATLAS coding standards require that the name of %s static member %<%D%> start with %<m_%> or %<s_%>.", access, f);
                CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (f), url);
              }
            }
            else {
              if ((TREE_CODE (TREE_TYPE (f)) == POINTER_TYPE ||
                   handle_p (TREE_TYPE (f))) &&
                  name[0] == 'p' && name[1] == '_')
              {
                // Allow member pointers to start with p_.
                // Not in the coding guidelines, but common.
              }
              else {
                warning_at (DECL_SOURCE_LOCATION(f), 0,
                            "ATLAS coding standards require that the name of %s non-static member %<%D%> start with %<m_%>.", access, f);
                CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (f), url);
              }
            }
          }
        }
      }
    }
  }
}


// Called after parsing a function.
// Check function arguments here.
void naming_pregen_callback (void* gcc_data, void* /*user_data*/)
{
  tree ast = (tree)gcc_data;
  if (TREE_CODE (ast) == FUNCTION_DECL) {
    if (!in_atlas_source (ast)) return;
    if (ignore_decl_p (ast)) return;
    for (tree arg = DECL_ARGUMENTS (ast); arg; arg = DECL_CHAIN (arg)) {
      if (TREE_CODE (arg) == PARM_DECL) {
        tree id = DECL_NAME (arg);
        if (id) {
          const char* name = IDENTIFIER_POINTER(id);
          if (name[0] == 'm' && name[1] == '_') {
            warning_at (DECL_SOURCE_LOCATION (arg), 0,
                        "ATLAS coding standards require that the name of function parameter %<%D%> not start with %<m_%>.", arg);
            CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (arg), url);
          }
          else if (name[0] == '_') {
            if (strcmp (name, "__in_chrg") == 0) return;
            warning_at (DECL_SOURCE_LOCATION (arg), 0,
                        "ATLAS coding standards require that the name of function parameter %<%D%> not start with %<_%>.", arg);
            CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (arg), url);
          }
#if 0
          // Don't enforce this for now.
          else if (name[0] == 's' && name[1] == '_') {
            warning_at (DECL_SOURCE_LOCATION (arg), 0,
                        "ATLAS coding standards require that the name of function parameter %<%D%> not start with %<s_%>.", arg);
            CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION (arg);
          }
#endif
        }
      }
    }
  }
}


} // anonymous namespace


void init_naming_checker (plugin_name_args* plugin_info)
{
  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_DECL,
                     naming_finishdecl_callback,
                     NULL);
  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_TYPE,
                     naming_finishtype_callback,
                     NULL);
  register_callback (plugin_info->base_name,
                     PLUGIN_PRE_GENERICIZE,
                     naming_pregen_callback,
                     NULL);
}
