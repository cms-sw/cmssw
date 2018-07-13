/**
 * @file CheckerGccPlugins/src/gaudi_inheritance_plugin.cxx
 * @author scott snyder <snyder@bnl.gov>
 * @date Aug, 2014
 * @brief gcc plugin to check for obsolete Gaudi inheritance.
 *
 * This plugin warns about any classes which directly derive from
 * one of the classes Algorithm, AlgTool, or Service, suggesting
 * to use the AthenaBaseComps versions instead.
 *
 * The AthenaBaseComps classes themselves (AthAlgorithm, AthAlgTool,
 * AthService) are excluded from the check.
 *
 * Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
 */


#include <iostream>
#include "checker_gccplugins.h"
#include "tree.h"
#include "cp/cp-tree.h"
#include "diagnostic.h"
#include "print-tree.h"


namespace {


/**
 * @brief Helper to format the name of the type @c t.
 */
const char* type_name (tree t)
{
  unsigned int save = flag_sanitize;
  flag_sanitize = 0;
  const char* ret = type_as_string (t,
                                    TFF_PLAIN_IDENTIFIER +
                                    //TFF_UNQUALIFIED_NAME +
                                    TFF_NO_OMIT_DEFAULT_TEMPLATE_ARGUMENTS);
  flag_sanitize = save;
  return ret;
}


bool in_gaudi_source (tree t)
{
  location_t loc = DECL_SOURCE_LOCATION (TYPE_MAIN_DECL (t));
  std::string file = LOCATION_FILE(loc);

  if (file.find ("/GaudiSvc/")       != std::string::npos) return true;
  if (file.find ("/GaudiAlg/")       != std::string::npos) return true;
  if (file.find ("/GaudiCommonSvc/") != std::string::npos) return true;
  if (file.find ("/GaudiCoreSvc/")   != std::string::npos) return true;
  if (file.find ("/GaudiExamples/")  != std::string::npos) return true;
  if (file.find ("/GaudiGSL/")       != std::string::npos) return true;
  if (file.find ("/GaudiHive/")      != std::string::npos) return true;
  if (file.find ("/GaudiKernel/")    != std::string::npos) return true;
  if (file.find ("/GaudiMP/")        != std::string::npos) return true;
  if (file.find ("/GaudiMonitor/")   != std::string::npos) return true;
  if (file.find ("/GaudiPartProp/")  != std::string::npos) return true;
  if (file.find ("/GaudiProfiling/") != std::string::npos) return true;
  if (file.find ("/GaudiPython/")    != std::string::npos) return true;
  if (file.find ("/GaudiUtils/")     != std::string::npos) return true;
  if (file.find ("/PartPropSvc/")    != std::string::npos) return true;
  if (file.find ("/RootCnv/")        != std::string::npos) return true;

  return false;
}


// Check TYPE, report errors about TOP_TYPE.
void gaudi_check1 (tree type, tree top_type)
{
  const char* url = "<https://twiki.cern.ch/twiki/bin/view/AtlasComputing/ImprovingSoftware> and <https://twiki.cern.ch/twiki/bin/view/AtlasComputing/AthenaBaseComps>";

  // Skip if no base info.
  tree binfo = TYPE_BINFO (type);
  if (!binfo)
    return;

  // Loop over direct base classes.
  int n_baselinks = BINFO_N_BASE_BINFOS (binfo);
  for (int i=0; i < n_baselinks; i++) {
    tree base_binfo = BINFO_BASE_BINFO (binfo, i);
    if (BINFO_TYPE(base_binfo)) {
      std::string bname = type_name (BINFO_TYPE (base_binfo));

      // Warn if we're deriving directly from one of the Gaudi
      // component base classes.

      if (bname == "Algorithm") {
        location_t loc = DECL_SOURCE_LOCATION(TYPE_MAIN_DECL (top_type));
        warning_at (loc, 0, 
                    "%<%D%> derives directly from %<Algorithm%>; "
                    "should derive from %<AthAlgorithm%> instead and "
                    "use its methods.  (See %s.)",
                    top_type, url);
      }

      else if (bname == "AlgTool") {
        location_t loc = DECL_SOURCE_LOCATION(TYPE_MAIN_DECL (top_type));
        warning_at (loc, 0, 
                    "%<%D%> derives directly from %<AlgTool%>; "
                    "should derive from %<AthAlgTool%> instead and "
                    "use its methods.  (See %s.)",
                    top_type, url);
      }

      else if (bname == "Service") {
        location_t loc = DECL_SOURCE_LOCATION(TYPE_MAIN_DECL (top_type));
        warning_at (loc, 0, 
                    "%<%D%> derives directly from %<Service%>; "
                    "should derive from %<AthService%> instead and "
                    "use its methods.  (See %s.)",
                    top_type, url);
      }

      else if (bname.substr (0, 8) == "extends<") {
        gaudi_check1 (BINFO_TYPE (base_binfo), top_type);
      }
    }
  }
}


/**
 * @brief Gaudi inheritance checker.  Called on type definitions.
 * @param gcc_data The type that was defined.
 */
void type_callback (void* gcc_data, void* /*user_data*/)
{
  // Select complete named struct/class types.
  tree t = (tree)gcc_data;
  tree tt = t;//TYPE_MAIN_VARIANT(t);
  if (TREE_CODE(tt) != RECORD_TYPE ||
#if defined(GCC_VERSION) && (GCC_VERSION >= 7000)
      TYPE_UNNAMED_P(tt) ||
#else
      TYPE_ANONYMOUS_P(tt) ||
#endif
      !COMPLETE_TYPE_P(tt))
  {
    return;
  }

  if (in_gaudi_source (t)) return;

  // Core classes for which we should skip this test.
  std::string name = type_name(tt);
  if (name == "AthAlgorithm" ||
      name == "AthReentrantAlgorithm" ||
      name == "AthAlgTool" ||
      name == "AthService" ||
      name == "SegMemSvc" ||
      name == "ActiveStoreSvc" ||
      name == "StoreGateSvc" ||
      name == "SGImplSvc" ||
      name == "SG::HiveMgrSvc" ||
      name == "AddressProviderSvc" ||
      name == "Algtest" ||
      name == "Algtooltest" ||
      name == "Servtest" ||
      name == "ClassIDSvc" ||
      name == "TrigMessageSvc" ||
      name == "ConversionSvc" ||
      name == "MinimalEventLoopMgr" ||
      name == "DataSvc" ||
      name == "FileStagerAlg" ||
      name == "TestWhiteBoard" ||
      name == "ConditionsCleanerTest" )
    return;
  if (name.substr (0, 12) == "__shadow__::")
    return;

  gaudi_check1 (tt, tt);
}


} // Anonymous namespace


/**
 * @brief Register the Gaudi inheritance checker.  Called on type definitions.
 */
void init_gaudi_inheritance_checker (plugin_name_args* plugin_info)
{
  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_TYPE,
                     type_callback,
                     NULL);
}
