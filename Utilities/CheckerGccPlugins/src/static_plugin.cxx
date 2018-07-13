/**
 * @file CheckerGccPlugins/src/static_plugin.cxx
 * @author scott snyder <snyder@bnl.gov>
 * @date Aug, 2015
 * @brief Check for uses of `static' that might spoil thread-safety.
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
#include "stringpool.h"
#include "attribs.h"


using namespace CheckerGccPlugins;


namespace {


void static_finishdecl_callback (void* gcc_data, void* /*user_data*/)
{
  tree decl = (tree)gcc_data;
  if (TREE_CODE (decl) == VAR_DECL &&
      TREE_STATIC (decl) &&
      !TREE_CONSTANT (decl) &&
      !TREE_READONLY (decl) &&
      !DECL_THREAD_LOCAL_P (decl) &&
      !lookup_attribute ("thread_safe", DECL_ATTRIBUTES (decl)))
  {
    tree name_node = DECL_NAME (decl);
    const char* name = IDENTIFIER_POINTER (name_node);
    if (name[0] == '_' && name[1] == '_') return;
    if (strncmp(name, "_ZGV", 4) == 0) return;
    tree ctx = DECL_CONTEXT (decl);
    tree ctxdecl = ctx;
    if (TYPE_P (ctxdecl))
      ctxdecl = TYPE_NAME (ctxdecl);
    if (!check_thread_safety_p (decl) &&
        !check_thread_safety_location_p (DECL_SOURCE_LOCATION (decl)))
    {
      return;
    }

    if (ctx && TREE_CODE (ctx) == FUNCTION_DECL) {
      warning_at (DECL_SOURCE_LOCATION (decl), 0,
                  "Use of non-const local static variable %<%D%> may not be thread-safe.",
                  decl);
    }
    else {
      warning_at (DECL_SOURCE_LOCATION (decl), 0,
                  "Use of non-const global static variable %<%D%> may not be thread-safe.",
                  decl);
    }
  }
}


} // anonymous namespace


void init_static_checker (plugin_name_args* plugin_info)
{
  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_DECL,
                     static_finishdecl_callback,
                     NULL);
}
