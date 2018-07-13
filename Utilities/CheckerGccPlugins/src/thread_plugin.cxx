/**
 * @file CheckerGccPlugins/src/thread_plugin.cxx
 * @author scott snyder <snyder@bnl.gov>
 * @date Sep, 2015
 * @brief Check for possible thread-safety violations.
 *
 * Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
 */

// FIXME: can i unify pointer_is_const_arg / expr_from_arg_p / expr_from_static_p / value_from_struct?
// FIXME: are the recursion conditions in the above sufficient?

#include <unordered_set>
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
#include <vector>
#include <unordered_map>
#include "stringpool.h"
#include "attribs.h"


using namespace CheckerGccPlugins;


namespace {


//*******************************************************************************

// stdlib functions assumed to be safe unless listed here.
// This list collected from those tagged as MT-Unsafe on the linux man pages.
const std::unordered_set<std::string> unsafe_stdlib_fns {
  "bindresvport", // glibc < 2.17
  "crypt",
  "cuserid",
  "drand48",
  "ecvt",
  "encrypt",
  "endfsent",
  "endgrent",
  "endpwent",
  "endttyent",
  "endusershell",
  "erand48",
  "ether_aton",
  "ether_ntoa",
  "exit",
  "fcloseall",
  "fcvt",
  "fmtmsg", // for glibc < 2.16
  "gamma",
  "gammaf",
  "gammal",
  "getdate",
  "getfsent",
  "getfsfile",
  "getfsspec",
  "getgrent",
  "getgrgid",
  "getgrname",
  "getlogin",
  "getlogin_r",
  "getopt",
  "getopt_long",
  "getopt_long_only",
  "getpass",
  "getpwent",
  "getpwnam",
  "getpwuid",
  "getttyent",
  "getttynam",
  "getusershell",
  "hcreate",
  "hdestroy",
  "hsearch",
  "jrand48",
  "l64a",
  "lcong48",
  "localeconv",
  "lrand48",
  "mblen",
  "mbrlen",
  "mbrtowc",
  "mbtowc",
  "mrand48",
  "mtrace",
  "muntrace",
  "nrand48",
  "ptsname",
  "putenv",
  "qecvt",
  "qfcvt",
  "re_comp",
  "re_exec",
  "readdir",
  "rexec",
  "rexec_af",
  "seed48",
  "setenv",
  "setfsent",
  "setgrent",
  "setkey",
  "setpwent",
  "setttyent",
  "setusershell",
  "siginterrupt",
  "srand48",
  "strerror",
  "strtok",
  "tmpnam",
  "ttyname",
  "ttyslot",
  "unsetenv",
  "wctomb",

  "asctime",
  "ctime",
  "gmtime",
  "localtime",
};


const char* url = "<https://twiki.cern.ch/twiki/bin/view/AtlasComputing/CheckerGccPlugins#thread_plugin>";


const pass_data thread_pass_data =
{
  GIMPLE_PASS, /* type */
  "thread", /* name */
  0, /* optinfo_flags */
  TV_NONE, /* tv_id */
  0, /* properties_required */
  0, /* properties_provided */
  0, /* properties_destroyed */
  0, /* todo_flags_start */
  0  /* todo_flags_finish */
};


class thread_pass : public gimple_opt_pass
{
public:
  thread_pass (gcc::context* ctxt)
    : gimple_opt_pass (thread_pass_data, ctxt)
  { 
#if GCC_VERSION < 7000
    graph_dump_initialized = false;
#endif
 }

  virtual unsigned int execute (function* fun) override
  { return thread_execute(fun); }

  unsigned int thread_execute (function* fun);

  virtual opt_pass* clone() override { return new thread_pass(*this); }
};


//*******************************************************************************
// Attribute handling.
//


enum Attribute {
  ATTR_CHECK_THREAD_SAFETY,
  ATTR_THREAD_SAFE,
  ATTR_NOT_THREAD_SAFE,
  ATTR_NOT_REENTRANT,
  ATTR_NOT_CONST_THREAD_SAFE,
  ATTR_ARGUMENT_NOT_CONST_THREAD_SAFE,
  NUM_ATTR
};


const char* attr_names[NUM_ATTR] = {
  "check_thread_safety",
  "thread_safe",
  "not_thread_safe",
  "not_reentrant",
  "not_const_thread_safe",
  "argument_not_const_thread_safe",
};


typedef uint32_t Attributes_t;


Attributes_t get_attributes (tree decl)
{
  Attributes_t mask = 0;
  for (unsigned i=0; i < NUM_ATTR; i++) {
    if (lookup_attribute (attr_names[i], DECL_ATTRIBUTES (decl))) {
      mask |= (1 << i);
    }
  }
  return mask;
}


bool has_attrib (Attributes_t attribs, Attribute attr)
{
  return (attribs & (1<<attr)) != 0;
}


typedef std::unordered_map<tree, std::pair<Attributes_t, location_t> > saved_attribs_t;
saved_attribs_t saved_attribs;
void check_attrib_consistency (Attributes_t attribs, tree decl)
{
  saved_attribs_t::iterator it = saved_attribs.find (decl);
  if (it == saved_attribs.end())
    saved_attribs[decl] = std::make_pair (attribs, DECL_SOURCE_LOCATION (decl));
  else if (attribs != it->second.first) {
    warning_at (DECL_SOURCE_LOCATION (decl), 0,
                "Inconsistent attributes between declaration and definition of function %<%D%>.",
                decl);
    inform (it->second.second, "Declaration is here:");
    for (unsigned i=0; i < NUM_ATTR; i++) {
      Attribute ia = static_cast<Attribute> (i);
      if (has_attrib (attribs, ia) && !has_attrib (it->second.first, ia))
        inform (DECL_SOURCE_LOCATION (decl),
                "Definition has %<%s%> but declaration does not.",
                attr_names[i]);
      else if (!has_attrib (attribs, ia) && has_attrib (it->second.first, ia))
        inform (it->second.second,
                "Declaration has %<%s%> but definition does not.",
                attr_names[i]);
    }
    CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION(decl), url);
  }
}


//*******************************************************************************


bool startswith (const char* haystack, const char* needle)
{
  while (*needle) {
    if (*haystack++ != *needle++) return false;
  }
  return true;
}


// If EXPR is a COMPONENT_REF, array reference, etc, return the complete
// object being referenced.
tree get_inner (tree expr)
{
  if (expr && handled_component_p (expr)) {
#if GCC_VERSION < 8000    
    HOST_WIDE_INT pbitsize, pbitpos;
#else
    poly_int64 pbitsize, pbitpos;
#endif
    tree poffset;
    machine_mode pmode;
    int punsignedp, pvolatilep;
    int preversep;
    expr = get_inner_reference (expr,
                                &pbitsize,
                                &pbitpos, &poffset,
                                &pmode, &punsignedp,
                                &preversep,
                                &pvolatilep
#if GCC_VERSION < 7000                                
                                , false
#endif
                                );
  }
  return expr;
}


bool has_thread_safe_attrib (tree decl)
{
  return lookup_attribute ("thread_safe", DECL_ATTRIBUTES (decl));
}


// Is DECL in the std:: namespace?
bool is_in_std (tree decl)
{
  for (tree ctx = DECL_CONTEXT (decl); ctx;
       ctx = DECL_P (ctx) ? DECL_CONTEXT (ctx) : TYPE_CONTEXT (ctx))
  {
    if (TREE_CODE (ctx) != NAMESPACE_DECL) continue;
    tree nsname = DECL_NAME (ctx);
    if (!nsname) continue;
    if (strcmp (IDENTIFIER_POINTER (nsname), "std") == 0) return true;
  }
  return false;
}


// Is TYPE std::mutex?
bool is_mutex (tree type)
{
  const char* name = type_as_string (type,
                                     TFF_PLAIN_IDENTIFIER +
                                     TFF_SCOPE +
                                     TFF_CHASE_TYPEDEF);
  if (strcmp (name, "std::mutex") == 0) return true;
  if (strcmp (name, "std::shared_timed_mutex") == 0) return true;
  if (strcmp (name, "std::recursive_mutex") == 0) return true;
  if (strcmp (name, "boost::shared_mutex") == 0) return true;
  if (strcmp (name, "TVirtualMutex") == 0) return true;
  return false;
}


// Is TYPE a std::atomic of a fundamental type?
bool is_atomic (tree type)
{
  if (TREE_CODE (type) != RECORD_TYPE) return false;
  const char* name = type_as_string (type,
                                     TFF_PLAIN_IDENTIFIER +
                                     TFF_SCOPE +
                                     TFF_CHASE_TYPEDEF);
  if (strncmp (name, "const ", 6) == 0)
    name += 6;

  if (strncmp (name, "std::atomic<", 12) != 0 &&
      strncmp (name, "std::__atomic_base<", 19) != 0)
    return false;

  tree targs = CLASSTYPE_TI_ARGS (type);
  if (!targs) return false;
  if (NUM_TMPL_ARGS (targs) < 1) return false;
  tree arg = TMPL_ARG (targs, 1, 0);

  if (INTEGRAL_TYPE_P (arg)) return true;
  if (SCALAR_FLOAT_TYPE_P (arg)) return true;
  if (POINTER_TYPE_P (arg)) return true;
  return false;
}


// Is TYPE a thread-local type?
bool is_thread_local (tree type)
{
  const char* name = type_as_string (type,
                                     TFF_PLAIN_IDENTIFIER +
                                     TFF_SCOPE +
                                     TFF_CHASE_TYPEDEF);
  if (strncmp (name, "boost::thread_specific_ptr<", 27) == 0) return true;
  return false;
}


// Does EXPR represent a static object that's not marked as thread-safe?
bool static_p (tree expr)
{
  if (!expr) return false;
  if (!TREE_STATIC (expr) && !(DECL_P (expr) && DECL_EXTERNAL (expr))) {
    return false;
  }

  if (DECL_P (expr)) {
    if (TREE_CODE (expr) == FUNCTION_DECL) return false;
    if (DECL_THREAD_LOCAL_P (expr)) return false;
    if (has_thread_safe_attrib (expr)) return false;
    if (is_in_std (expr)) return false;
    location_t loc = DECL_SOURCE_LOCATION (expr);
    if (loc != UNKNOWN_LOCATION) {
      const char* file = LOCATION_FILE (loc);
      if (startswith (file, "/usr/include/")) return false;
    }

    tree id = DECL_NAME (expr);
    const char* name = IDENTIFIER_POINTER (id);
    if (strcmp (name, "gInterpreterMutex") == 0) return false;
  }

  return true;
}


bool const_p (tree expr)
{
  if (TREE_CONSTANT (expr)) return true;
  if (TREE_READONLY (expr)) return true;
  tree typ = TREE_TYPE (expr);
  if (typ && TREE_READONLY (typ)) return true;

  while (TREE_CODE (typ) == ARRAY_TYPE) {
    typ = TREE_TYPE (typ);
    if (typ && TREE_READONLY (typ)) return true;
  }

  return false;
}


void check_mutable (tree expr, gimplePtr stmt, function* fun, const char* what)
{
  while (true) {
    switch (TREE_CODE (expr)) {
    case COMPONENT_REF:
      if (TYPE_READONLY(TREE_TYPE(TREE_OPERAND(expr, 0))) &&
          DECL_MUTABLE_P (TREE_OPERAND (expr, 1)))
      {
        tree op = TREE_OPERAND (expr, 1);
        tree optype = TREE_TYPE (op);
        if (has_thread_safe_attrib (op)) return;
        if (is_mutex (optype)) return;
        if (is_atomic (optype)) return;
        if (is_thread_local (optype)) return;
        warning_at (gimple_location (stmt), 0,
                    "%s %<%E%> of type %<%T%> within thread-safe function %<%D%>; may not be thread-safe.",
                    what, op, optype, fun->decl);
        CheckerGccPlugins::inform_url (gimple_location (stmt), url);
        return;
      }
      expr = TREE_OPERAND (expr, 0);
      break;

    case ARRAY_REF:
    case ARRAY_RANGE_REF:
    case BIT_FIELD_REF:
    case REALPART_EXPR:
    case IMAGPART_EXPR:
    case VIEW_CONVERT_EXPR:
      expr = TREE_OPERAND (expr, 0);
      break;

    default:
      return;
    }
  }
}


// Test to see if the type of X is a struct/class with at least one
// not_const_thread_safe member.  Returns that member or NULL_TREE.
tree has_not_const_thread_safe_member (tree x)
{
  tree typ = TREE_TYPE (x);
  if (TREE_CODE (typ) != RECORD_TYPE) return NULL_TREE;
  if (CP_AGGREGATE_TYPE_P (typ)) return NULL_TREE;
  typ = TYPE_MAIN_VARIANT (typ);

#if GCC_VERSION >= 8000
  for (tree meth = TYPE_FIELDS (typ); meth; meth = TREE_CHAIN (meth)) {
    if (TREE_CODE (meth) != FUNCTION_DECL) continue;
#else
  for (tree meth = TYPE_METHODS (typ); meth; meth = TREE_CHAIN (meth)) {
#endif
    if (DECL_ARTIFICIAL (meth)) continue;
    if (has_attrib (get_attributes (meth), ATTR_NOT_CONST_THREAD_SAFE)) {
      return meth;
    }
  }
  return NULL_TREE;
}


// Check for direct use of a static value.
void check_direct_static_use (Attributes_t attribs, gimplePtr stmt, function* fun)
{
  if (has_attrib (attribs, ATTR_NOT_REENTRANT))
    return;

  size_t nop = gimple_num_ops (stmt);
  for (size_t i = 0; i < nop; i++) {
    tree op = gimple_op (stmt, i);

    tree optest = get_inner (op);
    if (static_p (optest)) {
      if (!const_p (optest)) {
        warning_at (gimple_location (stmt), 0,
                    "Use of static expression %<%E%> of type %<%T%> within function %<%D%> may not be thread-safe.",
                    op, TREE_TYPE(op),fun->decl);
        if (DECL_P (optest)) {
          inform (DECL_SOURCE_LOCATION (optest),
                  "Declared here:");
        }
        CheckerGccPlugins::inform_url (gimple_location (stmt), url);
      }
      else if (tree meth = has_not_const_thread_safe_member (optest)) {
        warning_at (gimple_location (stmt), 0,
                    "Use of const static expression %<%E%> of type %<%T%> within function %<%D%> may not be thread-safe.",
                    op, TREE_TYPE(op), fun->decl);
        if (DECL_P (optest)) {
          inform (DECL_SOURCE_LOCATION (optest),
                  "Declared here:");
        }
        inform (DECL_SOURCE_LOCATION (meth),
                "Because it has a method declared not_const_thread_safe:");
        CheckerGccPlugins::inform_url (gimple_location (stmt), url);
      }
    }
  }
}


// Check for assigning from an address of a static object
// into a pointer/reference, or for discarding const.
//   OP: operand to check
//   STMT: gimple statement being checked
//   FUN: function being checked
void check_assign_address_of_static (Attributes_t attribs,
                                     tree op,
                                     gimplePtr stmt,
                                     function* fun)
{
  if (op && TREE_CODE (op) == ADDR_EXPR) {
    while (op && TREE_CODE (op) == ADDR_EXPR)
      op = TREE_OPERAND (op, 0);
  }

  tree optest = get_inner (op);
  if (static_p (optest) && !has_attrib (attribs, ATTR_NOT_REENTRANT)) {
    if (!const_p (optest)) {
      warning_at (gimple_location (stmt), 0,
                  "Pointer or reference bound to static expression %<%E%> of type %<%T%> within function %<%D%>; may not be thread-safe.",
                  op, TREE_TYPE(op), fun->decl);
      if (DECL_P (optest)) {
        inform (DECL_SOURCE_LOCATION (optest),
                "Declared here:");
      }
      CheckerGccPlugins::inform_url (gimple_location (stmt), url);
    }
    else if (tree meth = has_not_const_thread_safe_member (optest)) {
      warning_at (gimple_location (stmt), 0,
                  "Use of const static expression %<%E%> of type %<%T%> within function %<%D%> may not be thread-safe.",
                  op, TREE_TYPE(op), fun->decl);
      if (DECL_P (optest)) {
        inform (DECL_SOURCE_LOCATION (optest),
                "Declared here:");
      }
      inform (DECL_SOURCE_LOCATION (meth),
              "Because it has a method declared not_const_thread_safe:");
      CheckerGccPlugins::inform_url (gimple_location (stmt), url);
    }
  }
}


// Test to see if a pointer value comes directly or indirectly from
// a const pointer function argument.
tree pointer_is_const_arg (tree val)
{
  //fprintf (stderr, "pointer_is_const_arg_p\n");
  //debug_tree(val);
  tree valtest = get_inner (val);
  tree valtype = TREE_TYPE (valtest);
  if (!POINTER_TYPE_P(valtype) || !TYPE_READONLY (TREE_TYPE (valtype)))
    return NULL_TREE;

  if (TREE_CODE (val) == ADDR_EXPR)
    val = TREE_OPERAND (val, 0);
  if (TREE_CODE (val) == COMPONENT_REF)
    val = get_inner (val);
  if (TREE_CODE (val) == MEM_REF)
    val = TREE_OPERAND (val, 0);

  if (TREE_CODE (val) != SSA_NAME) return NULL_TREE;
  if (SSA_NAME_VAR (val) && TREE_CODE (SSA_NAME_VAR (val)) == PARM_DECL) return val;

  gimplePtr stmt = SSA_NAME_DEF_STMT (val);
  if (!stmt) return NULL_TREE;
  //debug_gimple_stmt (stmt);
  //fprintf (stderr, "code %s\n", get_tree_code_name(gimple_expr_code(stmt)));
  
  if (is_gimple_assign (stmt) && (gimple_expr_code(stmt) == VAR_DECL ||
                                  gimple_expr_code(stmt) == PARM_DECL ||
                                  gimple_expr_code(stmt) == POINTER_PLUS_EXPR ||
                                  gimple_expr_code(stmt) == ADDR_EXPR))
  {
    //fprintf (stderr, "recurse\n");
    return pointer_is_const_arg (gimple_op(stmt, 1));
  }
  else if (gimple_code (stmt) == GIMPLE_PHI) {
    size_t nop = gimple_num_ops (stmt);
    for (size_t i = 0; i < nop; i++) {
      tree op = gimple_op (stmt, i);
      tree ret = pointer_is_const_arg (op);
      if (ret) return ret;
    }
  }
  return NULL_TREE;
}


void warn_about_discarded_const (Attributes_t attribs,
                                 tree expr,
                                 gimplePtr stmt,
                                 function* fun)
{
  tree parm = pointer_is_const_arg (expr);
  if (parm) {
    if (has_attrib (attribs, ATTR_ARGUMENT_NOT_CONST_THREAD_SAFE)) return;
    if (expr == parm) {
      warning_at (gimple_location (stmt), 0,
                  "Const discarded from expression %<%E%> of type %<%T%> within function %<%D%>; may not be thread-safe.",
                  expr, TREE_TYPE(expr), fun->decl);
    }
    else {
      warning_at (gimple_location (stmt), 0,
                  "Const discarded from expression %<%E%> of type %<%T%> (deriving from parameter %<%E%>) within function %<%D%>; may not be thread-safe.",
                  expr, TREE_TYPE(expr), parm, fun->decl);
    }
  }
  else {
    if (has_attrib (attribs, ATTR_NOT_CONST_THREAD_SAFE)) return;
    warning_at (gimple_location (stmt), 0,
                "Const discarded from expression %<%E%> of type %<%T%> within function %<%D%>; may not be thread-safe.",
                expr, TREE_TYPE(expr), fun->decl);
  }
  CheckerGccPlugins::inform_url (gimple_location (stmt), url);
}


// LHS is the def of an assignment.  Check if it has been declared
// as thread_safe.
bool is_lhs_marked_thread_safe (tree lhs)
{
  if (TREE_CODE (lhs) == VAR_DECL) {
    return has_thread_safe_attrib (lhs);
  }

  if (TREE_CODE (lhs) != SSA_NAME) return false;

  tree def = lhs;

  if (!SSA_NAME_VAR (def)) {
    // For something like
    //  int* store [[gnu::thread_safe]] = const_cast<int*> (foo());
    // we can get
    //
    //  _1 = foo();
    //  store_2 = _1;
    //
    // so that the attribute isn't visible in the first stmt.
    // Check for a single immediate use.
    gimple* stmt;
    use_operand_p use_p;
    if (single_imm_use (lhs, &use_p, &stmt)) {
      if (def_operand_p def_p = SINGLE_SSA_DEF_OPERAND(stmt, SSA_OP_ALL_DEFS)) {
        def = get_def_from_ptr (def_p);
      }
    }
  }

  if (SSA_NAME_VAR (def)) {
    return has_thread_safe_attrib (SSA_NAME_VAR (def));
  }

  return false;
}


// Called when LHS does not have const type.
void check_discarded_const (Attributes_t attribs,
                            tree op,
                            tree lhs,
                            gimplePtr stmt,
                            function* fun)
{
  tree optest = get_inner (op);
  tree optype = TREE_TYPE (optest);
  if (POINTER_TYPE_P(optype) && TYPE_READONLY (TREE_TYPE (optype))) {
    if (!is_lhs_marked_thread_safe (lhs)) {
      // Allow const_cast if LHS is explicitly marked thread_safe.
      warn_about_discarded_const (attribs, op, stmt, fun);
    }
  }
}


void check_assignments (Attributes_t attribs, gimplePtr stmt, function* fun)
{
  if (gimple_code (stmt) != GIMPLE_ASSIGN) return;
  size_t nop = gimple_num_ops (stmt);
  if (nop < 2) return;

  tree lhs = gimple_op (stmt, 0);
  if (!lhs) return;

  check_mutable (lhs, stmt, fun, "Setting mutable field");

  // Is the LHS a pointer/ref?
  tree lhs_type = TREE_TYPE (lhs);
  if (!POINTER_TYPE_P (lhs_type)) return;
  bool lhs_const = TYPE_READONLY (TREE_TYPE (lhs_type));

  // Does RHS point to something static, or is it const or mutable?
  for (size_t i = 1; i < nop; i++) {
    tree op = gimple_op (stmt, i);

    // Check for discarding const if LHS is non-const.
    if (!lhs_const)
      check_discarded_const (attribs, op, lhs, stmt, fun);
    
    check_assign_address_of_static (attribs, op, stmt, fun);

    if (op && TREE_CODE (op) == ADDR_EXPR) {
      while (op && TREE_CODE (op) == ADDR_EXPR)
        op = TREE_OPERAND (op, 0);
    }

    if (!lhs_const)
      check_mutable (op, stmt, fun, "Taking non-const reference to mutable field");
  }
}


// Warn about checked function calling unchecked function,
// unless known to be ok.
void check_thread_safe_call (Attributes_t attribs,
                             tree fndecl,
                             gimplePtr stmt,
                             function* fun)
{
  if (check_thread_safety_p (fndecl)) return;
  std::string fnname = decl_as_string (fndecl, TFF_SCOPE + TFF_NO_FUNCTION_ARGUMENTS);

  bool unsafe_stdlib = false;
  location_t loc = DECL_SOURCE_LOCATION (fndecl);
  if (loc != UNKNOWN_LOCATION) {
    const char* file = LOCATION_FILE (loc);
    if (startswith (file, "/usr/include/")) {
      // Assume functions from the system library are ok unless we know
      // they aren't.
      if (unsafe_stdlib_fns.count (fnname) == 0)
        return;
      unsafe_stdlib = true;
    }
    if (is_in_std (fndecl)) return;

    // Don't warn about calls to Gaudi.
    if (strstr (file, "/Gaudi") != nullptr)
      return;

    // Or ROOT.
    if (strstr (file, "/ROOT/") != nullptr || strstr (file, "/rootsys/") != nullptr)
      return;

    // Or CORAL.
    if (strstr (file, "/CORAL") != nullptr)
      return;
  }

  if (startswith (fnname.c_str(), "__")) return;
  if (startswith (fnname.c_str(), "boost::")) return;
  if (fnname == "operator new") return;
  if (fnname == "operator new []") return;
  if (fnname == "operator delete") return;
  if (fnname == "operator delete []") return;
  if (fnname == "_mm_pause") return;

  // ROOT functions.
  if (fnname == "operator<<") {
    std::string fnname_full = decl_as_string (fndecl, TFF_SCOPE);
    if (startswith (fnname_full.c_str(), "operator<<(TBuffer")) return;
  }
  if (fnname == "operator>>") {
    std::string fnname_full = decl_as_string (fndecl, TFF_SCOPE);
    if (startswith (fnname_full.c_str(), "operator>>(TBuffer")) return;
  }
  if (fnname == "ROOT::GenerateInitInstanceLocal") return;

  if (unsafe_stdlib || has_attrib (attribs, ATTR_NOT_THREAD_SAFE)) {
    warning_at (gimple_location (stmt), 0,
                "Non-thread-safe function %<%D%> called from thread-safe function %<%D%>; may not be thread-safe.",
                fndecl, fun->decl);
    inform (loc, "Declared here:");
    CheckerGccPlugins::inform_url (gimple_location (stmt), url);
  }
  else {
#if 0
    warning_at (gimple_location (stmt), 0,
                "Unchecked function %<%D%> called from thread-safe function %<%D%>; may not be thread-safe.",
                fndecl, fun->decl);
    inform (loc, "Declared here:");
    CheckerGccPlugins::inform_url (gimple_location (stmt), url);
#endif
  }
}


// A function calling a not_reentrant function must be not_reentrant.
void check_not_reentrant_call (Attributes_t attribs,
                               Attributes_t fnattribs,
                               tree fndecl,
                               gimplePtr stmt,
                               function* fun)
{
  if (has_attrib (fnattribs, ATTR_NOT_REENTRANT) &&
      !has_attrib (attribs, ATTR_NOT_REENTRANT))
  {
    warning_at (gimple_location (stmt), 0,
                "Function %<%D%> calling not_reentrant function %<%D%> must also be not_reentrant.",
                fun->decl, fndecl);
    CheckerGccPlugins::inform_url (gimple_location (stmt), url);
  }
}


// A const member function calling a not_const_thread_safe on the same
// object must also be not_const_thread_safe.
void check_not_const_thread_safe_call (Attributes_t attribs,
                                       Attributes_t fnattribs,
                                       tree fndecl,
                                       gimplePtr stmt,
                                       function* fun)
{
  if (!DECL_CONST_MEMFUNC_P (fun->decl)) return;
  if (!has_attrib (fnattribs, ATTR_NOT_CONST_THREAD_SAFE)) return;
  if (has_attrib (attribs, ATTR_NOT_CONST_THREAD_SAFE)) return;
  if (gimple_call_num_args (stmt) < 1) return;
  tree arg0 = gimple_call_arg (stmt, 0);
  if (TREE_CODE (arg0) != SSA_NAME) return;
  tree var = SSA_NAME_VAR (arg0);
  if (!var) return;
  if (TREE_CODE (var) != PARM_DECL) return;
  if (DECL_NAME (var) != this_identifier) return;

  warning_at (gimple_location (stmt), 0,
              "Const member function %<%D%> calling not_const_thread_safe member function %<%D%> with same object must also be not_const_thread_safe.",
              fun->decl, fndecl);
  CheckerGccPlugins::inform_url (gimple_location (stmt), url);
}


// Check to see if VAL derives from a function argument.
bool expr_from_arg_p (tree val)
{
  //debug_tree(val);
  if (TREE_CODE (val) == ADDR_EXPR)
    val = TREE_OPERAND (val, 0);
  if (TREE_CODE (val) == COMPONENT_REF)
    val = get_inner (val);
  if (TREE_CODE (val) == MEM_REF)
    val = TREE_OPERAND (val, 0);

  if (TREE_CODE (val) != SSA_NAME) return false;
  if (SSA_NAME_VAR (val) && TREE_CODE (SSA_NAME_VAR (val)) == PARM_DECL) return true;

  gimplePtr stmt = SSA_NAME_DEF_STMT (val);
  if (!stmt) return false;
  //debug_gimple_stmt (stmt);
  //fprintf (stderr, "code %s\n", get_tree_code_name(gimple_expr_code(stmt)));

  if (is_gimple_assign (stmt) && (gimple_expr_code(stmt) == VAR_DECL ||
                                  gimple_expr_code(stmt) == PARM_DECL ||
                                  gimple_expr_code(stmt) == POINTER_PLUS_EXPR ||
                                  gimple_expr_code(stmt) == ADDR_EXPR ||
                                  gimple_expr_code(stmt) == MEM_REF))
  {
    //fprintf (stderr, "recurse\n");
    return expr_from_arg_p (gimple_op(stmt, 1));
  }
  else if (gimple_code (stmt) == GIMPLE_PHI) {
    size_t nop = gimple_num_ops (stmt);
    for (size_t i = 0; i < nop; i++) {
      tree op = gimple_op (stmt, i);
      bool ret = expr_from_arg_p (op);
      if (ret) return ret;
    }
  }
  return false;
}


// Check to see if VAL derives from a static object.
tree expr_from_static_p (tree val)
{
  //debug_tree(val);
  if (static_p (val)) return val;
  
  if (TREE_CODE (val) == ADDR_EXPR)
    val = TREE_OPERAND (val, 0);
  if (TREE_CODE (val) == COMPONENT_REF)
    val = get_inner (val);
  if (TREE_CODE (val) == MEM_REF)
    val = TREE_OPERAND (val, 0);

  if (TREE_CODE (val) == VAR_DECL && static_p (val)) return val;
  if (TREE_CODE (val) != SSA_NAME) return NULL_TREE;

  gimplePtr stmt = SSA_NAME_DEF_STMT (val);
  if (!stmt) return NULL_TREE;
  //debug_gimple_stmt (stmt);
  //fprintf (stderr, "code %s\n", get_tree_code_name(gimple_expr_code(stmt)));

  if (is_gimple_assign (stmt) && (gimple_expr_code(stmt) == VAR_DECL ||
                                  gimple_expr_code(stmt) == PARM_DECL ||
                                  gimple_expr_code(stmt) == POINTER_PLUS_EXPR ||
                                  gimple_expr_code(stmt) == ADDR_EXPR ||
                                  gimple_expr_code(stmt) == MEM_REF))
  {
    //fprintf (stderr, "recurse\n");
    return expr_from_static_p (gimple_op(stmt, 1));
  }
  else if (gimple_code (stmt) == GIMPLE_PHI) {
    size_t nop = gimple_num_ops (stmt);
    for (size_t i = 0; i < nop; i++) {
      tree op = gimple_op (stmt, i);
      tree ret = expr_from_static_p (op);
      if (ret) return ret;
    }
  }
  return NULL_TREE;
}


// Test to see if STMT is a call to a virtual destructor.
// This is a bit annoying, since we can't directly get the function decl
// for a virtual call.  Instead, from the call we get the class type
// and the vtable index.  We need to search the list of methods for
// the class to find the one with the proper index.
bool is_virtual_dtor_call (gimplePtr stmt)
{
  if (gimple_call_fndecl (stmt)) return false;
  tree fn = gimple_call_fn (stmt);
  if (!fn || TREE_CODE (fn) != OBJ_TYPE_REF) return false;
  tree obj = OBJ_TYPE_REF_OBJECT (fn);
  tree ndx_tree = OBJ_TYPE_REF_TOKEN (fn);
  int ndx = TREE_INT_CST_LOW (ndx_tree);

  if (TREE_CODE (obj) != SSA_NAME) return false;
  tree typ = TREE_TYPE (obj);
  if (TREE_CODE (typ) == POINTER_TYPE)
    typ = TREE_TYPE (typ);
  if (TREE_CODE (typ) != RECORD_TYPE) return false;

#if GCC_VERSION >= 8000
  vec<tree, va_gc>* method_vec = CLASSTYPE_MEMBER_VEC (typ);
#else
  vec<tree, va_gc>* method_vec = CLASSTYPE_METHOD_VEC (typ);
#endif
  tree p;
  unsigned ix;
  FOR_EACH_VEC_SAFE_ELT (method_vec, ix, p) {
    if (p && TREE_CODE (p) == FUNCTION_DECL) {
      tree fn_ndx_tree = DECL_VINDEX (p);
      if (fn_ndx_tree && TREE_CODE (fn_ndx_tree) == INTEGER_CST) {
        int ndx_tree = TREE_INT_CST_LOW (fn_ndx_tree);
        if (ndx_tree == ndx) {
          return DECL_DESTRUCTOR_P (p);
        }
      }
    }
  }
  return false;
}


// Test to see if a value comes directly or indirectly from
// a structure.  Return the structure object if so, and set FIELD
// to the referenced member.
// If REQUIRE_POINTER is true, then require that the value be a pointer.
tree value_from_struct (tree val, bool require_pointer, tree& field)
{
  if (TREE_CODE (val) == COMPONENT_REF) {
    field = TREE_OPERAND (val, 1);
    return get_inner (val);
  }

  tree valtest = get_inner (val);
  tree valtype = TREE_TYPE (valtest);
  if (require_pointer && !POINTER_TYPE_P(valtype))
    return NULL_TREE;

  while (TREE_CODE (val) != SSA_NAME) {
    switch (TREE_CODE (val)) {
    case COMPONENT_REF:
      field = TREE_OPERAND (val, 1);
      return get_inner (val);
    case ADDR_EXPR:
    case ARRAY_REF:
    case MEM_REF:
      val = TREE_OPERAND (val, 0);
      break;
    case SSA_NAME:
      break;
    default:
      return NULL_TREE;
    }
  }

  gimplePtr stmt = SSA_NAME_DEF_STMT (val);
  if (!stmt) return NULL_TREE;
  //debug_gimple_stmt (stmt);
  //fprintf (stderr, "code %s\n", get_tree_code_name(gimple_expr_code(stmt)));
  
  if ((is_gimple_assign (stmt) || gimple_nop_p (stmt)) &&
      (gimple_expr_code(stmt) == VAR_DECL ||
       gimple_expr_code(stmt) == PARM_DECL ||
       gimple_expr_code(stmt) == POINTER_PLUS_EXPR ||
       gimple_expr_code(stmt) == ADDR_EXPR ||
       gimple_expr_code(stmt) == COMPONENT_REF))
  {
    //fprintf (stderr, "recurse\n");
    return value_from_struct (gimple_op(stmt, 1), require_pointer, field);
  }
  else if (gimple_code (stmt) == GIMPLE_PHI) {
    size_t nop = gimple_num_ops (stmt);
    for (size_t i = 0; i < nop; i++) {
      tree op = gimple_op (stmt, i);
      tree ret = value_from_struct (op, require_pointer, field);
      if (ret) return ret;
    }
  }
  return NULL_TREE;
}


// A function passing an argument on to an argument_not_const_thread_safe
// function must also be argument_not_const_thread_safe.
void check_argument_not_const_thread_safe_call (Attributes_t attribs,
                                                Attributes_t fnattribs,
                                                tree fndecl,
                                                gimplePtr stmt,
                                                function* fun)
{
  if (!has_attrib (fnattribs, ATTR_ARGUMENT_NOT_CONST_THREAD_SAFE)) return;
  if (has_attrib (attribs, ATTR_ARGUMENT_NOT_CONST_THREAD_SAFE)) return;

  // Check to see if any of the arguments being passed to fndecl
  // come from an argument.
  unsigned nargs = gimple_call_num_args (stmt);
  for (unsigned i=0; i < nargs; i++) {
    tree arg = gimple_call_arg (stmt, i);
    if (expr_from_arg_p (arg)) {
      warning_at (gimple_location (stmt), 0,
                  "Function %<%D%> passing argument %<%E%> of type %<%T%> to argument_not_const_thread_safe function %<%D%> must also be argument_not_const_thread_safe.",
                  fun->decl, arg, TREE_TYPE(arg), fndecl);
      CheckerGccPlugins::inform_url (gimple_location (stmt), url);
    }
  }

  if (!DECL_CONST_MEMFUNC_P (fun->decl)) return;

  // Check to see if any of the arguments being passed to fndecl
  // come from member data.
  for (unsigned i=0; i < nargs; i++) {
    tree arg = gimple_call_arg (stmt, i);
    tree field = NULL_TREE;
    tree s = value_from_struct (arg, false, field);
    
    if (s) {
      if (TREE_CODE (s) == MEM_REF)
        s = TREE_OPERAND (s, 0);
      if (TREE_CODE (s) != SSA_NAME) continue;
      s = SSA_NAME_VAR (s);
      if (TREE_CODE (s) != PARM_DECL) continue;
      if (DECL_NAME (s) != this_identifier) continue;
      //debug_tree (s);
      warning_at (gimple_location (stmt), 0,
                  "Const member function %<%D%> passing member data %<%E%> of type %<%T%> to argument_not_const_thread_safe function %<%D%> must also be argument_not_const_thread_safe.",
                  fun->decl, field, TREE_TYPE(field), fndecl);
      CheckerGccPlugins::inform_url (gimple_location (stmt), url);
    }
  }
}


void check_static_argument_not_const_thread_safe_call (Attributes_t attribs,
                                                       Attributes_t fnattribs,
                                                       tree fndecl,
                                                       gimplePtr stmt,
                                                       function* fun)
{
  if (!has_attrib (fnattribs, ATTR_ARGUMENT_NOT_CONST_THREAD_SAFE)) return;
  if (has_attrib (attribs, ATTR_NOT_REENTRANT)) return;

  // Check to see if any of the arguments being passed to fndecl
  // come from const static data.
  unsigned nargs = gimple_call_num_args (stmt);
  for (unsigned i=0; i < nargs; i++) {
    tree arg = gimple_call_arg (stmt, i);
    if (expr_from_static_p (arg)) {
      warning_at (gimple_location (stmt), 0,
                  "Function %<%D%> passing const static argument %<%E%> of type %<%T%> to argument_not_const_thread_safe function %<%D%> must be not_reentrant.",
                  fun->decl, arg, TREE_TYPE(arg), fndecl);
      CheckerGccPlugins::inform_url (gimple_location (stmt), url);
    }
  }
}


// Check passing an address of a static object to a called function
// by non-const pointer/ref.
//   ARG_TYPE: type of argument
//   ARG: argument to test
//   STMT: gimple statement being checked
//   FUN: function being checked
void check_pass_static_by_call (Attributes_t attribs,
                                tree fndecl,
                                tree arg_type,
                                tree arg,
                                gimplePtr stmt,
                                function* fun)
{
  if (!POINTER_TYPE_P (arg_type)) return;

  if (arg && TREE_CODE (arg) == ADDR_EXPR) {
    while (arg && TREE_CODE (arg) == ADDR_EXPR)
      arg = TREE_OPERAND (arg, 0);
  }
  tree argtest = get_inner (arg);

  if (static_p (argtest) && !has_attrib (attribs, ATTR_NOT_REENTRANT)) {
    if (!const_p (argtest)) {

      // Ok if it's an atomic value being passed to a member function.
      tree arg_test = arg_type;
      if (POINTER_TYPE_P (arg_test))
        arg_test = TREE_TYPE (arg_test);
      if (fndecl && is_atomic (arg_test) && is_atomic (DECL_CONTEXT (fndecl)))
        return;

      // Ok if it's a mutex.
      if (is_mutex (arg_test)) return;

      // FNDECL could be null in the case of a call through a function pointer.
      // Try to print a nice error in that case.
      tree fnerr = fndecl;
      if (!fnerr) {
        fnerr = error_mark_node;
        tree fn = gimple_call_fn (stmt);
        if (TREE_CODE (fn) == SSA_NAME &&
            FUNCTION_POINTER_TYPE_P (TREE_TYPE (fn)))
        {
          gimplePtr def_stmt = SSA_NAME_DEF_STMT (fn);
          if (is_gimple_assign (def_stmt)) {
            fnerr = gimple_op (def_stmt, 1);
          }
        }
      }

      warning_at (gimple_location (stmt), 0,
                  "Static expression %<%E%> of type %<%T%> passed to pointer or reference function argument of %<%E%> within function %<%D%>; may not be thread-safe.",
                  arg, TREE_TYPE(arg), fnerr, fun->decl);
      if (DECL_P (argtest)) {
        inform (DECL_SOURCE_LOCATION (argtest),
                "Declared here:");
      }
      CheckerGccPlugins::inform_url (gimple_location (stmt), url);
    }
  }
}


// Check for discarding const from a pointer/ref in a function call.
//   ARG_TYPE: type of argument
//   ARG: argument to test
//   STMT: gimple statement being checked
//   FUN: function being checked
void check_discarded_const_in_funcall (Attributes_t attribs,
                                       tree arg_type,
                                       tree arg,
                                       gimplePtr stmt,
                                       function* fun)
{
  bool lhs_const = TYPE_READONLY (TREE_TYPE (arg_type));

  // Non-const reference binding to something that's not const is ok.
  if (!lhs_const && TREE_CODE (arg_type) == REFERENCE_TYPE &&
      !TYPE_READONLY (TREE_TYPE (arg)))
  {
    return;
  }

  // We can get a false positive if a function return value is declared const;
  // for example:
  //    const std::string foo()
  //    {
  //      std::string tmp;
  //      tmp += 'a';
  //      return tmp;
  //    }
  //
  // In the call to operator+= above, the this argument will be the
  // SSA_NAME referencing the RESULT_DECL for the function return value,
  // which is marked READONLY.  Avoid this by skipping the test for
  // the outermost pointer types for a RESULT_DECL.
  bool is_result = (TREE_CODE (arg) == SSA_NAME &&
                    SSA_NAME_VAR (arg) &&
                    TREE_CODE (SSA_NAME_VAR (arg)) == RESULT_DECL);

  tree argtest = get_inner (arg);

  tree ctest = TREE_TYPE (argtest);
  while (POINTER_TYPE_P (ctest) && POINTER_TYPE_P (arg_type)) {
    if (!is_result &&
        TREE_READONLY (TREE_TYPE (ctest)) &&
        !TREE_READONLY (TREE_TYPE (arg_type)))
    {
      // We don't want to warn about calls to destructors.
      // We've already filtered out calls to non-virtual destructors,
      // but virtual destructors are more work to recognize.
      // So don't make that test until we're ready to emit a warning.
      if (!is_virtual_dtor_call (stmt))
      {
        warn_about_discarded_const (attribs, arg, stmt, fun);
      }
      return;
    }
    ctest = TREE_TYPE (ctest);
    arg_type = TREE_TYPE (arg_type);
    is_result = false;
  }
}


// Check for discarding const from the return value of a function.
//   STMT: gimple statement being checked
//   FUN: function being checked
void check_discarded_const_from_return (Attributes_t attribs,
                                        gimplePtr stmt,
                                        function* fun)
{
  tree lhs = gimple_call_lhs (stmt);
  if (!lhs) return;
  tree lhs_type = TREE_TYPE (lhs);
  if (!POINTER_TYPE_P (lhs_type)) return;
  bool lhs_const = TYPE_READONLY (TREE_TYPE (lhs_type));
  if (lhs_const) return;

  if (is_lhs_marked_thread_safe (lhs)) return;

  tree rhs_type = gimple_expr_type (stmt);
  if (POINTER_TYPE_P (rhs_type))
    rhs_type = TREE_TYPE (rhs_type);

  if (rhs_type && TYPE_READONLY (rhs_type))
  {
    warn_about_discarded_const (attribs, gimple_call_fn(stmt), stmt, fun);
  }
}


void check_calls (Attributes_t attribs, gimplePtr stmt, function* fun)
{
  if (gimple_code (stmt) != GIMPLE_CALL) return;

  // Can happen with ubsan.
  if (gimple_call_fn (stmt) == NULL_TREE) return;

  check_discarded_const_from_return (attribs, stmt, fun);

  bool ctor_dtor_p = false;

  tree fndecl = gimple_call_fndecl (stmt);
  if (fndecl) {
    // Skip calls to compiler-internal functions.
    if (DECL_ARTIFICIAL (fndecl)) return;
    tree name = DECL_NAME (fndecl);
    if (name) {
      const char* namestr = IDENTIFIER_POINTER (name);
      if (namestr && namestr[0] == '_' && namestr[1] == '_')
        return;
      if (strcmp (namestr, "operator delete") == 0) return;
      if (strcmp (namestr, "operator delete []") == 0) return;
    }
    
    Attributes_t fnattribs = get_attributes (fndecl);
    check_thread_safe_call (fnattribs, fndecl, stmt, fun);
    check_not_reentrant_call (attribs, fnattribs, fndecl, stmt, fun);
    check_not_const_thread_safe_call (attribs, fnattribs, fndecl, stmt, fun);
    check_argument_not_const_thread_safe_call (attribs, fnattribs, fndecl, stmt, fun);
    check_static_argument_not_const_thread_safe_call (attribs, fnattribs, fndecl, stmt, fun);

    if (DECL_CONSTRUCTOR_P (fndecl) || DECL_DESTRUCTOR_P (fndecl))
      ctor_dtor_p = true;
  }

  unsigned nargs = gimple_call_num_args (stmt);
  if (nargs < 1) return;
  tree fntype = TREE_TYPE (gimple_call_fn (stmt));
  if (TREE_CODE (fntype) == POINTER_TYPE)
    fntype = TREE_TYPE (fntype);
  tree arg_types = TYPE_ARG_TYPES (fntype);

  for (unsigned i=0; arg_types && i < nargs; i++, arg_types = TREE_CHAIN(arg_types))
  {
    if (i == 0 && ctor_dtor_p) continue;
    
    tree arg_type = TREE_VALUE (arg_types);
    tree arg = gimple_call_arg (stmt, i);

    if (!POINTER_TYPE_P (arg_type)) continue;

    check_discarded_const_in_funcall (attribs, arg_type, arg, stmt, fun);
    check_pass_static_by_call (attribs, fndecl, arg_type, arg, stmt, fun);
  }
}


void check_returns (gimplePtr stmt, function* fun)
{
  if (gimple_code (stmt) != GIMPLE_RETURN) return;
  //debug_gimple_stmt (stmt);
  tree retval = gimple_op (stmt, 0);
  tree field = NULL_TREE;
  tree s = value_from_struct (retval, true, field);
  if (s && TREE_READONLY (TREE_TYPE (s))) {
    warning_at (gimple_location (stmt), 0,
                "Returning non-const pointer or reference member %<%E%> of type %<%T%> from structure %<%D%> within const member function %<%D%>; may not be thread-safe.",
                field, TREE_TYPE(field), TREE_TYPE (s), fun->decl);
    inform (DECL_SOURCE_LOCATION (field), "Declared here:");
    CheckerGccPlugins::inform_url (gimple_location (stmt), url);
  }
}


unsigned int thread_pass::thread_execute (function* fun)
{
  // Return if we're not supposed to check this function.
  if (!check_thread_safety_p (fun->decl))
    return 0;

  // Also skip compiler-internal functions.
  if (fun->decl) {
    tree name = DECL_NAME (fun->decl);
    if (name) {
      const char* namestr = IDENTIFIER_POINTER (name);
      if (namestr && namestr[0] == '_' && namestr[1] == '_') {
        return 0;
      }

      // ROOT special case.
      // The ClassDef macro injects this into user classes; the inline
      // definition of this uses static data.
      if (strcmp (namestr, "CheckTObjectHashConsistency") == 0) {
        return 0;
      }
    }
  }


  Attributes_t attribs = get_attributes (fun->decl);
  check_attrib_consistency (attribs, fun->decl);

  const bool static_memfunc_p = DECL_CONST_MEMFUNC_P (fun->decl);
  tree rettype = TREE_TYPE (DECL_RESULT (fun->decl));
  const bool nonconst_pointer_return_p = POINTER_TYPE_P (rettype) && !TYPE_READONLY (TREE_TYPE (rettype));
  const bool not_const_thread_safe = has_attrib(attribs, ATTR_NOT_CONST_THREAD_SAFE);

  basic_block bb;
  FOR_EACH_BB_FN(bb, fun) {
    for (gimple_stmt_iterator si = gsi_start_bb (bb); 
         !gsi_end_p (si);
         gsi_next (&si))
    {
      gimplePtr stmt = gsi_stmt (si);
      //debug_gimple_stmt (stmt);

      check_direct_static_use (attribs, stmt, fun);
      check_assignments (attribs, stmt, fun);
      check_calls (attribs, stmt, fun);

      if (static_memfunc_p &&
          nonconst_pointer_return_p && !not_const_thread_safe)
        check_returns (stmt, fun);
    }
  }

  return 0;
}


// Warn about static/mutable fields in class type TYPE.
void check_mutable_static_fields (tree type)
{
  tree decl = TYPE_NAME (type);

  for (tree f = TYPE_FIELDS (type); f; f = DECL_CHAIN (f)) {
    if (TREE_CODE (f) == FIELD_DECL)
    {
      if (DECL_MUTABLE_P (f) &&
          !has_thread_safe_attrib (f) &&
          !is_mutex (TREE_TYPE (f)) &&
          !is_atomic (TREE_TYPE (f)) &&
          !is_thread_local (TREE_TYPE (f)))
      {
        warning_at (DECL_SOURCE_LOCATION(f), 0,
                    "Mutable member %<%D%> of type %<%T%> within thread-safe class %<%D%>; may not be thread-safe.",
                    f, TREE_TYPE(f), decl);
        CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION(f), url);
      }
    }
    else if (TREE_CODE (f) == VAR_DECL)
    {
      if (static_p (f) &&
          !const_p (f) &&
          !has_thread_safe_attrib (f) &&
          !is_atomic (TREE_TYPE (f)))
      {
        warning_at (DECL_SOURCE_LOCATION(f), 0,
                    "Static member %<%D%> of type %<%T%> within thread-safe class %<%D%>; may not be thread-safe.",
                    f, TREE_TYPE(f), decl);
        CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION(f), url);
      }
    }
  }
}


void find_overridden_functions_r (tree type,
                                  tree fndecl,
                                  std::vector<tree>& basedecls);

// FNDECL is a virtual function defined in TYPE.
// Fill BASEDECLS with base class functions overridden by FNDECL.
// Based on look_for_overrides from search.c.
void find_overridden_functions (tree type,
                                tree fndecl,
                                std::vector<tree>& basedecls)
{
  basedecls.clear();
  tree binfo = TYPE_BINFO (type);
  tree base_binfo;

  for (int ix = 0; BINFO_BASE_ITERATE (binfo, ix, base_binfo); ix++)
    {
      tree basetype = BINFO_TYPE (base_binfo);

      if (TYPE_POLYMORPHIC_P (basetype))
	find_overridden_functions_r (basetype, fndecl, basedecls);
    }
}


void find_overridden_functions_r (tree type,
                                  tree fndecl,
                                  std::vector<tree>& basedecls)
{
  tree fn = look_for_overrides_here (type, fndecl);
  if (fn) {
    basedecls.push_back (fn);
    return;
  }

  /* We failed to find one declared in this class. Look in its bases.  */
  find_overridden_functions (type, fndecl, basedecls);
}


void check_attrib_match (tree type, tree fndecl, tree basedecl, const char* attrname)
{
  bool b_has = lookup_attribute (attrname, DECL_ATTRIBUTES (basedecl));
  bool d_has = lookup_attribute (attrname, DECL_ATTRIBUTES (fndecl));

  if (b_has && !d_has) {
    warning_at (DECL_SOURCE_LOCATION(fndecl), 0,
                "Virtual function %<%D%> within class %<%D%> does not have attribute %<%s%>, but overrides %<%D%> which does.",
                fndecl, type, attrname, basedecl);
    inform (DECL_SOURCE_LOCATION (basedecl),
            "Overridden function declared here:");
    CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION(fndecl), url);
  }
  else if (d_has && !b_has) {
    warning_at (DECL_SOURCE_LOCATION(fndecl), 0,
                "Virtual function %<%D%> within class %<%D%> has attribute %<%s%>, but overrides %<%D%> which does not.",
                fndecl, type, attrname, basedecl);
    inform (DECL_SOURCE_LOCATION (basedecl),
            "Overridden function declared here:");
    CheckerGccPlugins::inform_url (DECL_SOURCE_LOCATION(fndecl), url);
  }
}


// Warn about overriding virtual functions with inconsistent threading attributes.
void check_virtual_overrides (tree type)
{
#if GCC_VERSION >= 8000
  for (tree meth = TYPE_FIELDS (type); meth; meth = TREE_CHAIN (meth)) {
    if (TREE_CODE (meth) != FUNCTION_DECL) continue;
#else
  for (tree meth = TYPE_METHODS (type); meth; meth = TREE_CHAIN (meth)) {
#endif
    if (DECL_ARTIFICIAL (meth)) continue;
    if (!DECL_VIRTUAL_P (meth)) continue;

    std::vector<tree> basedecls;
    find_overridden_functions (type, meth, basedecls);

    for (tree basedecl : basedecls) {
      check_attrib_match (type, meth, basedecl, "not_reentrant");
      check_attrib_match (type, meth, basedecl, "thread_safe");
      check_attrib_match (type, meth, basedecl, "not_thread_safe");
      check_attrib_match (type, meth, basedecl, "argument_not_const_thread_safe");
      check_attrib_match (type, meth, basedecl, "not_const_thread_safe");
    }
  }
}


// Called after a type declaration.
// Check class/struct members here.
void thread_finishtype_callback (void* gcc_data, void* /*user_data*/)
{
  tree type = (tree)gcc_data;
  if (TREE_CODE (type) != RECORD_TYPE) return;

  // Skip checking `aggregate' types --- essentially POD types.
  // However, this test is not reliable for template types --- we get called
  // with the template itself.  The template will not have had the
  // not_aggregate flag set, so CP_AGGREGATE_TYPE will always be true.
  if (CP_AGGREGATE_TYPE_P (type) && !CLASSTYPE_TEMPLATE_INFO (type)) return;

  tree decl = TYPE_NAME (type);
  if (!check_thread_safety_p (decl)) return;
  check_mutable_static_fields (type);
  check_virtual_overrides (type);
}


void thread_finishdecl_callback (void* gcc_data, void* /*user_data*/)
{
  tree decl = (tree)gcc_data;
  if (TREE_CODE(decl) == FUNCTION_DECL) {
    Attributes_t attribs = get_attributes (decl);
    check_attrib_consistency (attribs, decl);
  }
}


} // anonymous namespace


void init_thread_checker (plugin_name_args* plugin_info)
{
  struct register_pass_info pass_info = {
    new thread_pass(g),
    "ssa",
    0,
    PASS_POS_INSERT_AFTER
  };
  
  register_callback (plugin_info->base_name,
                     PLUGIN_PASS_MANAGER_SETUP,
                     NULL,
                     &pass_info);

  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_TYPE,
                     thread_finishtype_callback,
                     NULL);

  register_callback (plugin_info->base_name,
                     PLUGIN_FINISH_DECL,
                     thread_finishdecl_callback,
                     NULL);
}
