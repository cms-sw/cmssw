
#include <string>

// is ok, because const-qualified
const static int g_staticConst = 23;
static int const& g_ref_staticConst = g_staticConst;
static int const* g_ptr_staticConst = &g_staticConst;


// results in a warning by GlobalStaticChecker
static int g_static;
static int * g_ptr_static = &g_static;


int main()
{
    g_static = 23;

    return 0;
}
