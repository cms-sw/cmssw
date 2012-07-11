
#include <string>

// is ok, because const-qualified
const static int g_staticConst = 23;

// results in a warning by GlobalStaticChecker
static int g_static;

int main()
{
    g_static = 23;

    return 0;
}
