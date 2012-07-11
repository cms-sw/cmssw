
#include <string>

class Foo
{
public:
    void bar()
    {
        // will produce a warning by StaticLocalChecker
        static int evilStaticLocal;
    }
};

int main()
{
    return 0;
}

