
#include <string>

class Foo
{
public:
    void bar()
    {
        // will produce a warning by StaticLocalChecker
        static int evilStaticLocal = 0;
		static int & intRef = evilStaticLocal;
		static int * intPtr = & evilStaticLocal;

		// no warnings here
		static const int c_evilStaticLocal = 0;
		static int const& c_intRef = evilStaticLocal;
		static int const* c_intPtr = &evilStaticLocal;
		static const int * c_intPtr_equivalent = &evilStaticLocal;

		static int const* const* c_intPtrPtr = &( c_intPtr);
    }
};

int main()
{
    return 0;
}

