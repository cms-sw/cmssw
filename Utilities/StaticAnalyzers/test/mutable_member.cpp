
#include <string>

class Foo
{
public:
    // will produce a warning by MutableMemberChecker
    mutable int evilMutableMember;
    int goodMember;
};

int main()
{
    return 0;
}
