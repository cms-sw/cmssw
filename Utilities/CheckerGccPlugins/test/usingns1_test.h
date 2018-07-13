// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

namespace std {}
using namespace std;
using std::qq;
int xx;

namespace foo {}
namespace boost { namespace aux { using namespace foo; } }

void baz()
{
  using namespace std;
}

namespace {}
