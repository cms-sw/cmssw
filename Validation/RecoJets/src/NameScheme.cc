#include "Validation/RecoJets/interface/NameScheme.h"

NameScheme::NameScheme():
  name_("Default"),
  link_("_")
{
}

NameScheme::NameScheme(const char* name):
  name_(name),
  link_("_")
{
}

NameScheme::NameScheme(const char* name, const char* link):
  name_(name),
  link_(link)
{
}

NameScheme::~NameScheme()
{

}

TString
NameScheme::name(const int i)
{
  TString namestr( name_ );
  namestr += link_;
  namestr += i;
  return namestr;
}

TString
NameScheme::name(const char* name)
{
  TString namestr( name_ );
  namestr += link_;
  namestr += name;
  return namestr;
}

TString
NameScheme::name(const char* name, const int i)
{
  TString namestr( name_ );
  namestr += link_;
  namestr += name;
  namestr += "_";
  namestr += i;
  return namestr;
}

TString
NameScheme::name(const char* name, const int i, const int j)
{
  TString namestr( name_ );
  namestr += link_;
  namestr += name;
  namestr += "_";
  namestr += i;
  namestr += "_";
  namestr += j;
  return namestr;
}

TString
NameScheme::name(ofstream& file, const char* name)
{
  TString namestr( name_ );
  namestr += link_;
  namestr += name;

  file << namestr << "\n";
  return namestr;
}

TString
NameScheme::name(ofstream& file, const char* name, const int i)
{
  TString namestr( name_ );
  namestr += link_;
  namestr += name;
  namestr += "_";
  namestr += i;

  file << namestr << "\n";
  return namestr;
}

TString
NameScheme::name(ofstream& file, const char* name, const int i, const int j)
{
  TString namestr( name_ );
  namestr += link_;
  namestr += name;
  namestr += "_";
  namestr += i;
  namestr += "_";
  namestr += j;

  file << namestr << "\n";
  return namestr;
}
