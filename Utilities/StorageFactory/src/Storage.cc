#include "Utilities/StorageFactory/interface/Storage.h"
#include <cassert>

Storage::Storage (void)
{}

Storage::~Storage (void)
{}

//////////////////////////////////////////////////////////////////////
IOSize
Storage::read (IOBuffer into, IOOffset pos)
{ return read (into.data (), into.size (), pos); }

IOSize
Storage::read (void *into, IOSize n, IOOffset pos)
{
  // FIXME: this is not thread safe!  split into separate interface
  // that a particular storage can choose to support or not?  make
  // sure that throw semantics are correct here!
  // FIXME: use saveposition object in case exceptions are thrown?
  IOOffset here = position ();
  position (pos);
  n = read (into, n);
  position (here);
  return n;
}

//////////////////////////////////////////////////////////////////////
IOSize
Storage::write (IOBuffer from, IOOffset pos)
{ return write (from.data (), from.size (), pos); }

IOSize
Storage::write (const void *from, IOSize n, IOOffset pos)
{
  // FIXME: this is not thread safe!  split into separate interface
  // that a particular storage can choose to support or not?  make
  // sure that throw semantics are correct here!

  // FIXME: use saveposition object in case exceptions are thrown?
  IOOffset here = position ();
  position (pos);
  n = write (from, n);
  position (here);
  return n;
}

//////////////////////////////////////////////////////////////////////
IOOffset
Storage::position (void) const
{
  Storage *self = const_cast<Storage *> (this);
  return self->position (0, CURRENT);
}

IOOffset
Storage::size (void) const
{
  // FIXME: use saveposition object in case exceptions are thrown?
  Storage *self = const_cast<Storage *> (this);
  IOOffset here = position ();
  self->position (0, END);
  IOOffset size = position ();
  self->position (here); // FIXME: VERIFY()?
  return size;
}

void
Storage::rewind (void)
{ position (0); }

//////////////////////////////////////////////////////////////////////
void
Storage::preseek (const IOBuffer * /* offsets */, IOSize /* buffers */)
{}

//////////////////////////////////////////////////////////////////////
void
Storage::flush (void)
{}

void
Storage::close (void)
{}

//////////////////////////////////////////////////////////////////////
bool
Storage::eof (void) const
{ return position () == size (); }
