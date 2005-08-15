#ifndef UTILITIES_GENERAL_MTMAP_H
#define UTILITIES_GENERAL_MTMAP_H
//
//   V 0.0 
//
#include "Utilities/Threads/interface/ThreadUtils.h"
#include <map>

/**  a thread-safe "map"
 */
template<class K, class T>
class MTmap {
public:

  typedef std::map< K, T, std::less<K> > SMAP;
  typedef typename SMAP::const_iterator const_iterator;
  typedef typename SMAP::iterator iterator;
  typedef typename SMAP::value_type value_type;
  typedef typename SMAP::key_type key_type;
  typedef T data_type;
    
  /// constructor
  MTmap(){}


  /// destructor
  ~MTmap(){}

  iterator find(const K& k) {
    LockMutex a(mutex_);
    return me.find(k);
  }

  const_iterator find(const K& k) const {
    LockMutex a(mutex_);
    return me.find(k);
  }

  T & operator[](const K& k) {
    LockMutex a(mutex_);
    return me[k];
  }
 
  std::pair<iterator,bool> insert(const value_type & elem) {
    LockMutex a(mutex_);
    return me.insert(elem);
  }
  
  iterator begin() {
    return me.begin();
  }

  const_iterator begin() const {
    return me.begin();
  }

  iterator end() {
    return me.end();
  }
 
  const_iterator end() const {
    return me.end();
  }

  size_t size() const {
    return me.size();
  }

  bool empty() const {
    return me.empty();
  }

  SMAP & backdoor() { return me;}
  const SMAP & backdoor() const { return me;}

  LockMutex::Mutex & mutex() const { return mutex_;}

  void erase(iterator pos) {
    LockMutex a(mutex_);
    me.erase(pos);
  }
 
  void clear() {
    LockMutex a(mutex_);
    me.clear();
  }

protected:

  SMAP me;

  mutable LockMutex::Mutex mutex_;
};


/** return object T for this thread...
 */
template<class T>
class ThreadObjects {
public:
  typedef MTmap<int, T> SMAP;

  typedef typename SMAP::const_iterator const_iterator;
  typedef typename SMAP::iterator iterator;
  typedef std::pair<const int , T> value_type;

  iterator find() { return them.find(here());}

  const_iterator find() const { return them.find(here());}

  T & operator()() { return them[here()];}

  T & get() { return them[here()];}

  std::pair<iterator,bool> insert(const T & t) {
    return them.insert(value_type(here(),t));
  }
  void erase() {
    them.erase(find());
  }

  iterator begin() {
    return them.begin();
  }

  const_iterator begin() const {
    return them.begin();
  }

  iterator end() {
    return them.end();
  }
 
  const_iterator end() const {
    return them.end();
  }

  static int here() { return thread_self_tid();}	   
private:

  SMAP them;

};

#endif // UTILITIES_GENERAL_MTMAP_H
