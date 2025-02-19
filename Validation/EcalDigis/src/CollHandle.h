/*
 * $Id: CollHandle.h,v 1.6 2010/02/11 00:15:21 wmtan Exp $
 */

#ifndef EcalDigis_CollHandle_h
#define EcalDigis_CollHandle_h

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/** Utilitity class for handling an EDM data collection. This class
 * acts as a wrapper of the EDM collection.
 * <P>An InputTag indentifying the collection is passed to the constructor.
 * The collection is retrieved from the event by a call to the read() method.
 * The CollHandle class instance can then be used as a pointer to the retrieved
 * collection.
 * <P>Absence of the collection from the event can be optionnaly tolerated: see
 * failIfNotFound parameter of the CollHandle() constructor.
 * <P>In case the collection was not (yet) retrieved from the collection, it
 * acts as a pointers to an empty collection.
 * <P>The templace parameter T specifies the type of the data collection.
 */
template<class T>
class CollHandle {
  //constructor(s) and destructor(s)
public:
  /** Constructs a CollHandle.
   * @param tag InputTag identifying the data collection
   * @param failIfNotFound pass true if the absence of the collection
   * in the event must be considered as an error. See read() method.
   */
  CollHandle(const edm::InputTag& tag,
	     bool failIfNotFound = true,
	     bool notFoundWarn = true): tag_(tag),
					currentColl_(&emptyColl_),
					notFoundAlreadyWarned_(false),
					failIfNotFound_(failIfNotFound),
					notFoundWarn_(notFoundWarn){}
  
  //method(s)
public:
  /** Retrieves the collection from the event. If failIfNotFound is true and
   * the collection is not found, then an edm::Exception is thrown. For other
   * case of exception throw see edm::Event::getByLabel() method documentation.
   * If failIfNotFound is false and the collection is not found, an empty
   * collection is used; a warn message will be logged the first time
   * the collection is not found.
   * @param event the EDM event the collection must be retrieved from.
   */
  void read(const edm::Event& event){
    //    try{
    edm::Handle<T> hColl;
    event.getByLabel(tag_, hColl);
  
    //If we must be tolerant to product absence, then
    //we must check validaty before calling Handle::operator*
    //to prevent exception throw:
    if(!failIfNotFound_     // product-not-found tolerant mode
       && !hColl.isValid()){// and the product was not found
      if(notFoundWarn_
	 && !notFoundAlreadyWarned_){//warning logged only once
	edm::LogWarning("ProductNotFound") << tag_
					   << " product "
          "of type '" << typeid(T).name() << "' was not found!";
	notFoundAlreadyWarned_ = true;
      }
      currentColl_ = &emptyColl_;
    } else {
      currentColl_ = &(*hColl);
    }
  }

  
  /** Accessor to a member of the collection retrieved by read method().
   * Considering h a CollHandle instance, h->f() is equivalent to (*h).f().
   */
  const T* operator->() const{ return currentColl_;}

  /** Gets the collection retrieved by read() method. If the collection was
   * absent from the event an empty collection is returned.
   */
  const T& operator*() const { return *currentColl_;}

  edm::InputTag tag() const { return tag_; }
  
private:

  //attribute(s)
protected:
private:
  /** tag identifying the data collecion
   */
  const edm::InputTag tag_;

  /** Pointer to the last read collection, points to emptColl be default
   */
  const T* currentColl_;

  /** An empty collection used as default when the collection was not retrieved
   * from the event.
   */
  const T emptyColl_;

  /** Used to emit warn message in case of collection absence only once.
   */
  bool notFoundAlreadyWarned_;

  /** Switch for collection absence toleration.
   */
  bool failIfNotFound_;

  /** Switch for the warning in case of not found collection
   */
  bool notFoundWarn_;
};

#endif
