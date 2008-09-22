from xml.dom import minidom

# iterator object to iterate over all the elements of a (dom.minidom) DOM tree, starting from a given element
class DOMIterator:
  element = None

  # can be initialized with a DOM node or a DOM document (in the latter case the main element is used)
  def __init__(self, element = None):
    if isinstance(element, minidom.Document):
      self.element = element.documentElement
    elif isinstance(element, minidom.Node):
      self.element = element
    else:
      raise TypeError, "type not supported by DOMIterator"

  def __iter__(self):
    return self

  def next(self):
    if self.element is None:
      raise StopIteration

    value = self.element

    if self.element.firstChild:
      self.element = self.element.firstChild
    elif self.element.nextSibling:
      self.element = self.element.nextSibling
    else:
      self.element = self.element.parentNode.nextSibling

    return value


# remove text from DOM nodes, making it suitable for document.toprettyxml() 
def dom_strip(document):
  for element in DOMIterator(document):
    if element.nodeType == 3:
      if element.nodeValue.count("\n") < 2:
        element.parentNode.removeChild(element)
        element.unlink()
      else:
        element.nodeValue = ""

