## @package RSSParser
# \brief Lighweight rss parser for ValidationTools
#
# Developers:
#   Victor E. Bazterra
#   Kenneth James Smith


import urllib
import time
from sgmllib import SGMLParser

import Configuration
import ErrorManager
import FeedParser

CERN = Configuration.variables['CERNSITE']
FNAL = Configuration.variables['FNALSITE']

## Auxiliary for html parser
class PREParser(SGMLParser):
 
  def reset(self):
    SGMLParser.reset(self)
    self.get_data = False
    self.text = ''

  def start_pre(self, tags):
    self.get_data = True
    text = self.get_starttag_text()

  def handle_data(self, data):
    if self.get_data == True:
      self.text = data

  def end_pre(self):
    self.get_data = False


## Main parser for rss information from DBS
class Parser:

  ## Defaul constructor
  def __init__(self):
    self.rss = None
    self.__rssgenerator = Configuration.variables['RssGenerator']
    self.__lfnforsite   = Configuration.variables['LFNForSite']     

  ## Set the request string
  def rssGenerator(self, string):
    self.__rssgenerator = string
    
  def getLFNsForSite(self, string): 
    self.__lfnforsite = string

  ## Request to the rss generator
  def request(self,  usermode='user', tier='any', group='any', dbsinst='cms_dbs_prod_global', release='any', datatype='any', dataset='any'):
    string = self.__rssgenerator + 'userMode=' + usermode + '&' 
    string = string + 'kw_tier=' + tier + '&'
    string = string + 'dbsInst=' + dbsinst + '&'
    string = string + 'app=' + release + '&'
    string = string + 'primType=' + datatype + '&'
    string = string + 'primD=' + dataset
    self.rss = FeedParser.parse(string) 

  ## Get the LFN for the last update in feed.
  def trigger(self, usermode='user', dbinst='cms_dbs_prod_global', site=FNAL, format='cff', keyword=''):
    if len(self.rss.entries) == 0 :
      raise ErrorManager.RSSParserError, 'There is not information in the channel.'
    index = -1
    for i in range(len(self.rss.entries)):
      if i == 0:
        lastUp = self.rss.entries[i].updated_parsed
        if keyword == '' or keyword.startswith('+') and str(self.rss.entries[i]).find(keyword[1:]) != -1 or keyword.startswith('-') and str(self.rss.entries[i]).find(keyword[1:]) == -1:
          index = i
      else:
        update = self.rss.entries[i].updated_parsed
        if update > lastUp:
          lastUp = update
          if keyword == '' or keyword.startswith('+') and str(self.rss.entries[i]).find(keyword[1:]) != -1 or keyword.startswith('-') and str(self.rss.entries[i]).find(keyword[1:]) == -1:
            index = i
    if index < 0:
      raise ErrorManager.RSSParserError, 'There is not a file block associated to this trigger.'
    string = self.__lfnforsite + 'dbsInst=' + dbinst + '&'
    string = string + 'site=' + site + '&'
    string = string + 'datasetPath=' + str(self.rss.entries[index].title) +'&'
    string = string + 'what=' + format  + '&'
    string = string + 'userMode=' + usermode  
    preparser = PREParser()
    sock = urllib.urlopen(string)
    preparser.feed(sock.read())
    sock.close()
    if preparser.text != '':
      return preparser.text
    raise ErrorManager.RSSParserError, 'There is not LFNs associated to the block name %s.' % self.rss.entries[index].title
