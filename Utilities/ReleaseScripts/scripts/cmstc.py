"""CMS TagCollector Python API
"""

__author__ = "Miguel Ojeda"
__copyright__ = "Copyright 2010-2011, CERN CMS"
__credits__ = ["Miguel Ojeda"]
__license__ = "Unknown"
__maintainer__ = "Miguel Ojeda"
__email__ = "mojedasa@cern.ch"
__status__ = "Staging"

_tagcollector_url = 'https://cmstags.cern.ch/tc/'

import urllib
import urllib2
import cookielib
import json
import getpass

class TagCollector(object):
	"""CMS TagCollector Python API"""

	def __init__(self):
		self._url = _tagcollector_url
		self._cj = cookielib.CookieJar()
		self._opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(self._cj))

	def _open(self, page, params = None, data = None):
		url = self._url + page + '?'
		if params:
			url += urllib.urlencode(params)
		if data:
			data = urllib.urlencode(data)
		try:
			return self._opener.open(url, data).read()
		except urllib2.HTTPError as e:
			raise Exception(e.read().strip())

	def _openjson(self, page, params = None, data = None):
		return json.loads(self._open(page, params, data))

	def signIn(self, username, password):
		"""Sign in to TagCollector."""
		self._open('CmsTCLogin', data = {'username': username, 'password': password})

	def signInInteractive(self):
		"""Sign in to TagCollector, asking for the username and password."""
		username = raw_input('Username: ')
		password = getpass.getpass()
		self.signIn(username, password)

	def signOut(self):
		"""Sign out of TagCollector."""
		self._open('signOut')

	def getPackageTags(self, package):
		"""Get the tags published in TagCollector for a package.
		Note: TagCollector's published tags are a subset of CVS' tags."""
		return self._openjson('py_getPackageTags', {'package': package})

	def getPackageTagDescriptionFirstLine(self, package, tag):
		"""Get the first line of the descriptions of a tag."""
		return self._openjson('py_getPackageTagDescriptionFirstLine', {'package': package, 'tag': tag})

	def getPackageTagReleases(self, package, tag):
		"""Get the releases where a tag is."""
		return self._openjson('py_getPackageTagReleases', {'package': package, 'tag': tag})

	def getReleasesTags(self, releases, diff = False):
		"""Get the tags of one or more release.
		Optionally, return only the tags that differ between releases."""
		releases = json.dumps(releases)
		diff = json.dumps(diff)
		return self._openjson('py_getReleasesTags', {'releases': releases, 'diff': diff})

	def getReleaseTags(self, release):
		"""Get the tags of one release."""
		return self.getReleasesTags((release, ))

	def approveTagsets(self, tagset_ids, comment = ''):
		"""Approve one or more tagsets.
		Requirement: Signed in as a Release Manager for each tagset's release."""
		tagset_ids = json.dumps(tagset_ids)
		self._open('approveTagsets', {'tagset_ids': tagset_ids, 'comment': comment})

	def getPackagesPendingApproval(self):
		"""Get New Package Requests which are Pending Approval."""
		return self._openjson('py_getPackagesPendingApproval')

	def getPackageManagersRequested(self, package):
		"""Get the Package Managers (administrators and developers) requested in a New Package Request."""
		return self._openjson('py_getPackageManagersRequested', {'package': package})

	def approvePackage(self, package):
		"""Approve a New Package Request.
		Requirement: Signed in as a Creator (i.e. people in the top-level .admin/developers file).
		Warning: This does *not* create the package in CVS."""
		self._open('approveNewPackageRequest', {'package_name': package})

	def getIBs(self, filt = '', limit = 10):
		"""Get the name and creation date of Integration Builds.
		By default, it only returns the latest 10 IBs.
		Optionally, filter by name."""
		return self._openjson('py_getIBs', {'filt': filt, 'limit': limit})

