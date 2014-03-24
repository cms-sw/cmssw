---
title: CMS Offline Software
layout: default
related:
 - { name: "Project page", link: "https://github.com/cms-sw/cmssw" }
 - { name: "Feedback", link: "https://github.com/cms-sw/cmssw/issues/new" }
---
# FAQs

* auto-gen TOC:
{:toc}

# General questions

### Where can I learn about git, github in general?

- Generic git help can be found at <http://git-scm.com/book/>.
- github.com specific help can be found at <https://help.github.com>.
- A 15 minutes, interactive, git tutorial <http://try.github.com/levels/1/challenges/1>
- A nice video tutorial about git <http://www.youtube.com/watch?v=ZDR433b0HJY>
- A nice interactive tutorial about branches in git <http://pcottle.github.io/learnGitBranching/>
- Even more FAQs <https://git.wiki.kernel.org/index.php/Git_FAQ>

### How do I subscribe to github?

In order to develop CMSSW you will need a github account.

* In case you don't have one already, simply go to:

  [https://github.com/join]()

  and follow the instructions to create a new account. Make sure you use a
  username people can recognize you easily or to specify your real name.

* In case you already have an account you can simply use "the Sign in" dialog and
put your username and password.

  [https://github.com/login]()

Once you are done you should also setup your personal information:

    git config --global user.name <First Name> <Last Name>
    git config --global user.email <Your-Email-Address>
    git config --global user.github <Your-Just-Created-GitHub-Account>

Keep also in mind that git uses `$VISUAL` not `$CVS_EDITOR` for edit commit
messages, so you might want to adapt your shall profile as well.

Finally, make sure you [register in github your ssh
key](https://help.github.com/articles/generating-ssh-keys).

# Working with CMSSW on github

### How do I checkout one or more packages?

If you are in a CMSSW area (remember to do `cmsenv`) you can simply use:
  
    git cms-addpkg <package-name>

once you have developments you can checkout dependent packages by doing.

    git cms-checkdeps

[To learn more about git cms-addpkg click here](git-cms-addpkg.html).

You can also find a complete tutorial [here](tutorial.html).


### How do I develop a new feature using git?

Please have a look at the [full blown tutorial about proposing new
changes in CMSSW](tutorial.html).

### How do I check the status of my pull request(s).

Go to the [CMS Topic Collector][cms-topic-collector]. There you'll find all the
open requests and their approval status.

### How do I make sure my topic branch is updated with the latest developments?

Simply merge the release branch into your topic branch:

    git checkout new-feature
    git fetch official-cmssw
    git merge official-cmssw/CMSSW_6_2_X

or in one command:

    git pull official-cmssw CMSSW_6_2_X

assuming you are on the `new-feature` branch already. 

For more information about merging branching read
[here](http://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging).

You can also have a look at the CMS git tutorial found [here](tutorial).

### What about UserCode?

Please have a look at the [UserCode FAQ](usercode-faq).

### I used to do X in CVS how do I do the same in git?

Please have a look at the [Rosetta Stone](rosetta.html) page which has a few
conversions from CVS-speak to git-speak. Keep in mind that due to different
designs not all the things which you do in one are possible in the other.

### How do I tag a single package?

Tagging a single package is not possible with git, when you tag something
you'll always tag the full repository, tags are only aliases to commits.
However tags are cheap, so we can afford to tag a single integration build.

### How can I limit the diff between two tags to one single package?

You can do it by specifying a path at the end of the git command:

    git diff TAG1..TAG2 <some-path>

### How do you delete a branch in git?

In order to delete a branch from your repository you can do:

    git push my-cmssw --delete <branch-name>

or you can use the [web based github interface described
here](https://github.com/blog/1377-create-and-delete-branches).

### Will you migrate all the release tags we used to have for `CMSSW`?

Yes, all release (`CMSSW_X_Y_Z`) tags currently in CVS will be available in
git.

### Will you migrate all the per package tags?

No. Per package tags will not be migrated. You can however have a look at the
[Dealing with CVS History page](cvs-interaction.html) to see how you can get
old tags which were not in any release. This will also be useful to import
packages which did not end up in any release.

### What is the policy for tagging?

No tags other than "release tags" will be allowed inside the _official-cmssw_
repository, so there is not a particular need for a convention for tags.

In the git model, changes are proposed via private branches 
which are made into Pull Requests. Given a Pull Request gets automatically
assigned a unique ID (like for tagsets), we will not have a particular
convention, treating them as private tags in the CVS model.

The only recommendation so far has been use "use a somewhat descriptive names of the actual content".

### Is it possible to display a graphical view of my branches?

For a text based output, you can use either:

    git show-branch <branch-1> <branch-2> ... <branch-n>

or

    git log --graph --abbrev-commit <branch-1> <branch-2> ... <branch-n>

Moreover there are a number of graphical GUIs including
gitk (Linux, Mac, Windows, included in git) or [SourceTree](http://www.sourcetreeapp.com)
(Mac, Windows).

### How can I do showtags?

`showtags` is _CVS_ centric in the sense in git we have no per package tags
anymore. 

To replace it, you can get the modified files with raw git commands by doing:

     git diff --name-only $CMSSW_VERSION

(of course, drop --name-only if you want the full diff).

A slightly more elaborate way of getting the modified packages is:

     git diff --name-only $CMSSW_VERSION | cut -f1,2 -d/ | sort -u

However in git what makes more sense is to find out the topic branches which 
are on top of some base release. You can get these of with:

     git log --graph --merges --oneline $CMSSW_VERSION..

For example:

     git log --graph --oneline --merges CMSSW_7_0_X_2013-07-17-0200..

gives:

     * ef036dd Merge pull request #119 from ktf/remove-broken-file
     * ec831ca Merge pull request #103 from xiezhen/CMSSW_7_0_X
     * d057e80 Merge pull request #97 from inugent/ValidationforMcM53X
     * 085470d Merge pull request #89 from ianna/mf-geometry-payload
     * 1b87cbc Merge pull request #94 from inugent/TauolappandTauSpinner
     * 4ecd70d Merge pull request #124 from gartung/fix-for-llvm-3.3

which shows the importance of good naming for the branches.

If you drop `--merges` you can also get the single commits that people had in
one branch:

     git log --graph --oneline CMSSW_7_0_X_2013-07-17-0200..

will give you:

     *   ef036dd Merge pull request #119 from ktf/remove-broken-file
     |\
     | * 1219f84 Remove completely broken file.
     *   ec831ca Merge pull request #103 from xiezhen/CMSSW_7_0_X
     |\
     | * 0696b0b add testline
     | * 3ae6ab3 added
     | * c2fe08f align with V04-02-08 and fix head report
     | * de3794e align with V04-02-08 and fix head report
     | * e1f8dc2 align with V04-02-08 and fix head report
     | * bac83ea align with V04-02-08
     | * 8cfd791 align with V04-02-08
     | * fd3c705 align with V04-02-08
     | * d7c1596 adopt to schema change
     *   d057e80 Merge pull request #97 from inugent/ValidationforMcM53X
     |\
     | * f6bd948 Updating to cvs head-V00-02-30
     *   085470d Merge pull request #89 from ianna/mf-geometry-payload
     |\
     | * 16f2ebf Use standard geometry file record for magnetic field payload.
     | * 62db3c5 Add Magnetic Field geometry readers from DB payload.
     | * f1fd7e6 Magnetic field payload producers including job configurations.
     | * 0f4acaa Magnetic field payload producers including job configurations.
     | * 526b4f1 Remove GEM from Extended scenario.
     | * bea321c Scripts, configurations and metadata to produce Extended 2019 scenario payloads including GEM.
     | * 87e94e5 Add GEM reco geometry record.
     *   1b87cbc Merge pull request #94 from inugent/TauolappandTauSpinner
     |\
     | * b8cc783 Delete TauSpinnerCMS.cc.~1.2.~
     | * ce66597 adding Tauola 1.1.3 with TauSpinner interface
     | * ec56829 adding Tauola 1.1.3 with TauSpinner interface
     | * 865866a adding Tauola 1.1.3 with TauSpinner interface
     * 4ecd70d Merge pull request #124 from gartung/fix-for-llvm-3.3
     * 2b11a9d fix for api changes in llvm 3.3

which once again shows the importance of good comments.

### How to I retract my own pull request?

Simply close it using the standard GitHub GUI when looking at it.

* Go to the Pull Request page, either by clinking on the list on GitHub, or by
  clicking on it on the Topic Collector.
* Scroll down to the bottom of the discussion related to your pull request.
* Click on "Close"

The pull request will disappear from the list of open pull requests in both
GitHub and the Topic Collector.

### Do you have any more in-depth FAQs?

Yes, please look at the [Advanced Usage](advanced-usage) section.

### How do I receive notifications about pull requests for a given package / subsystem?

Please open an issue at

https://github.com/cms-sw/cmssw/issues/new

specifying which package / subsytem whose changes you would like to be notified about.

### Do you have a nice tutorial on how to develop CMSSW on git?

Yes, please have a look at the [CMSSW git tutorial pages](tutorial.html).

### How do I access the old CVS repository to check what was really there?

The old CVS repository is available *READ-ONLY* by setting:

    export CVSROOT=":ext:<cern-user-account>@lxplus5.cern.ch:/afs/cern.ch/user/c/cvscmssw/public/CMSSW"
    export CVS_RSH=ssh
    # setenv CVSROOT ":ext:<cern-user-account>@lxplus5.cern.ch:/afs/cern.ch/user/c/cvscmssw/public/CMSSW"
    # setenv CVS_RSH ssh

where of course `<cern-user-account>` needs to be substituted with your CERN
account login. Notice that starting on the 15th of October this will be the
only way to access it.

Moreover, if you want to simply browse the old repository via web, you can point your browser to:

<http://cvs.web.cern.ch/cvs/cgi-bin/viewcvs.cgi>

### How do I ask a question?

If you have more questions about git and CMSSW on git, please use [this
form][new-faq-form].

### How do I contribute to these pages?

The documentation you are reading uses [GitHub Pages](http://pages.github.com)
to publish web pages. To contribute to it you need to:

- Register to github.
- Fork the cmssw project under your account ([click here to do
  it](https://github.com/cms-sw/cmssw/fork)).
- Checkout the `gh-pages` branch:

      git clone -b gh-pages git@github.com:cms-sw/cmssw.git cmssw-pages

- Edit the documentation and push it to your branch:

      <edit-some-documentation>
      git commit <my-changed-files>
      git push git@github.com:<your-github-username>/cmssw.git gh-pages

- Create a "pull request" for you changes by going 
  [here](https://github.com/cms-sw/cmssw/pull/new/gh-pages).

This will trigger a discussion (and most likely immediate approval) of your
documentation changes.

[new-faq-form]: https://github.com/cms-sw/cmssw/issues/new
