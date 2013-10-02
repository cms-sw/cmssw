---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---

## Tutorial: proposing changes to CMSSW.

This tutorial wil guide you step to step from an empty directory to 
a new feature being proposed in the CMS Topic Collector. For more general
questions please have a look at the [FAQ](faq.html) pages.

### Before you start.

Please make sure you registered to GitHub and that you have provided them
a ssh public key to access your private repository. For more information see
the [FAQ](faq.html).

### Create a CMSSW area

Set up the work area just like you used to.

    > scram project CMSSW_7_0_0_pre0
    > cd CMSSW_7_0_0_pre0/src
    > cmsenv

### Checkout a few packages using git cms-addpkg

`git cms-addpkg` behaves like the old addpkg. 
The src directory is populated with just what we asked for (here
`Tutorial/Test`).

    > git cms-addpkg DataFormats/TestObjects
    > ls DataFormats/TestObjects
    BuildFile.xml

By default it would simply use the current release tag (e.g.
CMSSW_7_0_0_pre0). At any point you can list the available branches via `git
branch`.

    > git branch
    * from-CMSSW_7_0_0_pre0

the `*` next to the branch name will tell you which one is the current one.
The name of the branch is meant to show you the tag which was used as a base
for the branch.


### Create a new topic branch and move to it:

In order to simplify your workflow, the best thing to do is to branch
immediately and assign a mnemonic name to your newly created branch, so that
you can track what you were doing in an easier way.

    > git checkout -b my-new-feature
    Switched to a new branch 'my-new-feature'

For example you could call your branch "new-hlt-menu-XYZ",
"bugfix-for-clustering", etc. Compared to CVS, `git checkout <branch>` means 
*"Make `branch` the current `branch` in the local work area"*.
The `-b` option is needed to create the branch if it does not exists.


### Modify sources and commit.

    > cd DataFormats/TestObjects
    > nedit BuildFile.xml
    > git commit -m "Test feature" BuildFile.xml
    [my-new-feature 8135042] Test feature
     1 file changed, 1 insertion(+)

You have just committed to your **LOCAL** copy of the repository.

### Show remote repositories

Git supports multiple remote repository. In particular we will use:

- *my-cmssw*: your personal repository, used to propose changes
- *official-cmssw*: read only authoritative CMSSW sources 

To show what remotes are available you can use `git remote show`:

    > git remote show
    my-cmssw
    official-cmssw

The general workflow for development will be the following:
- Fetch official sources from `official-cmssw`.
- Create a feature branch and add your additions on top.
- Create a [Pull Request](https://help.github.com/articles/using-pull-requests).

 Each pull request you do to the [official CMSSW
 repository](https://github.com/cms-sw/cmssw) will also appear in the 
 [CMS Topic Collector][topic-collector], together with the approval information.

### Push all your work to your private github repository

Your private repository can be found at:

    https://github.com/<your-username>/cmssw

If you have not set it up yet, do so now by [clicking
here](https://github.com/cms-sw/cmssw/fork). You can publish branches to it
by doing:

    > git push my-cmssw <branch-name>

in this case:

    > git push my-cmssw my-new-feature

**You need to PUSH YOUR changes** to make them available to others in your
private repository. Notice you only need to specify the name of the branch once
then git will remember about it and you can push your new developments by
simply doing `git push my-cmssw`.


### Do a pull request

A **pull request** is a request to merge a given addition in your local repository
to a branch in the official CMSSW repository.  **Think of it as publishing a
tagset in Tag Collector.** To start one click on the "Pull request" button which
find on your personal repository.

![Pull request](images/pull-request.png)

### Select the topic branch which you want to propose and its target.

The GUI for doing "Pull requests" looks like below:

![Pull request](images/pull-request-gui.png)

If you click on the "Edit" button in the top right corner, you can select the
"head branch" i.e. your addition on to of the official CMSSW.

![Pull request](images/head-branch.png)

Or change the "base branch", i.e. the one to which you want to have your changes merged.

![Pull requests GUI](images/base-branch.png)

Remember to *write a sensible comment* just like you used to do for the
publishing of tagsets and click on Send Pull Request.

### Go to the Topic Collector and check approval status

Pull requests get automatically tracked in the [CMS Topic
Collector][topic-collector], just like tagsets were tracked in the CMS topic
collector.

![Topic Collector](images/topic-collector.png)

[topic-collector]: https://cern.ch/cmsgit/cmsgit
