# git hub repo writer
# repo at https://github.com/SVVSDICAI/FishNetStreamCapture
import os
from git import Repo
import sys

PATH_OF_GIT_REPO = r'./FishLadderStreamCapture'
COMMIT_MESSAGE = "automated update"

GITHUB_KEY = os.getenv('GITHUB_KEY')

if len(sys.argv) > 1 and sys.argv[1] == "init": # if this file is run from the terminal (python RepoUpdate.py init) to initialize the github login info and clone the repo to the current directory
    # if the user provides a repo link, use that instead of the default PATH_OF_GIT_REPO
    if len(sys.argv) == 3:
        PATH_OF_GIT_REPO = sys.argv[2]
        
    print("cloning repo")
    # github login info
    username = "automated"
    remote = f"https://{username}:{GITHUB_KEY}@github.com/SVVSDICAI/FishNetStreamCapture.git"
    remote = remote.replace("\n", "")
    print(remote)
    Repo.clone_from(remote, PATH_OF_GIT_REPO) # This must be run initially to ensure that the github login info is set

def git_push(PATH_OF_GIT_REPO=PATH_OF_GIT_REPO): # function to push the updates to github
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add(update=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name="origin")
        print("pushing changes...")
        origin.push()
    except:
        print("Some error occured while pushing the code")
        
def git_pull(PATH_OF_GIT_REPO=PATH_OF_GIT_REPO): # function to pull from github
    print("pulling repo...")
    repo = Repo(PATH_OF_GIT_REPO)
    origin = repo.remotes.origin
    origin.pull()
