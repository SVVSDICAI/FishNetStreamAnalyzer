# git hub repo writer
# repo at https://github.com/SVVSDICAI/FishNetStreamCapture
from git import Repo
import sys

PATH_OF_GIT_REPO = r'./FishLadderStreamCapture'
COMMIT_MESSAGE = "automated update"

GITHUB_KEY = r'/github_key'

if len(sys.argv) > 1 and sys.argv[1] == "init": # if this file is run from the terminal (python RepoUpdate.py init) to initialize the github login info and clone the repo to the current directory
    print("cloning repo")
    # github login info
    username = "cogrpar"
    # read github password
    with open(GITHUB_KEY, 'r') as key:
    	password = key.read()
    	key.close()
    remote = f"https://{username}:{password}@github.com/SVVSDICAI/FishNetStreamCapture.git"
    remote = remote.replace("\n", "")
    print(remote)
    Repo.clone_from(remote, PATH_OF_GIT_REPO) # This must be run initially to ensure that the github login info is set

def git_push(): # function to push the updates to github
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add(update=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name="origin")
        print("pushing changes...")
        origin.push()
    except:
        print("Some error occured while pushing the code")
        
def git_pull(): # function to pull from github
    print("pulling repo...")
    repo = Repo(PATH_OF_GIT_REPO)
    origin = repo.remotes.origin
    origin.pull()
