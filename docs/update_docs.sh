#! /bin/bash
## -v|--version: snapshot version number
## -o|--offline: just generate documents and don't transfer to gh-pages

GEN_OFFLINE=false
# Default values of arguments
VERSION=NULL

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        -o|--offline)
        GEN_OFFLINE=true
        shift # Remove --initialize from processing
        ;;
        -v=*|--version=*)
        VERSION="${arg#*=}"
        shift # Remove --cache= from processing
        ;;
    esac
done

if [[ $(git branch | grep ${VERSION} 2>/dev/null | wc -l) -ne 1 ]]
then
  echo -e "\nNo matching snapshot version number: ${1}"
  echo -e "\nNeeds:"
  echo -e "\t\"-v=<#> or --version=<#>\" -  snapshot version number"
  echo -e "Optional:"
  echo -e "\t\"-o or --offline\" - if provided, documentation will be generated but not hosted on gh-pages"
  echo -e "\nEx. ~\$ ./update_docs.sh -v=0.0\n"
  exit
elif [[ $(pip freeze | egrep "Sphinx|sphinx-rtd-theme" 2>/dev/null | wc -l) -ne 2 ]]
then
  echo -e "\nPackage requirements not met"
  echo -e "\nNeeds:"
  echo -e "\t1 Sphinx: document generation software"
  echo -e "\t2 sphinx-rtd-theme: the readthedocs theme"
  echo -e "\nRerun the script after installing the above two packages\n"
  exit
fi

while getopts 'o' opt; do
    case $opt in
        -o|--offline) gen_offline_docs=true ;;
        :) echo "Missing argument for option -$OPTARG"; exit 1;;
       \?) echo "Unknown option -$OPTARG"; exit 1;;
    esac
done

echo "Generating documentation for $VERSION..."
git checkout $VERSION
SNAPSHOT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$SNAPSHOT_BRANCH" != "$VERSION" ]]; then
  echo 'Aborting script';
  exit 1;
fi
git pull
sphinx-apidoc -f -o source/ ../pymasq/
make clean html
echo "DONE"

if ${GEN_OFFLINE}; then
  exit 0;
fi
echo "Pushing generated source documents to $VERSION..."
git add source/
git commit -m "AUTOMATED COMMIT: document generation script generated new source files"
git push origin $VERSION
echo "DONE"

echo "Transferring generated files to gh-pages branch..."
cd ..
# Move the docs to the top-level directory, stash for checkout
mv docs/build/html ./
git checkout gh-pages
GH_PAGES_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$GH_PAGES_BRANCH" != "gh-pages" ]]; then
  echo 'Aborting script';
  exit 1;
fi
echo "DONE"

# Remove all files that are not in the .git dir and .gitignore
echo "Deleting previous documentation files..."
find . -not -name ".git*" -type f -maxdepth 1 -delete


# Remove the remaining artifacts. Some of these are artifacts of the
# LAST gh-pages build, and others are remnants of the package itself.
# You will have to amend this to be more specific to your project.
declare -a leftover=(".cache/"
                     ".idea/"
                     "build/"
                     "build_tools/"
                     "docs/"
                     "examples/"
                     "gh_doc_automation/"
                     "gh_doc_automation.egg-info/"
                     "_downloads/"
                     "_images/"
                     "_modules/"
                     "_sources/"
                     "_static/"
                     "includes"
                     "pymasq/")

# Check for each left over file/dir and remove it, or echo that it's
# not there.
for left in "${leftover[@]}"
do
    rm -r "$left" || echo "$left does not exist; will not remove"
done
echo "DONE"

# We need this empty file for git not to try to build a jekyll project.
echo "Moving files around..."
touch .nojekyll
mv html/* ./
rm -r html/
echo "DONE"

# Add everything, get ready for commit.
echo "Uploading documentation for $VERSION to gh-pages branch..."
git add --all
git commit -m "AUTOMATED COMMIT: creating documentation for snapshot version $VERSION"
git pull
git push origin gh-pages
echo "DONE"
