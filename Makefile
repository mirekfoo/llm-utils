help:
	@echo "Available targets:"
	@echo "  help                                     - Show this help message"
	@echo ""
	@echo "  llm-utils-self-install                   - Install llm-utils for development"
	@echo ""
	@echo "  mkdocs CMD=build|serve|gh-deploy         - [Build / Serve/ Deploy to GitHub Pages] web docs using mkdocs"
	@echo "  mkdocs-clean                             - Clean the web docs"
	@echo ""
	@echo "  mddocs-build                             - Build markdown docs using mddocs"
	@echo "  mddocs-clean                             - Clean the markdown docs"
	@echo "  mddocs-run                               - Run again mddocs to update docs"	
	@echo ""
	@echo "  bumpver LEVEL=major|minor|patch          - Bump version"
	
# --------------------------------------------------

STAMP = @if [ ! -d ".stamps" ]; then mkdir -p ".stamps"; fi && touch $@

# --------------------------------------------------

LLM_UTILS_SELF_INSTALL = .stamps/llm-utils-self-install.done

# install editable pyutils AFTER mkdocs-pyapi, mddocs to avoid unwanted pyutils reinstall due to github source-pinned dependency
$(LLM_UTILS_SELF_INSTALL): $(MKDOCS_INSTALL) $(MDDOCS_INSTALL)
	pip install -e .
	$(STAMP)

llm-utils-self-install: $(LLM_UTILS_SELF_INSTALL)

# --------------------------------------------------

PROJECT_SRC := $(wildcard src/llm_utils/*.py)

# --------------------------------------------------

MKDOCS_INSTALL = .stamps/mkdocs-install.done

$(MKDOCS_INSTALL):
	pip install git+https://github.com/mirekfoo/mkdocs-pyapi.git 
	$(STAMP)

MKDOCS_DIR = docs-web

$(MKDOCS_DIR):
	@if [ ! -d "$(MKDOCS_DIR)" ]; then mkdir -p "$(MKDOCS_DIR)"; fi

mkdocs: $(MKDOCS_INSTALL) $(MKDOCS_DIR)
	mkdocs-pyapi $(CMD)

mkdocs-clean:
	rm -f $(MKDOCS_DIR)/*.yml
	rm -rf docs-web-site

# --------------------------------------------------

MDDOCS_INSTALL = .stamps/mddocs-install.done

$(MDDOCS_INSTALL):
	pip install git+https://github.com/mirekfoo/mddocs.git 
	$(STAMP)

mddocs-install: $(MDDOCS_INSTALL)

# --------------------------------------------------

MDDOCS_DIR = docs-md

MDDOCS_GENERATE = .stamps/mddocs-generate.done

$(MDDOCS_GENERATE): $(MDDOCS_INSTALL) $(PROJECT_SRC)
	mddocs
	$(STAMP)

mddocs-build: \
	$(MDDOCS_GENERATE)

mddocs-clean:
	rm -rf $(MDDOCS_DIR)
	rm -f $(MDDOCS_GENERATE)

mddocs-run: \
	mddocs-clean \
	$(MDDOCS_GENERATE)

# --------------------------------------------------

BUMPVER_INSTALL = .stamps/bumpver-install.done

$(BUMPVER_INSTALL):
	pip install bumpver 
	$(STAMP)

bumpver: $(BUMPVER_INSTALL)
	bumpver update --$(LEVEL)
