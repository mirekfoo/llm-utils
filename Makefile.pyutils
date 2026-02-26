help:
	@echo "Available targets:"
	@echo "  help                                     - Show this help message"
	@echo ""
	@echo "  self-dev-install                         - Install pyutils for development"
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

THIS_PROJECT := pyutils
PROJECT_SRC := $(wildcard $(THIS_DIR)/src/$(THIS_PROJECT)/*.py)
	
# --------------------------------------------------

THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
THIS_DIR      := $(dir $(abspath $(THIS_MAKEFILE)))

ROOT_DIR ?= $(THIS_DIR)

# --------------------------------------------------

LOCAL_STAMP_DIR ?= $(THIS_DIR)/.stamps
LOCAL_STAMP = @if [ ! -d "$(LOCAL_STAMP_DIR)" ]; then mkdir -p "$(LOCAL_STAMP_DIR)"; fi && touch $@

ROOT_STAMP_DIR ?= $(ROOT_DIR)/.stamps
ROOT_STAMP = @if [ ! -d "$(ROOT_STAMP_DIR)" ]; then mkdir -p "$(ROOT_STAMP_DIR)"; fi && touch $@

# --------------------------------------------------

DEPS := 

define DEP_INSTALL_RULE
$(ROOT_STAMP_DIR)/$(1)-install:
	pip install $(1)
	$$(ROOT_STAMP)
endef

$(foreach d,$(DEPS),$(eval $(call DEP_INSTALL_RULE,$(d))))

#.PHONY: deps-install
deps-install: $(addprefix $(ROOT_STAMP_DIR)/,$(addsuffix -install,$(DEPS)))

# --------------------------------------------------

THIS_PROJECT_DEV_INSTALL = $(ROOT_STAMP_DIR)/$(THIS_PROJECT)-install

# install editable pyutils AFTER mkdocs-pyapi, mddocs to avoid unwanted pyutils reinstall due to github source-pinned dependency
$(THIS_PROJECT_DEV_INSTALL): $(MKDOCS_INSTALL) $(MDDOCS_INSTALL)
	pip install -e .
	$(ROOT_STAMP)

self-dev-install: deps-install $(THIS_PROJECT_DEV_INSTALL)

# --------------------------------------------------

MKDOCS_INSTALL = $(ROOT_STAMP_DIR)/mkdocs-install.done

$(MKDOCS_INSTALL):
	pip install git+https://github.com/mirekfoo/mkdocs-pyapi.git 
	$(ROOT_STAMP)

MKDOCS_DIR = $(THIS_DIR)/docs-web

$(MKDOCS_DIR):
	@if [ ! -d "$(MKDOCS_DIR)" ]; then mkdir -p "$(MKDOCS_DIR)"; fi

mkdocs: $(MKDOCS_INSTALL) $(MKDOCS_DIR)
	mkdocs-pyapi $(CMD)

mkdocs-clean:
	rm -f $(MKDOCS_DIR)/*.yml
	rm -rf docs-web-site

# --------------------------------------------------

MDDOCS_INSTALL = $(ROOT_STAMP_DIR)/mddocs-install.done

$(MDDOCS_INSTALL):
	pip install git+https://github.com/mirekfoo/mddocs.git 
	$(ROOT_STAMP)

mddocs-install: $(MDDOCS_INSTALL)

# --------------------------------------------------

MDDOCS_DIR = $(THIS_DIR)/docs-md

MDDOCS_GENERATE = $(ROOT_STAMP_DIR)/mddocs-generate.done

$(MDDOCS_GENERATE): $(MDDOCS_INSTALL) $(PROJECT_SRC)
	mddocs
	$(ROOT_STAMP)

mddocs-build: \
	$(MDDOCS_GENERATE)

mddocs-clean:
	rm -rf $(MDDOCS_DIR)
	rm -f $(MDDOCS_GENERATE)

mddocs-run: \
	mddocs-clean \
	$(MDDOCS_GENERATE)

# --------------------------------------------------

BUMPVER_INSTALL = $(LOCAL_STAMP_DIR)/bumpver-install.done

$(BUMPVER_INSTALL):
	pip install bumpver 
	$(LOCAL_STAMP)

bumpver: $(BUMPVER_INSTALL)
	bumpver update --$(LEVEL)
