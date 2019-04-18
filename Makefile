MAKE_PATHS:=$(dir $(shell find -mindepth 2 -maxdepth 2 -name Makefile))

.PHONY: all
all:
	+$(foreach PATH,$(MAKE_PATHS),make -e -C $(PATH);)

.PHONY: setup
setup:
	+$(foreach PATH,$(MAKE_PATHS),make -e -C $(PATH) setup;)

.PHONY: clean
clean:
	+$(foreach PATH,$(MAKE_PATHS),make -e -C $(PATH) clean;)
