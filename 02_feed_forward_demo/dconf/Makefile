SDIR :=.
ODIR :=.
IDIRS:=. ../inc

SHARED:=lib$(notdir $(realpath .)).so

SRCS:=$(shell find $(SDIR) -name *.cpp)

OBJS:=$(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(SRCS))
LIB_MEMBERS:=$(notdir $(OBJS))

INC:=$(addprefix -I,$(IDIRS))

CXX :=g++
LIB :=g++

WARNING :=-Wall -Wextra -pedantic -Wno-write-strings -Wno-unused-parameter
CXXFLAGS:=$(WARNING) $(INC) -fpic -std=c++14
LIBFLAGS:=-shared

.PHONY: all
all: $(SHARED)

$(SHARED): $(OBJS)
	@echo "> BUILDING $(notdir $(realpath .))"
	$(LIB) $(LIBFLAGS) $^ -o $@
	@echo "< BUILDING $(notdir $(realpath .)) DONE"

$(ODIR)/%.o: $(SDIR)/%.c
	@echo "> COMPILING: $<"
	$(CC)   $(CCFLAGS) -c $< -o $@
	@echo "< COMPILING: $< DONE"
$(ODIR)/%.o: $(SDIR)/%.cpp
	@echo "> COMPILING: $<"
	$(CXX)  $(CXXFLAGS) -c $< -o $@
	@echo "< COMPILING: $< DONE"
$(ODIR)/%.o: $(SDIR)/%.S
	@echo "> COMPILING: $<"
	$(ASMC) $(ASMFLAGS) -c $< -o $@
	@echo "< COMPILING: $< DONE"

.PHONY: clean
clean:
	@echo "> CLEANING $(notdir $(realpath .))"
	-rm -vf $(SHARED) $(OBJS) | sed -e 's/^/* /'
	@echo "< CLEANING $(notdir $(realpath .)) DONE"
