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
	@echo "> BUILDING $<"
	$(LIB) $(LIBFLAGS) $^ -o $@
	@echo "< DONE BUILDING $<"

$(ODIR)/%.o: $(SDIR)/%.c
	@echo "> COMPILING: $<"
	$(CC)   $(CCFLAGS) -c $< -o $@
	@echo "< DONE COMPILING: $<"
$(ODIR)/%.o: $(SDIR)/%.cpp
	@echo "> COMPILING: $<"
	$(CXX)  $(CXXFLAGS) -c $< -o $@
	@echo "< DONE COMPILING: $<"
$(ODIR)/%.o: $(SDIR)/%.S
	@echo "> COMPILING: $<"
	$(ASMC) $(ASMFLAGS) -c $< -o $@
	@echo "< DONE COMPILING: $<"

.PHONY: clean
clean:
	@echo "> CLEANING"
	-rm -vf $(SHARED) $(OBJS)
	@echo "< DONE CLEANING"
