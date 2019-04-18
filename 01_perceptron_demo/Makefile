IDIRS:=inc
SDIR :=src
ODIR :=obj
DDIR :=dep
BDIR :=bin
ELF  :=$(BDIR)/main.elf
HEX  :=$(BDIR)/main.hex
BIN  :=$(BDIR)/main.bin
ASM  :=$(BDIR)/main.asm
LDIR :=$(HOME)/Documents/proj/PC/libs
LIDIR:=$(LDIR)/inc
LADIR:=$(LDIR)/arc

LIBS:=
DEFS:=

######END#OF#CONFIGURATION#VARIABLES######

LIBS+=
DEFS+=$(EDEFS)

LIBS  :=$(addprefix -l,$(LIBS))
DEFS  :=$(addprefix -D,$(DEFS))

C_SRCS  :=$(shell find $(SDIR) -name *.c  )
CXX_SRCS:=$(shell find $(SDIR) -name *.cpp)
ASM_SRCS:=$(shell find $(SDIR) -name *.S  )

C_OBJS  :=$(patsubst $(SDIR)/%.c,  $(ODIR)/%.o,$(C_SRCS)  )
CXX_OBJS:=$(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(CXX_SRCS))
ASM_OBJS:=$(patsubst $(SDIR)/%.S,  $(ODIR)/%.o,$(ASM_SRCS))
OBJS:=$(C_OBJS) $(CXX_OBJS) $(ASM_OBJS)

IDIRS+=$(LIDIR)
IDIRS+=$(DCDIR)
INC:=$(addprefix -I,$(IDIRS))

CC  :=gcc
CXX :=g++
ASMC:=gcc
LD  :=g++

WARNING :=-Wall -Wextra -pedantic -Wno-write-strings
CCFLAGS :=$(WARNING) $(INC)  $(DEFS) -std=c99
CXXFLAGS:=$(WARNING) $(INC)  $(DEFS) -std=c++14
ASMFLAGS:=$(WARNING) $(INC)  $(DEFS)
LDFLAGS :=$(WARNING) -L$(LADIR) -L$(DCDIR) -Wl,-R$(DCDIR)

.SECONDARY: $(OBJS)

.PHONY: all
all: $(ELF)
	@echo "> BULDING"
	objcopy -j .data -j .text -O ihex   $(ELF) $(HEX)
	objcopy -j .data -j .text -O binary $(ELF) $(BIN)
	objdump -Dz $(ELF) > $(ASM)
	size $(HEX)
	@echo "< BUILDING DONE"

$(ELF): $(OBJS)
	@echo "> LINKING"
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@
	@echo "< LINKING DONE"

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

$(DDIR)/%.d: $(SDIR)/%.c
	@echo "> BUILDING DEPENDENCY: $<"
	$(CC)   $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCY: $< DONE"
$(DDIR)/%.d: $(SDIR)/%.cpp
	@echo "> BUILDING DEPENDENCY: $<"
	$(CXX)  $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCY: $< DONE"
$(DDIR)/%.d: $(SDIR)/%.S
	@echo "> BUILDING DEPENDENCY: $<"
	$(ASMC) $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCY: $< DONE"

.PHONY: setup
setup:
	@echo "> MAKING DIRECTORIES"
	mkdir -p $(BDIR) $(ODIR) $(DDIR) $(IDIRS) $(SDIR)
	touch $(BDIR)/.empty $(ODIR)/.empty $(DDIR)/.empty
	@echo "< MAKING DIRECTORIES DONE"

.PHONY: clean
clean:
	@echo "> CLEANING"
	-rm -vfr ./$(ODIR)/* ./$(DDIR)/* ./$(BDIR)/*
	+make -C $(DCDIR) clean
	@echo "< CLEANING DONE"

ifeq ($(NODEP),)
-include $(patsubst $(SDIR)/%.c,  $(DDIR)/%.d,$(C_SRCS)  )
-include $(patsubst $(SDIR)/%.cpp,$(DDIR)/%.d,$(CXX_SRCS))
-include $(patsubst $(SDIR)/%.S,  $(DDIR)/%.d,$(ASM_SRCS))
endif
