# ============================================================
# Toolchain selection
#   TARGET_OS=windows  -> cross-compile 64-bit Windows exe (MinGW-w64)
#   default (unset)    -> build native Linux binary
# ============================================================

ifeq ($(TARGET_OS),windows)
	CXX      = x86_64-w64-mingw32-g++
	EXE_EXT  = .exe
	# Static link to ease distribution on Windows; remove -static* if undesired.
	CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -MMD -MP \
	           -static -static-libstdc++ -static-libgcc
else
	CXX      = g++
	EXE_EXT  =
	CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -MMD -MP
endif

# Always use -p on this Linux host so nested dirs are created correctly
MKDIR_P = mkdir -p

# ============================================================
# Feature toggles
#   LOG=1       -> enable verbose engine logging
#   AUTO_PLAY=1 -> auto-play after user move
#   BUILD=debug -> add -g instead of -O2
# ============================================================

ifeq ($(LOG),1)
	CXXFLAGS += -DENGINE_LOGGING
endif

ifeq ($(AUTO_PLAY),1)
	CXXFLAGS += -DENGINE_AUTO_PLAY
endif

ifeq ($(BUILD),debug)
	CXXFLAGS += -g
else
	CXXFLAGS += -O2
endif

# ============================================================
# Directories and target
# ============================================================

SRC_DIR   = src
BUILD_DIR = build
TARGET    = program$(EXE_EXT)

# ============================================================
# Source discovery and object / dep mapping
# ============================================================

# Recursively find all .cpp files in SRC_DIR
rwildcard = $(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
SRCS      := $(call rwildcard,$(SRC_DIR)/,*.cpp)

# Map, e.g. src/core/board/board.cpp -> build/core/board/board.o
OBJS      := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS      := $(OBJS:.o=.d)

# ============================================================
# Default target
# ============================================================

all: $(TARGET)

# ============================================================
# Link step
# ============================================================

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@

# ============================================================
# Compile step (ensures directory for each .o exists)
# ============================================================

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ============================================================
# Dependencies
# ============================================================

-include $(DEPS)

# ============================================================
# Cleanup
# ============================================================

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
