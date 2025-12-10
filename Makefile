# Compiler and flags
# Set TARGET_OS=windows to build a 64-bit Windows executable with MinGW; defaults to native toolchain.
ifeq ($(TARGET_OS),windows)
	CXX = x86_64-w64-mingw32-g++
	EXE_EXT = .exe
	# Static link to ease distribution on Windows; remove -static if undesired.
	CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -MMD -MP -static -static-libstdc++ -static-libgcc
else
	CXX = g++
	EXE_EXT =
	CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -MMD -MP
endif

# Build type (optional: make BUILD=debug)
ifeq ($(BUILD),debug)
	CXXFLAGS += -g
else
	CXXFLAGS += -O2
endif

# Directories
SRC_DIR   = src
BUILD_DIR = build

# Output binary
TARGET = program$(EXE_EXT)

# Find all .cpp files under src/ (recursively), portable across shells
rwildcard = $(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
SRCS := $(call rwildcard,$(SRC_DIR)/,*.cpp)

# Turn e.g. src/main/engine.cpp -> build/main/engine.o
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# Default target
all: $(TARGET)

# Link step
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@

# Compile step (creates folders as needed)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Auto-include dependency files
-include $(DEPS)

# Cleanup
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
