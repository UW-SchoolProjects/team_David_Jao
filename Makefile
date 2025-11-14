# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -MMD -MP

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
TARGET = program

# Find all .cpp files under src/ (recursively)
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')

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
