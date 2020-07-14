CC=g++
CLTUNE_PATH?=/usr/local/CLTune
CLTUNE_INCLUDE?=$(CLTUNE_PATH)/include
CLTUNE_LIBRARY?=$(CLTUNE_PATH)/build

default:
	@echo "\033[93m[i]\033[0m Compiling... 🔨"
	@$(CC) -Wall -Wextra -Werror sort_tuner.cpp -o sort -I $(CLTUNE_INCLUDE) -Wl,-rpath $(CLTUNE_LIBRARY)  -L $(CLTUNE_LIBRARY) -lcltune
	@echo "\033[92m[✓]\033[0m Done compiling program! 🎉"

clean:
	@echo "\033[93m[i]\033[0m Cleaning up... 🗑️"
	@$(RM) sort
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉"
